
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

path='processed_listings.csv'
file=pd.read_csv(path)
#print(file.columns)
labels=file.price.values.reshape(-1,1)
#print(labels[:5])
#归一化处理
my_min=min(labels)
my_max=max(labels)
my_mean=np.mean(labels)
label_list=[]
for label in labels:
    label=(label-my_mean)/(my_max-my_min)
    label_list.append(label.item())

#处理特征 一共51个特征 27750
features=pd.concat((file.iloc[:,2],file.iloc[:,8:11],file.iloc[:,17:]),axis=1)
feature_list=features.values
#print(feature_list[0])

room_type_dict={'Entire home/apt':0,'Private room':1,'Shared room':2}
for list in feature_list:
    list[0]=room_type_dict[list[0]]
    list[4]=0 if list[4]=='False' else 1
#print(feature_list[0])
    
#构建可用于训练的pytorch数据集
feature_list=np.array(feature_list[:27750],dtype=np.float32)
label_list=np.array(label_list[:27750],dtype=np.float32)

class MyDataset(Dataset):
    def __init__(self,feature_list,labels,device) -> None:
        super(MyDataset,self).__init__()
        self.feature_list=feature_list
        self.labels=labels
        self.device=device
    def __len__(self):
        return self.feature_list.shape[0]
    def __getitem__(self, index):
        return self.feature_list[index],self.labels[index]
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_len=feature_list.shape[0]
n_len=int(0.8*_len)
train_dataset=MyDataset(feature_list[:n_len],label_list[:n_len],device)
test_dataset=MyDataset(feature_list[n_len:],label_list[n_len:],device)
batch_size=64
train_data=torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
test_data=torch.utils.data.DataLoader(test_dataset,batch_size,shuffle=False)

#构建网络
class MLPnet(nn.Module):
    def __init__(self,num_features,num_hiddens):
        super(MLPnet,self).__init__()
        self.fc1=nn.Linear(num_features,2*num_hiddens,dtype=torch.float)
        self.fc2=nn.Linear(2*num_hiddens,num_hiddens)
        self.fc3=nn.Linear(num_hiddens,1)
    def forward(self,x):
        x=F.tanh(self.fc1(x))
        x=F.tanh(self.fc2(x))
        y=self.fc3(x)
        return y
    

num_epochs=50
lr=0.01
loss_f=nn.MSELoss()
net=MLPnet(feature_list.shape[1],8)

def train(num_epochs,train_data,loss_f,net,lr,device):
    net=net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    loss_list,test_list = [],[]
    for i in range(num_epochs):
        total_loss,total_test_loss = 0.0,0.0
        for x,y in train_data:
            x,y=x.to(device),y.to(device)
            pred=net(x)
            optimizer.zero_grad()
            loss=loss_f(pred,y)
            loss.backward()
            optimizer.step()
            iter_loss=torch.mean(loss).item()
            total_loss+=iter_loss
            loss_list.append(iter_loss)
        with torch.no_grad():
            for data, label in test_data:
                data, label = data.to(device), label.to(device)
                test_output = net(data)
                test_loss=torch.mean(loss_f(test_output,label)).item()
                total_test_loss += test_loss
                test_list.append(total_test_loss)
        print("epoch %d,loss is %f, test loss is %f" %
              (i + 1, total_loss/len(train_data), total_test_loss / len(test_data),))
    return loss_list,test_list


#绘制损失函数图像
def draw_image(loss_list,test_list,num_epoch):
    plt.figure()
    plt.plot(range(num_epoch*len(train_data)),loss_list,label='train_loss',c='blue')
    plt.plot(range(num_epoch*len(test_data)),test_list,label='test_loss',c='red')
    plt.legend()
    plt.show()


loss_list,test_list=train(num_epochs,train_data,loss_f,net,lr,device)
torch.save(net,'net_state.pth')
draw_image(loss_list,test_list,num_epochs)