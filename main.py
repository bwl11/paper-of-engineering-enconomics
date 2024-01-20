from torch import optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data_processing import MLPnet,feature_list,MultiMLPnet,train_data


device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_epochs=20
lr=0.001
loss_f=nn.MSELoss()
net=MLPnet(feature_list.shape[1])
net1=MultiMLPnet(feature_list.shape[1],8)

def train(num_epochs,train_data,loss_f,net,lr,device):
    net=net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_list = []
    for i in range(num_epochs):
        total_loss = 0.0
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
        print("epoch %d: loss is %f" % (i+1,total_loss/len(train_data)))
    return loss_list


#绘制损失函数图像
def draw_image(loss_list,num_epoch):
    plt.figure()
    plt.plot(range(num_epoch*len(train_data)),loss_list)
    plt.show()


loss_list=train(num_epochs,train_data,loss_f,net,lr,device)
torch.save(net,'data/1mlp_state.pth')
torch.save(net.state_dict,'1state.pth')
#draw_image(loss_list,num_epochs)