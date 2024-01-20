import os
import pandas as pd
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


#处理原始数据
replace={"Baoshan":1,"Changning":2,"Hongkou":3,"Huangpu":4,"Jiading":5,"Jing'an":5,"Minhang":6,
         "Pudong":7,"Putuo":8,"Qingpu":9,"Xuhui":10,"Yangpu":11,"Zhabei":12}

PATH='data/for_tableau_2.csv'
file=pd.read_csv(PATH)
#print(file.head())

#所有因素：district,address,Latitude,Longitude,rent,bedrooms,living_dining,bathrooms,loft,
#sqmeters,entire_building,building_type,use_type_en,heat,ac,balcony,WIFI,outdoor_space,bathtub,floor_heat,oven,total_amens

#选取几个先验有关因素：district，bedrooms,dining-living bathrooms,loft,sqmeters,entire_building,buiding_type（是否临街），
#use_type(在预先处理时先把residential的找出来),heat,ac（空调）,balcony,WIFI,outdoor_space(?),floor_heat,oven,total_amens

#print(file.iloc[1],file.iloc[1][2])
all_features=file.iloc[:].values
#print(all_features[0])
len_features=len(all_features) #总共的数据数
labels=file.rent.values.reshape(-1,1) #最后要预测的出租价格
my_min=min(labels)
my_max=max(labels)
my_mean=np.mean(labels)
#print(my_max,my_min,my_mean)
std=np.std(labels)
#print(min,max)
#print(mean)
label_list=[]
#归一化
for label in labels:
    label=(label-my_mean)/(my_max-my_min)
    label_list.append(label.item())

#print(label_list[:10])

features=pd.concat((file.iloc[:,0],file.iloc[:,5:12],file.iloc[:,13:22]),axis=1)
feature_list=features.values
#print(feature_list[:5])
#将所有的区用数字替换，按拼音排序
for feature in feature_list:
    if feature[0] in replace:
        feature[0]=replace[feature[0]]
    else:
        feature[0]=-1
#print(feature_list[:5])
        
#将entire building这个特点替换成对应的数字 一共有四种类型
type_replace={"attached lane house":0,"detached lane house":1,"public housing":2,"villa":3}
for feature in feature_list:
    if feature[7] in type_replace:
        feature[7]=type_replace[feature[7]]
    else :
        feature[7]=-1
#print(feature_list[:5])
        
#接下来构建pytorch数据集
        
feature_list=np.array(feature_list,dtype=np.float32)
label_list=np.array(label_list,dtype=np.float32)
#print(feature_list.shape) 2608,17
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
dataset=MyDataset(feature_list,label_list,device)
batch_size=64
train_data=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)


#接下来开始训练并绘制损失函数曲线

#case 1:用最简单的一层MLP网络，相当于线性拟合
class MLPnet(nn.Module):
    def __init__(self,num_features):
        super(MLPnet,self).__init__()
        self.fc1=nn.Linear(num_features,1,dtype=torch.float)
    def forward(self,x):
        y=self.fc1(x)
        return y

#case 2:用多层网络
class MultiMLPnet(nn.Module):
    def __init__(self,num_features,num_hiddens):
        super(MultiMLPnet,self).__init__()
        self.fc1=nn.Linear(num_features,num_hiddens)

        self.fc3=nn.Linear(num_hiddens,1)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        y=self.fc3(x)
        return y

