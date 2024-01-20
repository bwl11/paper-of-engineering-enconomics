import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processing import my_min,my_mean,my_max


conditions=['上海市哪个区','卧室数量','客厅数量','浴室数量','是否为复式楼','面积','是否为整栋',
           '是否临街','有无暖气','空调数量','有无阳台','有无wifi','是否带院子','浴缸数','有无地暖',
           '有无烤箱']
feature_list=[]
print('提示：“有无”的问题0代表无，1代表有')
for condition in conditions:
    if condition=='上海市哪个区':
        a=input('该房在上海市哪个区？提示：Baoshan:1,Changning:2,Hongkou:3,Huangpu:4,Jiading:5,'
                'Jing an:5,Minhang:6,'
                'Pudong:7,Putuo:8,Qingpu:9,Xuhui:10,Yangpu:11,Zhabei:12   ')
        feature_list.append(int(a))
    elif condition=='是否临街':
        a=input('是否临街。提示：临街：0，不临街：1，public housing：2，在乡村：3  ')
        feature_list.append(int(a))
    else:
        a=input(condition+'?  ')
        feature_list.append(int(a))
b=sum(feature_list[-8:])
feature_list.append(b)
#print(feature_list)
feature_list=torch.tensor(feature_list,dtype=torch.float32)
feature_list=feature_list.view(1,-1)
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net1=torch.load('data/2mlp_state.pth')
pred=net1(feature_list.to(device))
#print(pred.cpu().item())
pred=(pred.cpu().item())*(my_max.item()-my_min.item())+my_mean
print('预测价格是：',pred)