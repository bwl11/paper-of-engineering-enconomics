import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

net=torch.load('data/1mlp_state.pth')
for name,param in net.named_parameters():
    if 'weight' in name:
        a=param
#a.tolist()即为各权重大小
print(a.tolist())