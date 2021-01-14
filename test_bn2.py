from src.layers import*
from src.loss import*
from src.optimizer import*
from src.activation import*
import torch.nn as nn
import torch
import numpy as np
import os
import math

# Data sets
x=np.array([[[[3,0,0],[6,5,0],[3,0,0]],
            [[-2,3,4],[-1,2,4],[3,1,2]]
                ],
            [[[2,0,1],[1,2,3],[3,2,1]],
            [[2,5,1],[1,-2,3],[3,0,1]]
                ]])
tensor_x=(torch.from_numpy(x)).float()
tensor_y=(torch.from_numpy(np.zeros_like(x))).float()
print(tensor_x.shape)
tensor_x.requires_grad=True
bn=nn.BatchNorm2d(2,momentum=0).float()
nn.init.constant_(bn.weight, np.float(1.0))
nn.init.constant_(bn.bias, np.float(0.0))
optimizer=torch.optim.SGD(bn.parameters(),lr=1,momentum=0)
cri=nn.MSELoss()
out=bn(tensor_x)
ll=cri(out,tensor_y)
ll.backward()
print('torch loss:')
print(ll.item())
print('torch bn out:')
print(out)
#print(bn.bias.grad)
print('torch bn grad:')
print(tensor_x.grad)
print(bn.weight.grad)
print(bn.bias.grad)

bn2=BatchNormalization2D(2,momentum=0)
opt=SGD([bn2],lr=1,momentum=0)
loss=MSELoss([bn2],x.shape[0])
out=bn2.forward(x)
print('loss:')
print(loss.loss(out,np.zeros_like(x)))
print('out')
print(out)
print('grad:')
loss.backward()