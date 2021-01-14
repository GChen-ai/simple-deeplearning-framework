import numpy as np
from src.layers import *
import torch
import torch.nn as nn
conv=Conv2d(1,1,3,3,1,1,initmethod='constant')

tconv=nn.Conv2d(1,1,3,1,1)
nn.init.constant_(tconv.weight, np.double(1.0))
nn.init.constant_(tconv.bias, np.double(0.0))
img=np.array([[[[3,0,4],[6,5,4],[3,0,2]]
                ]])

result=conv.forward(img)
print(result)
dx=conv.backward(2*(result-img)/9)
print(dx)

tensor_x=(torch.from_numpy(img)).double()
tensor_y=(torch.from_numpy(img)).double()
tensor_x.requires_grad=True
cri=nn.MSELoss()

out_torch=tconv(tensor_x)
ll=cri(out_torch,tensor_y)
ll.backward()
print(out_torch)
print(tconv.weight.grad)
print(tensor_x.grad)