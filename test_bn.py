from src.layers import*
from src.loss import*
from src.optimizer import*
from src.activation import*
import torch.nn as nn
import torch
import numpy as np
import os
import math
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

def get_data():
    # Load datasets.
    train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
        dtype=float, delimiter=',') 
    test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
        dtype=float, delimiter=',')
    #train_data=train_data[train_data[:,-1]!=2]
    #test_data=test_data[test_data[:,-1]!=2]
    train_x = train_data[:, :4]
    train_y = np.eye(3)[train_data[:, 4].astype(np.int64)]
    test_x = test_data[:, :4]
    test_y = np.eye(3)[test_data[:, 4].astype(np.int64)]

    return train_x, train_y, test_x, test_y
# Data sets
X, y, test_x, test_y=get_data()
tensor_x=(torch.from_numpy(test_x[3:5])).double()
tensor_y=(torch.from_numpy(np.zeros_like(test_x[3:5]))).double()
print(tensor_x.shape)
tensor_x.requires_grad=True
bn=nn.LayerNorm(4).double()
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

bn2=LayerNormalization(4)
opt=SGD([bn2],lr=1,momentum=0)
loss=MSELoss([bn2],test_x[3:5].shape[0])
out=bn2.forward(test_x[3:5])
print('loss:')
print(loss.loss(out,np.zeros_like(test_x[3:5])))
print('out')
print(out)
print('grad:')
loss.backward()