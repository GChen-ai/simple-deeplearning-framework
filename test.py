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
    #train_y=train_data[:, 4].astype(np.int64)
    train_y = np.eye(3)[train_data[:, 4].astype(np.int64)]
    test_x = test_data[:, :4]
    #test_y=test_data[:, 4].astype(np.int64)
    test_y = np.eye(3)[test_data[:, 4].astype(np.int64)]

    return train_x, train_y, test_x, test_y
# Data sets
X, y, test_x, test_y=get_data()

layers=[Linear(4,8),LayerNormalization(8),Sigmoid(),Linear(8,3),SoftMax()]
batch_size=16
lr=0.03
opt=SGD(layers,lr=lr)
loss=CrossEntropyLoss(layers,batch_size)
num_train, dim = X.shape
num_iters_per_epoch = int(math.floor(1.0*num_train/batch_size))
loss_sum=0
count=0
for epoch in range(1000):
    perm_idx = np.random.permutation(num_train)
    # perform mini-batch SGD update
    for it in range(num_iters_per_epoch):
        idx = perm_idx[it*batch_size:(it+1)*batch_size]
        batch_x = X[idx]
        batch_y = y[idx]
        for l in layers:
            l.train()
            batch_x=l.forward(batch_x)
        loss_sum+=loss.loss(batch_x,batch_y)
        count+=1
        loss.backward()
        opt.step()
        #print(batch_x.shape)
        #batch_x=batch_x.reshape(-1)
        acc_train=np.mean(np.argmax(batch_y, axis=1)==np.argmax(batch_y, axis=1))
        #print('loss: %.3f  acc: %.3f' %(loss_sum/count,acc_train))
    if epoch%100==0:
        tx=test_x
        for l in layers:
            l.val()
            tx=l.forward(tx)
        #print(tx)
        #print(test_y)
        acc=np.mean(np.argmax(tx, axis=1)==np.argmax(test_y, axis=1))
        #acc=np.mean(tx==test_y)
        print('test acc: %.3f' %(acc))
