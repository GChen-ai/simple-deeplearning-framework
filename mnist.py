from dataloader.loadmnist import load_mnist
import numpy as np
from src.layers import*
from src.optimizer import*
from src.loss import *
from src.activation import*
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
y_test=np.eye(10)[y_test]
y_train=np.eye(10)[y_train]
max_iter=5000
batch_size=16
model=[Conv2d(1,3,3,3,1,1),ReLu(),Maxpool2D(2),Conv2d(3,8,3,3,1,1),ReLu(),Maxpool2D(2),Conv2d(8,10,3,3,1,1),ReLu(),Avgpool2D(7),SoftMax2D()]
lr=0.03
opt=SGD(model,lr=lr)
loss=CrossEntropyLoss(model,batch_size)
loss_sum=0
count=0
for i in range(max_iter):
    mask=np.random.choice(x_train.shape[0],16)
    x_batch=x_train[mask]
    y_batch=y_train[mask]
    
    for l in model:
        l.train()
        x_batch=l.forward(x_batch)
        #print(x_batch.shape)
    loss_sum+=loss.loss(x_batch,y_batch)
    count+=1
    loss.backward()
    opt.step()
    #print(batch_x.shape)
    #batch_x=batch_x.reshape(-1)
    #print(x_batch)
    #print(y_batch)
    acc_train=np.mean(np.argmax(x_batch, axis=1)==np.argmax(y_batch, axis=1))
    #if (i%100==0):
        #print('loss: %.3f  acc: %.3f' %(loss_sum/count,acc_train))
    
    if i%100==0:
        acc=0
        start=0
        num=0
        for k in range(16,len(x_test),16):
            tx=x_test[start:k]
            for l in model:
                l.val()
                tx=l.forward(tx)
            #print(tx)
            #print(y_test)
            #print(np.argmax(tx,axis=1))
            acc+=np.sum(np.argmax(tx,axis=1)==np.argmax(y_test[start:k],axis=1))
            num+=tx.shape[0]
            #acc=np.mean(tx==test_y)
            start=k
        print('test acc: %.3f' %(acc/num))