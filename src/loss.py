import numpy as np
from src.activation import*
from src.basic import Loss

class MSELoss(Loss):
    def __init__(self,model,batch_size):
        super().__init__(model,batch_size)
        self.label_shape=None
    def loss(self,y,label):
        self.y=y
        self.label=label.reshape(y.shape)
        self.label_shape=self.y.size
        return np.mean((self.y-self.label)**2)
    def backward(self):
        dout=2*(self.y-self.label)/self.label_shape
        for l in self.model[::-1]:
            dout=l.backward(dout)

class BCELoss(Loss):
    def __init__(self,model,batch_size):
        super().__init__(model,batch_size)
    def loss(self,y,label):
        self.y=y
        self.label=label.reshape(y.shape)
        return -np.sum(self.label*np.log(self.y+1e-7)+(1-self.label)*np.log(1-self.y+ 1e-7))/self.batch_size

    def backward(self):
        dout=self.y
        for l in self.model[::-1]:
            if (l==self.model[-1]):
                dout=l.backward(dout,last=True,label=self.label)/self.batch_size
            else:
                dout=l.backward(dout)

class CrossEntropyLoss(Loss):
    def __init__(self,model,batch_size):
        super().__init__(model,batch_size)
    def loss(self,y,label):
        self.y=y
        self.label=label.reshape(y.shape)
        return -np.sum(self.label*np.log(self.y+1e-7))/self.batch_size
    def backward(self):
        dout=self.y
        for l in self.model[::-1]:
            if (l==self.model[-1]):
                dout=l.backward(dout,last=True,label=self.label)/self.batch_size
            else:
                dout=l.backward(dout)
        
    