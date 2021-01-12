import numpy as np
from src.function import*
from src.basic import*
class ReLu(Model):
    def __init__(self):
        super().__init__()
        self.NegMask=None
        self.out=None
    def forward(self,x):
        self.NegMask=(x>0)
        out=x*self.NegMask
        if (not self.valid):
            self.out=out
        return out
    def backward(self,dout,last=False,label=None):
        if last:
            return -dout*self.NegMask
        else:
            return dout*self.NegMask
        
class Sigmoid(Model):
    def __init__(self):
        super().__init__()
        self.x=None
        self.out=None
    def forward(self,x):
        x=sigmoid(x)
        if (not self.valid):
            self.out=x
        return x
    def backward(self,dout,last=False,label=None):
        if last:
            return dout-label.reshape(dout.shape)
        else:
            return dout*(1-self.out)*self.out

class SigmoidWithLogit(Model):
    def __init__(self):
        super().__init__()
        self.x=None
        self.out=None
    def forward(self,x):
        x=sigmoid(x)
        if (not self.valid):
            self.out=x
        return x
    def backward(self,dout,last=True,label=None):
        return dout-label.reshape(dout.shape)

class SoftMax(Model):
    def __init__(self,axis=1):
        super().__init__()
        self.x=None
        self.out=None
        self.axis=axis
    def forward(self,x):
        max_num=np.max(x,axis=self.axis).reshape(-1,1)
        x=x-max_num
        self.x=x
        exp=np.exp(x)
        sum_exp=np.sum(exp,axis=self.axis,keepdims=True)
        out=exp/sum_exp
        if (not self.valid):
            self.out=out
        return out
    def backward(self,dout,last=False,label=None):
        if (last):
            return dout-label.reshape(dout.shape)
        else:
            E=np.expand_dims(np.eye(dout.shape[1]),0).repeat(dout.shape[0],axis=0)
            x=np.expand_dims(self.x,2)
            return np.squeeze(np.matmul((E-x).transpose((0,2,1)),x))


class Tanh(Model):
    def __init__(self):
        super().__init__()
        self.x=None
        self.out=None
    def forward(self,x):
        x=tanh(x)
        if (not self.valid):
            self.out=x
        return x
    def backward(self,dout,last=False,label=None):
        if not last:
            return dout*(1-self.out**2)