import numpy as np
from src.activation import*
from src.initializer import*
from src.basic import Model
class Linear(Model):
    def __init__(self,insize,outsize,initmethod='xavier'):
        super().__init__()
        self.params['W']=None
        self.params['b']=None
        self.gradient['W']=None
        self.gradient['b']=None
        self.x=None
        self.__intweight(insize,outsize,initmethod)
        self.out=None
    def forward(self,x):
        out=np.dot(x,self.params['W'])+self.params['b']
        if (not self.valid):
            self.x=x
            self.out=out
        return out
    def backward(self,dout=1):
        dx=np.dot(dout,self.params['W'].T)
        self.gradient['W']=np.dot(self.x.T,dout)
        self.gradient['b']=np.sum(dout,axis=0)
        return dx
    def __intweight(self,insize,outsize,method):
        if (method=='normal'):
            self.params['W'],self.params['b']=Normal1D(insize,outsize)
        else:
            self.params['W'],self.params['b']=Xavier1D(insize,outsize)

class BatchNormalization(Model):
    def __init__(self,insize,momentum=0.9,eps=1e-6):
        super().__init__()
        self.params['beta']=None
        self.params['gamma']=None
        self.gradient['beta']=None
        self.gradient['gamma']=None
        self.x=None
        self.xhat=None
        self.train_mean=0
        self.train_var=0
        self.momentum=momentum
        self.eps=eps
        self.batch_mean=0
        self.batch_var=0
        self.__initweight(insize)
    def forward(self,x):
        #训练和测试需要分开,测试时使用的是训练时滑动平均得到的mean和var
        if (not self.valid):
            batch_mean=np.mean(x,axis=0)
            batch_var=np.var(x,axis=0)
            self.batch_mean=batch_mean
            self.batch_var=batch_var
            self.x=x
            x=(x-batch_mean)/(np.sqrt(batch_var+self.eps))
            self.xhat=x
            self.train_mean=self.momentum*self.train_mean+(1-self.momentum)*batch_mean
            self.train_var=self.momentum*self.train_var+(1-self.momentum)*batch_var
            out=x*self.params['gamma']+self.params['beta']
            return out
        else:
            return (x-self.train_mean)*self.params['gamma']/(np.sqrt(self.train_var+self.eps))+self.params['beta']

    def backward(self,dout):
        self.gradient['beta']=np.sum(dout,axis=0)
        self.gradient['gamma']=np.sum(dout*self.xhat,axis=0)

        dxhat=self.params['gamma']*dout
        dvar=np.sum(dxhat*(self.x-self.batch_mean)/(-2*(self.batch_var+self.eps)**1.5),axis=0)

        I=dxhat/np.sqrt(self.batch_var+self.eps)+dvar*2*(self.x-self.batch_mean)/self.x.shape[0]
        dx=I-np.sum(I,axis=0)/self.x.shape[0]
        return dx

    def __initweight(self,insize):
        self.params['gamma']=np.ones(insize)
        self.params['beta']=np.zeros(insize)

class LayerNormalization(Model):
    def __init__(self,insize,eps=1e-6):
        super().__init__()
        self.params['beta']=None
        self.params['gamma']=None
        self.gradient['beta']=None
        self.gradient['gamma']=None
        self.x=None
        self.xhat=None
        self.eps=eps
        self.batch_mean=0
        self.batch_var=0
        self.__initweight(insize)
    def forward(self,x):
        batch_mean=np.mean(x,axis=1).reshape((x.shape[0],1))
        batch_var=np.var(x,axis=1).reshape((x.shape[0],1))
        xhat=(x-batch_mean)/(np.sqrt(batch_var+self.eps))
        if (not self.valid):
            self.batch_mean=batch_mean
            self.batch_var=batch_var
            self.x=x
            self.xhat=xhat
        out=xhat*self.params['gamma']+self.params['beta']
        return out
        
    def backward(self,dout):
        self.gradient['beta']=np.sum(dout,axis=0)
        self.gradient['gamma']=np.sum(dout*self.xhat,axis=0)

        dxhat=self.params['gamma']*dout
        dvar=np.sum(dxhat*(self.x-self.batch_mean)/(-2*(self.batch_var+self.eps)**1.5),axis=1,keepdims=True)

        I=np.sum(dxhat/np.sqrt(self.batch_var+self.eps),axis=1,keepdims=True)+np.sum(dvar*2*(self.x-self.batch_mean),axis=1,keepdims=True)/self.x.shape[1]
        dx=dxhat/np.sqrt(self.batch_var+self.eps)+dvar*2*(self.x-self.batch_mean)/self.x.shape[1]-I/self.x.shape[1]
        return dx

    def __initweight(self,insize):
        self.params['gamma']=np.ones(insize)
        self.params['beta']=np.zeros(insize)


class Conv2d(Model):
    def __init__(self,filter_h,filter_w,stride=1,padding=0):
        super().__init__()
        self.filter_h=filter_h
        self.filter_w=filter_w
        self.stride=stride
        self.padding=padding
    def forward(self,img):


    def backward(self):

    def __initweight(self):