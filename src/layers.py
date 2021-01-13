import numpy as np
from src.activation import*
from src.initializer import*
from src.basic import Model
from src.function import*
class Linear(Model):
    def __init__(self,insize,outsize,initmethod='xavier',mode='normal'):
        super().__init__()
        self.params['W']=None
        self.params['b']=None
        self.gradient['W']=None
        self.gradient['b']=None
        self.x=None
        self.__intweight(insize,outsize,initmethod,mode)
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
    def __intweight(self,insize,outsize,method,mode='normal'):
        if (method=='normal'):
            self.params['W'],self.params['b']=Normal1D(insize,outsize)
        elif (method=='constant'):
            self.params['W'],self.params['b']=Constant1D(insize,outsize)
        elif (method=='kaiming'):
            self.params['W'],self.params['b']=Kaiming1D(insize,outsize,mode)
        else:
            self.params['W'],self.params['b']=Xavier1D(insize,outsize,mode)

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
    def __init__(self,inchannel,outchannel,filter_h,filter_w,stride=1,padding=0,initmethod='xavier'):
        super().__init__()
        self.filter_h=filter_h
        self.filter_w=filter_w
        self.stride=stride
        self.padding=padding
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.imgCol=None
        self.params['W']=None
        self.params['b']=None
        self.gradient['W']=None
        self.gradient['b']=None
        self.__initweight(inchannel,outchannel,filter_h,filter_w,initmethod)
        self.H=None
        self.W=None
        self.C=None
        self.N=None
    def forward(self,img):
        N,C,H,W=img.shape
        img=np.pad(img,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        out_h=(H+2*self.padding-self.filter_h)//self.stride+1
        out_w=(W+2*self.padding-self.filter_w)//self.stride+1
        imgCol=img2col(img,out_h,out_w,self.filter_h,self.filter_w,self.stride)
        imgCol=imgCol.reshape((imgCol.shape[0],-1,imgCol.shape[3]))
        if not self.valid:
            self.imgCol=imgCol
            self.N=N
            self.C=C
            self.H=img.shape[2]
            self.W=img.shape[3]
        #不同通道内容进行拼接
        output=np.matmul(self.params['W'].reshape((self.params['W'].shape[0],-1)),imgCol)+self.params['b']
        return output.reshape((output.shape[0],output.shape[1],out_h,out_w))
    def backward(self,dout):
        self.gradient['b']=np.sum(dout,axis=(0,2,3)).reshape(-1)
        x_hat=self.imgCol.transpose(1,2,0).reshape((self.inchannel*self.filter_w*self.filter_h,-1))
        dout_reshape=dout.transpose(1,2,3,0).reshape((self.outchannel,-1))
        self.gradient['W']=np.matmul(dout_reshape,x_hat.T).reshape(self.params['W'].shape)
        dx=np.matmul(self.params['W'].reshape((self.outchannel,-1)).T,dout_reshape)
        dx=dx.reshape((dx.shape[0],-1,self.N)).transpose(2,0,1)
        dx=col2img(dx,self.H,self.W,self.filter_h,self.filter_w,self.C,self.padding,self.stride)
        #print(self.gradient['W'])
        return dx


    def __initweight(self,inchannel,outchannel,filter_h,filter_w,method,mode='normal'):
        if (method=='normal'):
            self.params['W'],self.params['b']=Normal2D(inchannel,outchannel,filter_h,filter_w)
        elif (method=='constant'):
            self.params['W'],self.params['b']=Constant2D(inchannel,outchannel,filter_h,filter_w)
        elif (method=='kaiming'):
            self.params['W'],self.params['b']=Kaiming2D(inchannel,outchannel,filter_h,filter_w,mode)
        else:
            self.params['W'],self.params['b']=Xavier2D(inchannel,outchannel,filter_h,filter_w,mode)




class Maxpool2D(Model):
    def __init__(self,filter_shape,padding=0):
        super().__init__()
        self.filter_shape=filter_shape
        self.padding=padding
        self.output=None
        self.loc=None
        self.H=None
        self.W=None
    def forward(self,x):
        N,C,H,W=x.shape
        x=np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        outh=(H+2*self.padding-self.filter_shape)//self.filter_shape+1
        outw=(W+2*self.padding-self.filter_shape)//self.filter_shape+1
        imgCol=img2col(x,outh,outw,self.filter_shape,self.filter_shape,self.filter_shape)
        output=np.max(imgCol,axis=2)
        if (not self.valid):
            self.loc=np.argmax(imgCol,axis=2)
            self.W=x.shape[3]
            self.H=x.shape[2]
        return output.reshape((N,C,outh,outw))
    
    def backward(self,dout):
        max_loc = np.zeros((dout.size,self.filter_shape**2))
        max_loc[np.arange(self.loc.size),self.loc.flatten()]= dout.flatten()
        max_loc = max_loc.reshape(dout.shape + (self.filter_shape**2,)).transpose(0,1,4,2,3)
        dcol=max_loc.reshape((max_loc.shape[0],max_loc.shape[1]*max_loc.shape[2],-1))
        dx=col2img(dcol,self.H,self.W,self.filter_shape,self.filter_shape,dout.shape[1],self.padding,self.filter_shape)
        return dx



class Avgpool2D(Model):
    def __init__(self,filter_shape,padding=0):
        super().__init__()
        self.filter_shape=filter_shape
        self.padding=padding
        self.output=None
        self.loc=None
        self.H=None
        self.W=None
    def forward(self,x):
        N,C,H,W=x.shape
        x=np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        outh=(H+2*self.padding-self.filter_shape)//self.filter_shape+1
        outw=(W+2*self.padding-self.filter_shape)//self.filter_shape+1
        imgCol=img2col(x,outh,outw,self.filter_shape,self.filter_shape,self.filter_shape)
        output=np.mean(imgCol,axis=2)
        if (not self.valid):
            self.loc=np.argmax(imgCol,axis=2)
            self.W=x.shape[3]
            self.H=x.shape[2]
        return output.reshape((N,C,outh,outw))
    
    def backward(self,dout):
        avg= np.expand_dims(dout.flatten()/(self.filter_shape**2),1).repeat(self.filter_shape**2,axis=1)
        avg = avg.reshape(dout.shape + (self.filter_shape**2,)).transpose(0,1,4,2,3)
        dcol=avg.reshape((avg.shape[0],avg.shape[1]*avg.shape[2],-1))
        dx=col2img(dcol,self.H,self.W,self.filter_shape,self.filter_shape,dout.shape[1],self.padding,self.filter_shape)
        return dx
