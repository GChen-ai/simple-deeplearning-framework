import numpy as np

def Xavier1D(insize,outsize,mode='uniform'):
    if mode=='normal':
        mu=np.sqrt(2/(insize+outsize))
        w=np.random.normal(0,mu,(insize,outsize))
        b=np.zeros(outsize)
    else:
        var=np.sqrt(6/(insize+outsize))
        w=np.random.uniform(-var,var,(insize,outsize))
        b=np.zeros(outsize)
    return w,b

def Normal1D(insize,outsize):
    w=np.random.normal(0,1,(insize,outsize))
    b=np.zeros(outsize)
    return w,b

def Constant1D(insize,outsize):
    w=np.ones((insize,outsize))
    b=np.zeros(outsize)
    return w,b

def Kaiming1D(insize,outsize,a=0,mode='normal'):
    if mode=='uniform':
        var=np.sqrt(6/(insize*(1+a**2)))
        w=np.random.uniform(-var,var,(insize,outsize))
        b=np.zeros(outsize)
    else:
        mu=np.sqrt(2/(insize*(1+a**2)))
        w=np.random.normal(0,mu,(insize,outsize))
        b=np.zeros(outsize)
    return w,b


def Xavier2D(inchannel,outchannel,filter_h,filter_w,mode='uniform'):
    if mode=='normal':
        mu=np.sqrt(2/(inchannel+outchannel))
        w=np.random.normal(0,mu,(outchannel,inchannel,filter_h,filter_w))
        b=np.zeros((outchannel,1))
    else:
        var=np.sqrt(6/(inchannel+outchannel))
        w=np.random.uniform(-var,var,(outchannel,inchannel,filter_h,filter_w))
        b=np.zeros((outchannel,1))
    return w,b

def Normal2D(inchannel,outchannel,filter_h,filter_w):
    w=np.random.normal(0,1,(outchannel,inchannel,filter_h,filter_w))
    b=np.zeros((outchannel,1))
    return w,b

def Kaiming2D(inchannel,outchannel,filter_h,filter_w,a=0,mode='normal'):
    if mode=='uniform':
        var=np.sqrt(6/(inchannel*(1+a**2)))
        w=np.random.uniform(-var,var,(outchannel,inchannel,filter_h,filter_w))
        b=np.zeros((outchannel,1))
    else:
        mu=np.sqrt(2/(inchannel*(1+a**2)))
        w=np.random.normal(0,mu,(outchannel,inchannel,filter_h,filter_w))
        b=np.zeros((outchannel,1))
    return w,b

def Constant2D(inchannel,outchannel,filter_h,filter_w):
    w=np.ones((outchannel,inchannel,filter_h,filter_w))
    b=np.zeros((outchannel,1))
    return w,b