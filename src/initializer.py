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