from collections import OrderedDict
from src.basic import Optimizer
import numpy as np
class SGD(Optimizer):
    def __init__(self,model,lr=0.01,momentum=0.9):
        super().__init__(model)
        self.lr=lr
        self.momentum=momentum
        self.t=0
        self.v=OrderedDict()
        for i,layer in enumerate(self.model):
            for key,value in layer.params.items():
                self.v[key+str(i)]=np.zeros_like(value)

    def step(self):
        self.t+=1
        lr=self.lr/(1-self.momentum**self.t)
        for i,layer in enumerate(self.model):
            for key in layer.params.keys():
                self.v[key+str(i)]=self.momentum*self.v[key+str(i)]+(1-self.momentum)*layer.gradient[key]
                layer.params[key]-=lr*self.v[key+str(i)]

class Adam(Optimizer):
    def __init__(self,model,batch_size,lr=3e-4,beta1=0.9,beta2=0.999):
        super().__init__(model)
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.m=OrderedDict()
        self.v=OrderedDict()
        self.t=0
        for i,layer in enumerate(self.model):
            for key,value in layer.params.items():
                self.m[key+str(i)]=np.zeros_like(value)
                self.v[key+str(i)]=np.zeros_like(value)
        
    def step(self):
        self.t+=1
        lr=self.lr*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        for i,layer in enumerate(self.model):   
            for key in layer.params.keys():
                self.m[key+str(i)]=self.beta1*self.m[key+str(i)]+(1-self.beta1)*layer.gradient[key]

                self.v[key+str(i)]=self.beta2*self.v[key+str(i)]+(1-self.beta2)*(layer.gradient[key]**2)
                
                layer.params[key]-=lr*self.m[key+str(i)]/(np.sqrt(self.v[key+str(i)])+1e-8)

class RMSprop(Optimizer):
    def __init__(self,model,lr=1e-3,beta=0.99):
        super().__init__(model)
        self.lr=lr
        self.beta=beta
        self.r=OrderedDict()
        for i,layer in enumerate(self.model):
            for key,value in layer.params.items():
                self.r[key+str(i)]=np.zeros_like(value)
    def step(self):
        for i,layer in enumerate(self.model):
            for key in layer.params.keys():
                self.r[key+str(i)]=self.beta*self.r[key+str(i)]+(1-self.beta)*(layer.gradient[key]**2)

                layer.params[key]-=self.lr*layer.gradient[key]/(np.sqrt(self.r[key+str(i)]+1e-6))


class AdaGrad(Optimizer):
    def __init__(self,model,lr=1e-3):
        super().__init__(model)
        self.lr=lr
        self.r=OrderedDict()
        for i,layer in enumerate(self.model):
            for key,value in layer.params.items():
                self.r[key+str(i)]=np.zeros_like(value)
    def step(self):
        for i,layer in enumerate(self.model):
            for key in layer.params.keys():
                self.r[key+str(i)]=self.r[key+str(i)]+layer.gradient[key]**2

                layer.params[key]-=self.lr*layer.gradient[key]/(np.sqrt(self.r[key+str(i)])+1e-7)