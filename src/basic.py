import numpy as np
from collections import OrderedDict
class Model():
    def __init__(self,valid=False):
        self.valid=valid
        self.params=OrderedDict()
        self.gradient=OrderedDict()
    def forward(self):
        pass
    def backward(self):
        pass
    def val(self):
        self.valid=True
    def train(self):
        self.valid=False
    def __intweight(self):
        pass


class Loss():
    def __init__(self,model,batch_size):
        self.model=model
        self.y=None
        self.label=None
        self.batch_size=batch_size
    def loss(self,y,label):
        pass
    def backward(self):
        pass

class Optimizer():
    def __init__(self,model):
        self.model=model
    def step(self):
        pass
    def zero_grad(self):
        pass