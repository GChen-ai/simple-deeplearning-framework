# Simple Deeplearning Framework

![](https://img.shields.io/badge/Bulid-Passing-brightgreen)![](https://img.shields.io/badge/Powered%20By-Geng%20Chen-brightgreen)

**一个使用numpy实现的简单的深度学习框架.**   [English](README_EN.md)

## 框架结构

### basic.py

提供了三个父类:Model,Loss,Optimizer

1. Model为神经网络各个层的父类,layers,activation中的类都继承该类.其中包含了forward和backward方法用于前向传播和反向传播.

2. Loss为损失函数的父类,具体的损失函数在loss中实现.Loss中的backward方法用于计算网络中各个层的梯度.

3. Optimizer为各种优化器的父类,优化器的实现在optimizer中.其中的step方法用于更新网络中各层的参数.



### layers.py

实现了神经网络中使用的基础结构,以及对应的前向传播,反向传播梯度计算方法.

已实现:

1. Linear
2. BN (1D, 2D)
3. LN
4. Conv2d
5. Maxpool2D
6. Avgpool2D



### activation.py

实现了各种激活函数和其对应的前向传播,反向传播方法.

已实现:

1. ReLu
2. SIgmoid
3. SIgmoidWithLogit
4. SoftMax
5. Tanh



### initializer.py

实现了参数初始化方法.

在各个层中的__initweight()中进行调用.

已实现:

1. Xavier
2. Kaiming
3. 正态分布
4. 常数初始化



### optimizer.py

实现了基础的优化器.

step()中会根据网络中各个层在反向传播时计算出的梯度来更新每个层的参数.

已实现:

1. SGD(带动量)
2. Adam
3. RMSprop
4. AdaGrad



### loss.py

实现了各种loss.

loss()以网络输出和标签作为输入,用于计算函数损失.

backward()用于进行反向传播,计算各个层的梯度.

已实现:

1. MSE
2. BCE
3. CrossEntropy



### function.py

此文件中包含了常用的函数.

如sigmoid, relu, softmax.以及在卷积计算中使用的img2col和col2img等.



**同时,在test文件中提供了简单的测试使用方法.使用了鸢尾花数据集训练和测试了一个简单的全连接网络.使用如下命令即可执行:**

````python
python test.py
````



**在dataloader中提供了手写体数字的数据加载代码.在mnist.py中在此数据集上训练和测试数据.使用如下命令即可执行:**

````python
python mnist.py
````

结果如图:

![mnist](/media/cg/d29c68c7-e9aa-4b54-97a1-2128070e19fd/PycharmProjects/simple-deeplearning-framework/result.png)

**如在代码和使用中发现了任何问题,希望能在issue中向我指出,非常感谢**