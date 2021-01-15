# Simple Deeplearning Framework

![](https://img.shields.io/badge/Bulid-Passing-brightgreen)![](https://img.shields.io/badge/Powered%20By-Geng%20Chen-brightgreen)

**A simple deep learning framework implemented by Numpy.**   [中文](README.md)

## Framework

### basic.py

Three basic classes are provided: Model,Loss,Optimizer

1. Model is the parent class of each layer of the neural network. Classes in layers.py and activation.py inherit from this class. It contains forward and backward methods.
2. Loss is the parent class of Loss function, and the specific Loss function is implemented in loss.py. Backward method in loss is used to calculate the gradient of each layer in the network.
3. Optimizer is the parent class of the various optimizers. The implementation of the Optimizer is in the optimizer.py. The step method is used to update the parameters of each layer in the network.



### layers.py

This file implements the basic structure used in the neural network and the corresponding forward propagation and back propagation gradient calculation methods are implemented.

**Finished:**

1. Linear
2. BN (1D, 2D)
3. LN
4. Conv2d
5. Maxpool2D
6. Avgpool2D



### activation.py

This file implements various activation functions and their corresponding forward propagation and back propagation methods.

**Finished:**

1. ReLu
2. SIgmoid
3. SIgmoidWithLogit
4. SoftMax
5. Tanh



### initializer.py

This file implements the parameter initialization methods called in the initialization function.

**Finished:**

1. Xavier
2. Kaiming
3. Normal
4. Constant



### optimizer.py

This file implements the basic optimizers.

The step() method updates the parameters of each layer of the network according to the gradient calculated during the back propagation.

**Finished:**

1. SGD(带动量)
2. Adam
3. RMSprop
4. AdaGrad



### loss.py

This file implements various loss classed.

The loss() method takes the network outputs and the label as input to calculate the loss.

The backward () method is used for backpropagation to calculate the gradient for each layer.

**Finished:**

1. MSE
2. BCE
3. CrossEntropy



### function.py

This file contains commonly used functions.

**FInished:**

1. sigmoid
2. softmax
3. tanh
4. relu
5. img2col
6. col2img



**The test.py provides a simple way to build a network. A simple fully connected network was trained and tested using the Iris dataset. Use the following command to execute :**

````python
python test.py
````



**Handwritten numeric data loading code is provided in DataLoader. Train and test the data on this dataset in mnist.py. Use the following command to execute :**

````python
python mnist.py
````

The result:

![mnist](/media/cg/d29c68c7-e9aa-4b54-97a1-2128070e19fd/PycharmProjects/simple-deeplearning-framework/result.png)

**If you find any problems in the code, please kindly point them out to me in the issue. Thank you very much **