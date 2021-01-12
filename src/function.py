import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x,axis=1):
    max_x=np.max(x,axis=axis).reshape(-1,1)
    x=x-max_x
    exp=np.exp(x)
    sum_exp=np.sum(exp,axis=axis,keepdims=True)
    return exp/sum_exp

def relu(x):
    x[x<0]=0
    return x

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def img2col(img,filter_h,filter_w,stride=1,padding=0):
    N,C,H,W=img.shape
    out_h=(H+2*padding-filter_h)//stride+1
    out_w=(W+2*padding-filter_w)//stride+1
    img=np.pad(img,((0,0),(0,0),(padding,padding),(padding,padding)))
    out=np.zeros((N,C,filter_h,filter_w,out_h,out_w))
    print(out.shape)
    for i in range(filter_h):
        h_max=i+stride*filter_h
        for j in range(filter_w):
            w_max=j+stride*filter_w
            out[:, :, i, j, :, :] = img[:, :, i:h_max:stride, j:w_max:stride]
    out=out.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    print(out.shape)
    return out

img=np.array([[[[3,0,4],[6,5,4],[3,0,2]],
                [[1,2,0],[3,0,2],[1,0,3]],
                [[4,2,0],[1,2,0],[3,0,4]],
                [[3,0,4],[6,5,4],[3,0,2]],
                [[1,2,0],[3,0,2],[1,0,3]],
                [[4,2,0],[1,2,0],[3,0,4]],
                [[3,0,4],[6,5,4],[3,0,2]],
                [[1,2,0],[3,0,2],[1,0,3]],
                [[4,2,0],[1,2,0],[3,0,4]]]])
kernel=np.ones((3,3,3))
k=img2col(kernel)
print(img2col(img,3,3,1,1))


