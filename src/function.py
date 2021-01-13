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

def img2col(img,out_h,out_w,filter_h,filter_w,stride=1):
    N,C,H,W=img.shape
    out=np.zeros((N,C,filter_h*filter_w,out_h*out_w))
    for n in range(N):
        for c in range(C):
            hindex=0
            windex=0
            for k in range(out_h*out_w):
                if (windex+filter_w>W):
                    windex=0
                    hindex+=stride
                if (hindex+filter_h>H):
                    break
                out[n,c,:,k]=img[n,c,hindex:hindex+filter_h,windex:windex+filter_w].flatten()
                windex+=stride
    return out


def col2img(col,H,W,filter_h,filter_w,channel,padding,stride):
    #col shape:(batch, channel*filter_h*filter_w, outh*outw)
    N=col.shape[0]
    img=np.zeros((N,channel,H,W))
    col=col.reshape(N, channel, -1, col.shape[2])
    for n in range(N):
        for c in range(channel):
            windex=0
            hindex=0
            for k in range(col.shape[-1]):
                if (windex+filter_w>W):
                    windex=0
                    hindex+=stride
                if (hindex+filter_h>H):
                    break
                img[n,c,hindex:hindex+filter_h,windex:windex+filter_w]+=col[n,c,:,k].reshape((filter_h,filter_w))
                windex+=stride
    if padding>0:
        return img[:,:,padding:-padding,padding:-padding]
    else:
        return img
if __name__ == '__main__':
    img=np.array([[[[3,0,4],[6,5,4],[3,0,2]],
                    [[1,2,0],[3,0,2],[1,0,3]],
                    [[4,2,0],[1,2,0],[3,0,4]],
                    [[3,0,4],[6,5,4],[3,0,2]],
                    [[1,2,0],[3,0,2],[1,0,3]],
                    [[4,2,0],[1,2,0],[3,0,4]],
                    [[3,0,4],[6,5,4],[3,0,2]],
                    [[1,2,0],[3,0,2],[1,0,3]],
                    [[4,2,0],[1,2,0],[3,0,4]]]])
    kernel=np.ones((3,9,3,3))
    kernel=kernel.reshape(kernel.shape[0], -1)
    out=img2col(img,3,3,3,3,1)
    out=out.reshape((out.shape[0], -1, out.shape[3]))#不同的通道的内容进行拼接
    print(out.shape)
    out1=col2img(out,5,5,3,3,9,1,1)
    print(out1.shape)

