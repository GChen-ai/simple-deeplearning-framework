from src.layers import*
import numpy as np
pool=Avgpool2D(2)
img=np.array([[[[3,0,4,2],[6,5,4,1],[3,0,2,2],[1,1,1,1]]
                ],
                [[[3,0,4,2],[6,5,4,1],[3,0,2,2],[1,1,1,1]]
                ]])
print(img.shape)
out=pool.forward(img)
print(out)
back=pool.backward(out)