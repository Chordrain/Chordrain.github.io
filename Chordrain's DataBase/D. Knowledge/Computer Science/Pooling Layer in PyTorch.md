#deepLearning #CS #python #pytorch 

# Pooling Layer in PyTorch

有关于池化操作的概念在这里就不赘述了，你可以简单理解为，池化 (一般) 是 stride=kernel_size 的卷积，只不过卷积公式不一样。常见的池化有最大池化 MaxPooling，平均池化 MeanPooling 等，本节以最大池化为例进行讲解。

>torch.nn.MaxPool2d (_kernel_size_, _stride=None_, _padding=0_, _dilation=1_, _return_indices=False_, _ceil_mode=False_)

池化层的主要参数如上所示。其中，我们需要关注的只有 kernel_size，其他的保持默认就好，但是有一个参数 —— ceil_mode 需要解释一下。ceil_mode 的意思是当图像的尺寸无法被 stride 整除时，剩下的部分是否舍弃，如果值为 False 就是舍弃。

最大池化就是选择感受域中的最大值来代表整个感受域，可以通过下面的例子来感受最大池化的作用：

```python
import torch
from torch.nn import MaxPool2d
from torch import nn

class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
], dtype=torch.float32)
input = torch.reshape(input, (1, 1, 5, 5))
pool = myModel()
output = pool.forward(input)
print(output)
```

输出结果：

```
tensor([[[[2., 3.],
          [5., 1.]]]])
```