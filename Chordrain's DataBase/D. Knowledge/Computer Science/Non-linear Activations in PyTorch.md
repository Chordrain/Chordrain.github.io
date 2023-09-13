#deepLearning #python #pytorch #CS 

# Non-linear Activations in PyTorch

我们熟知或不熟知的激活函数也已经作为模块封装在 `torch.nn` 下，其使用的方法和 layer 是一样的，都是在模型的 `init` 函数中挂载到模型上。由于其只是一个函数，所以定义时也不需要什么参数。

>torch.nn.ReLU (_inplace=False_)

以 ReLU 函数为例，其只有一个参数 **inplace**，意思是是否进行原地操作，改变传入数据的值。

当传入的数据是一个多维的 tensor 时，ReLU 会对 tensor 中的每一个元素进行变换。ReLU 的函数表达式如下：

$$
\operatorname{ReLU}(x)=(x)^+=\operatorname{max}(0,x)
$$

下面是一个 `nn.Relu` 的使用案例：

```python
import torch
from torch.nn import MaxPool2d
from torch import nn

class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
    
    def forward(self, input):
        output = self.relu1(input)
        return output

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, -1, 2, 3, 1],
    [1, 2, 1, 8, 0],
    [-5, 2, -3, 1, -1],
    [2, 1, 0, 1, -1]
], dtype=torch.float32)
input = torch.reshape(input, (1, 1, 5, 5))
pool = myModel()
output = pool.forward(input)
print(output)
```

输出结果：

```
tensor([[[[1., 2., 0., 3., 1.],
          [0., 0., 2., 3., 1.],
          [1., 2., 1., 8., 0.],
          [0., 2., 0., 1., 0.],
          [2., 1., 0., 1., 0.]]]])
```

其他的激活函数见[官方文档](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#relu)。