#CS #deepLearning #python #pytorch 

# nn.Module in PyTorch

`torch.nn` 下有很多搭建神经网络结构的模块，具体参见[官方文档](https://pytorch.org/docs/stable/nn.html)。这一节我们仅关注 `Module` 这一模块，官方的对其的介绍是 "base class for **all** neural network modules"，可见其重要性。

在 PyTorch 中，神经网络模型也以类的形式封装，且应继承自 `nn.Model`，其骨架大概为：

```python
import torch.nn as nn

class Model(nn.Module):
	
	def __init__(self):
		super().__init__() # 这句必须写
		# 在此搭建神经网络的层结构
	
	def forward(self, input):
		# 在此实现正向传播算法，input 为神经网络的输入，最后应该返回神经网络的输出
```

下面是官方的一个示例：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

你可以通过 `print` 函数打印一个网络的结构 (要实例化对象)。上示网络的打印结果如下：

```
Model(
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
)
```