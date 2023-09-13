#deepLearning #CS #pytorch #python 

# Dataset & DataLoader in PyTorch

在 PyTorch 中加载数据主要依靠 Dataset 和 DataLoader 这两个类，前者规范了数据的表示形式，后者则提供了数据的加载方法。

## Dataset

Dataset 位于 `torch.utils.data` 模块下，这是一个抽象类，所有我们要加载的数据必须先封装成一个类，并且一定要继承 Dataset，实现 `getitem` 方法。接下来我们就来演示一下，如何创建我们自己的数据类。

```python
from torch.utils.data import Dataset

class MyData(Dataset):
	
	def __init__(self):
		pass
	
	def  __getitem__(self, idx):
		pass
	
	def __len__(self):
		pass
```

在 `init` 方法里，我们应该将一些有关于数据的必要信息作为属性挂载到类上；`getitem` 方法则是数据读取的接口，它接收一个参数 `idx`，这是要读取的数据项的索引值，我们理应返回对应的数据 data 及其标签 label；`len` 方法应返回全部数据的长度。

当然，类中的方法都是**依照需求而定**的。`getitem` 方法一定得有，只有这个方法被定义了，我们才可以用下标 `[]` 访问对应的数据，但是返回值由我们自己决定。

>小技巧：Dataset 已经对 `+` 运算符进行了重载，我们可以直接使用 `+` 来拼接两个数据集。 

## DataLoader

DataLoader 同样位于 `torch.utils.data` 模块下。Dataset 规范了数据的封装形式，而 DataLoader 则是提供了将数据加载到模型的接口，我们可以通过调整一些参数来改变加载数据的方式。

* 一些常用的参数：
	* `dataset`：`Dataset` 总数据
	* `batch_size`：`int` 一次加载的数据大小，默认为 `1`
	* `shuffle`：`bool` 在每个 epoch 打乱数据，默认为 `false`
	* `num_workers`：`int` 使用多少子进程来加载数据，`0` 意味着只使用主进程，默认为 `0`
	* `drop_last`：`bool` 如果按设置的 `batch_size` 分割数据后，发现有多余的数据，是否进行舍弃，默认为 `false`

下面是一个 DataLoader 的实例化示例：

```python
from torch.utils.data import DataLoader

mydata = MyData() # 定义在上面

data_loader = DataLoader(dataset=mydata, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
```

DataLoader 的实例化对象支持 `for` 循环迭代：

```python
for data, labels in data_loader: # data 和 labels 一维的大小由 batch_size 决定
	pass
```