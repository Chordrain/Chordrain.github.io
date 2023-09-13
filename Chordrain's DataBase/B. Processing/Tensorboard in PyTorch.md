#deepLearning #CS #pytorch #python 

# Tensorboard in PyTorch

Tensorboard 和 matplotlib 一样，都可用于画图，绘制学习曲线，但 tensorboard 的功能更为强大，使用起来更加方便。

* `torch.utils.tensorboard`

使用前要先进行安装：`pip install tensorboard`

Tensorboard 的运作跟 Jupyter notebook 有点相似，启动后都会占用一个端口，以网页的形式呈现，也需要一个工作目录。终端启动命令：

```
tensorboard --logdir=工作目录 --port=设置端口
```

---

如果因为 tensorboard 版本问题，运行时报错，请下载 tensorboardX：

```
pip install tensorboardX
```

然后从 tensorboardX 中导入 `from tensorboardX import SummaryWriter`。

## 01 SummaryWriter

* `from torch.utils.tensorboard import SummaryWriter`

SummaryWriter 返回一个对象，用这个对象可以向 tesorboard 中添加图表。

### 1.1 add_scalar

`add_scalar` 方法可以像 matplotlib 一样绘制函数图像，例如我们想做一张 $y=x^2$ 的函数图像：

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs") # 设置工作目录
for i in range(100):
	writer.add_scalar("y=x", i, i**2) # param1: 图像名称 param2: y param3: x
writer.close()
```

然后启动 tensorboard：

```
tensorboard --logdir=logs
```

能看到图像：

![](./../S.%20Static/Images/Tensorboard-in-PyTorch-01.png)

如果我们再次运行代码，绘制一个 $y=x$ 的函数图像，并且不改变图像的名称：

```python
from tensorboardX import SummaryWriter  
writer = SummaryWriter("logs")  
for i in range(100):  
	writer.add_scalar("y=x", i, i)  
writer.close()
```

你会发现图像变成了这样：

![](./../S.%20Static/Images/Transformer-in-PyTorch-02.png)

其实 tensorboard 确实完成了它的任务，它在同一幅图中绘制了两条函数图像，但是它会进行自动拟合，也就是画出了图中那些弧形的图线。想要取消这种糟心的状况只有一种方法：删除工作目录下对应的文件，消除其中一条线，并重启 tensorboard；如果全部删除，那么 tensorboard 中将不剩一张图片。

### 1.2 add_image

`add_image` 方法可以向 tensorboard 中添加图片，一般使用这个方法观察数据集。