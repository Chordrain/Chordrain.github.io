#deepLearning #CS #python #pytorch 

# Convolution Layer in PyTorch

在 PyTorch 中，卷积层的定义已经封装好了，只需要用户自己调用。[官方文档](https://pytorch.org/docs/stable/nn.html#convolution-layers)已列出了在 `torch.nn` 下的各种类型的卷积层，其中最常用的是 `Conv2d`，这一节也只讲解 `Conv2d` 的使用。

`Conv2d` 中比较重要的参数有：
- **in_channels** (int) – Number of channels in the input image
- **out_channels** (int) – Number of channels produced by the convolution
- **kernel_size** (int or tuple) – Size of the convolving kernel
- **stride** (int or tuple, optional) – Stride of the convolution. Default: 1
- **padding** (int or tuple or str, optional) – Padding added to all four sides of the input. Default: 0
- **padding_mode** (str, optional) – `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'zeros'`
- **dilation** (int or tuple, optional) – Spacing between kernel elements. Default: 1
- **groups** (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
- **bias** (bool, optional) – If `True`, adds a learnable bias to the output. Default: `True`

虽然有这么多，但我们需要关注的其实只有 **in_channels**, **out_channels**, **stride** 和 **padding**，其他保持默认就好。如果你学习过卷积网络，那你应该知道 stride 和 padding 的概念，如果没有也可以通过[该链接](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#convolution-animations)快速了解这两个参数的作用。另外，channel 的含义并非是一张图像的尺寸，如果你不知道 channel 的含义，请参考 [[CNN]]。

in_channels 规定了输入图像的 channel 的数量，而 out_channels 规定了输出的 channel 的数量。如果某层的 out_channels 设定为 2，那么该层最终会输出一张有 2 个 channel 的图像，并且会生成 2 个卷积核 (kernel)。

最后来说说 kernel_size，它决定了卷积核的大小，当值为 $n$ 时，卷积核的大小为 $n*n$。从理论上来讲，kernel_size 越小，可以获得越精细的特征提取，但计算量也增加了，同时容易过拟合；而当 kernel_size 越大时，模型计算量相对减少，性能相对更稳定，但也容易引起信息丢失、模糊等问题。卷积核是神经网络要学习的参数，所以具体的值不需要我们考虑。