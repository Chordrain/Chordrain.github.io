#CS #deepLearning #python #pytorch

# PyTorch

>PyTorch 是一个开源的深度学习框架，提供了各种张量操作并通过自动求导可以自动进行梯度计算，方便构建各种动态神经网络。支持使用 GPU/TPU 进行加速计算。

Tensor (张量) 是 PyTorch 中一个非常重要的概念，所有的输入输出和参数都是以 tensor 的形式呈现，学习如何创建和使用以及各种关于 tensor 的信息，请阅读 [[Tensor in PyTorch]]。

知道了 tensor 是什么了之后，就可以来了解一下如何在 PyTorch 中加载数据了。在 PyTorch 中加载数据主要涉及 `Dataset` 和 `DataLoader` 这两个类，请阅读 [[Dataset & DataLoader in PyTorch]]。此外，[[Tensorboard in PyTorch|tensorboard]] 是一个非常好用的训练过程可视化工具，也很值得学习。

接下来就可以正式进入到使用 PyTorch 搭建神经网络的学习了，阅读 [[nn.Module in PyTorch]] 来了解如何在 PyTorch 中搭建神经网络的骨架。然后，我们将以卷积神经网络为例，讲解在 PyTorch 中搭建神经网络的流程。你可以阅读 [[Convolution Layer in PyTorch]] 来了解卷积层在 PyTorch 中的使用，阅读 [[Pooling Layer in PyTorch]] 来了解池化层在 PyTorch 中的使用，阅读 [[Non-linear Activations in PyTorch]] 来了解激活函数在 PyTorch 中的使用。

通过以上的学习，你就应该对如何在 PyTorch 中搭建神经网络结构有了一定的了解了，下一步就是掌握神经网络的训练流程，阅读 [[Loss Function & Backpropagation in PyTorch]] 来了解 PyTorch 中的损失函数和反向传播。

如何在 PyTorch 中搭建 transformer 架构，请阅读 [[Transformer in PyTorch]]。