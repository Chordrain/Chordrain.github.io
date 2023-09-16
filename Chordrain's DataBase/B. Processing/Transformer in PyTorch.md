#deepLearning #CS #python #pytorch 

# Transformer in PyTorch

在学习本节内容之前，请确保你对 [Transformer](../D.%20Knowledge/Computer%20Science/Transformer.md) 的架构以及 PyTorch 的基本使用有足够的了解。本节内容主要以代码实现为主，不会深入原理。

Transformer 的架构如下：

![](./../S.%20Static/Images/Transformer-04.png)

可以看到，transformer 的架构中有多个块，而且它们很多是重复的，所以我们只要将这些块逐一攻破，然后再连接起来就完成了。接下来的内容全是关于这些块的 PyTorch 实现，再次强调，请确保你已经了解 transformer 的工作流程。

## Add & Norm

Add 就是做残差 (residule)，norm 就是做标准化，所以这一块的内容，就是先做残差，再对残差做 layer normalization。我们用 `SublayerConnection` 类来做连接，在这个类中，我们会实现 add & norm，但是，具体的 layer normalization 交由另一个类 `LayerNorm` 完成，`SublayerConnection` 只负责调用。以下是 `SublayerConnection` 的具体代码：

```python
class SublayerConnection(nn.Module):  
	"""  
	子层的连接: layer_norm(x + sublayer(x))  
	上述可以理解为一个残差网络加上一个LayerNorm归一化  
	"""  
	
	def __init__(self, size, dropout=0.1):  
		"""  
		:param size: d_model  
		:param dropout: drop比率  
		"""  
		super(SublayerConnection, self).__init__()  
		self.layer_norm = LayerNorm(size)  
		# TODO：在SublayerConnection中LayerNorm可以换成nn.BatchNorm2d  
		# self.layer_norm = nn.BatchNorm2d()  
		self.dropout = nn.Dropout(p=dropout)
        
	def forward(self, x, sublayer):  
		return self.dropout(self.layer_norm(x + sublayer(x)))
```

在上面的代码中，我们构建了一个连接层，在初始化函数中，我们构建了这个连接层的结构，将 layer normalization 层嫁接到了连接层，然后又加了 dropout 层，这一层是用来防止过拟合的，可做可不做，重点在 layer norm 上。在前向传播中 (forward)，`x` 是 self-attention 层的输入，`sublayer` 是指上一层，具体是哪一层由用户传入的参数决定，我们假设这里的 sublayer 就是 self-attention 层，所以 `sublayer(x)` 就是 self-attention 层的输出，` x + sublayer(x) ` 就得到了残差，然后再将残差丢进 `layer_norm` 中做 layer normalization，得到 add & norm 的结果，最后将结果丢进 `dropout`，返回 `dropout` 之后的结果就做完了。

接下来是 layer norm 的实现代码：

```python
class LayerNorm(nn.Module):  
	"""  
	构建一个LayerNorm Module  
	LayerNorm的作用：对x归一化，使x的均值为0，方差为1  
	LayerNorm计算公式：x-mean(x)/\sqrt{var(x)+\epsilon} = x-mean(x)/std(x)+\epsilon  
	"""  
	  
	def __init__(self, x_size, eps=1e-6):  
		"""  
		:param x_size: 特征的维度  
		:param eps: eps是一个平滑的过程，取值通常在（10^-4~10^-8 之间）  
		其含义是，对于每个参数，随着其更新的总距离增多，其学习速率也随之变慢。  
		防止出现除以0的情况。  
		  
		nn.Parameter将一个不可训练的类型Tensor转换成可以训练的类型parameter，  
		并将这个parameter绑定到这个module里面。  
		使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。  
		"""  
		super(LayerNorm, self).__init__()  
		self.ones_tensor = nn.Parameter(torch.ones(x_size)) # 按照特征向量大小返回一个全1的张量，并且转换成可训练的parameter类型  
		self.zeros_tensor = nn.Parameter(torch.zeros(x_size))  
		self.eps = eps  

    def forward(self, x):  
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True) # 求标准差  
		return self.ones_tensor * (x - mean) / (std + self.eps) + self.zeros_tensor # LayerNorm的计算公式
```

上述代码其实就是在实现下面的 layer norm 公式：

$$
y=\frac{x-\operatorname{E}(X)}{\sigma+\epsilon}*\gamma+\beta
$$

其中，$\gamma$ 和 $\beta$ 是放缩因子和移位因子，是学习参数；$\operatorname{E}(X)$ 是序列的均值，$\sigma$ 是标准差，而 $\epsilon$ 是我们自己设置的常数，目的是为了防止分母为零。上面的式子里，只有 $\gamma$ 和 $\beta$ 是需要学习的，其他都是给定或者计算出来的。所以在初始化函数里，我们只是往对象上创建了两个学习参数和 $\epsilon$ 这 3 个属性，而在 forward 中，则是实现了上面的公式。由于是矩阵运算，所以肯定得事先告诉我们输入矩阵的大小 `x_size`，这样才能确定学习参数的大小。

## Multi-Head Attention

多头注意力机制的介绍请参考 [[Self-attention#07 Multi-head Self-attention|这篇笔记]]。注意力机制的流程图可简化如下：

![](./../S.%20Static/Images/Transformer-in-PyTorch-01.png)

上图中，最后 $QKV$ 相乘得到的结果就是最终的特征 $B$，而在多头注意力中，最后会产生多个 $B$，我们需要经过线性变换，把这些 $B$ 统合到一起，变成一个特征 $B^\prime$。由于多头注意力机制要多做几遍普通的注意力机制所做的运算，所以我们先把正常的 self-attention 的函数写出来：

```python
def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :return: 经自注意力机制计算后的值
    """
    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    if mask is not None:
        """
        scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
        在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
        """
        mask.cuda()
        # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引
	    scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return torch.matmul(self_attn_softmax, value), self_attn_softmax
```

下面来解释一下上面的代码：

`d_k = query.size(-1)`：`size` 方法返回张量的形状，参数 `-1` 是指返回最后一维的长度。

`scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)`：核心代码，将 `query` 和 `key` 相乘求相关性，`key.transpose(-2,-1)` 的意思是交换张量的倒数第一维和倒数第二维。最后还要除以 `d_k` 的根号是因为这里采用的是 [[Self-attention#5.2 Scaled dot-product|scaled dot-product]]。

`dropout`：防止过拟合，可做可不做。

`torch.matmul(self_attn_softmax, value)`：将 key 和 query 相乘，得到相关性。

接下来实现 multi-head attention 层：

```python
class MultiHeadAttention(nn.Module):
    """
    多头注意力计算
    """

    def __init__(self, head, d_model, dropout=0.1):
        """
        :param head: 头数
        :param d_model: 词向量的维度，必须是head的整数倍
        :param dropout: drop比率
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model

        """
        由于多头注意力机制是针对多组Q、K、V，因此有了下面这四行代码，具体作用是，
        针对未来每一次输入的Q、K、V，都给予参数进行构建
        其中linear_out是针对多头汇总时给予的参数
        """
        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)  # batch_size大小，假设query的维度是：[10, 32, 512]，其中10是batch_size的大小

        """
        下列三行代码都在做类似的事情，对Q、K、V三个矩阵做处理
        其中view函数是对Linear层的输出做一个形状的重构，其中-1是自适应（自主计算）
        从这种重构中，可以看出，虽然增加了头数，但是数据的总维度是没有变化的，也就是说多头是对数据内部进行了一次拆分
        transopose(1,2)是对前形状的两个维度(索引从0开始)做一个交换，例如(2,3,4,5)会变成(2,4,3,5)
        因此通过transpose可以让view的第二维度参数变成n_head
        假设Linear成的输出维度是：[10, 32, 512]，其中10是batch_size的大小
        注：这里解释了为什么d_model // head == d_k，如若不是，则view函数做形状重构的时候会出现异常
        """
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        # x是通过自注意力机制计算出来的值， self.attn_softmax是相似概率分布
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)

        """
        下面的代码是汇总各个头的信息，拼接后形成一个新的x
        其中self.head * self.d_k，可以看出x的形状是按照head数拼接成了一个大矩阵，然后输入到linear_out层添加参数
        contiguous()是重新开辟一块内存后存储x，然后才可以使用.view方法，否则直接使用.view方法会报错
        """
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)
```

