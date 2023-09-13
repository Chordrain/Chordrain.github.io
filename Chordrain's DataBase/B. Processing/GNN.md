#deepLearning #cs #neuralNetwork 

# GNN：图神经网络

GNN 的全称是 Graph Neural Network，也就是图神经网络。这是神经网络的一个大类，专门用于处理图。
## 01 什么是 Graph

在讲解图神经网络之前，我们必须知道“图 (graph)”的概念。

首先要明确的一件事是，这里的“图”指的并非图像 (image)，而是数据结构中的图 (graph)，用 $G$ 来表示，一张图由 $V(节点)$ 和 $E(边)$ 组成，一个 $G$ 应该包含 3 个重要矩阵：

* 邻接矩阵 $A$：用来描述节点与节点之间的连接关系，一般取值在 0 和 1 之间。
* 度矩阵 $D$：用来记录每个节点的度，这是一个对角矩阵，其中对角元素 $D_{ii}=\Sigma_{j}A_{ij}$。
* 特征矩阵 $X$：用来描述每个节点的特征。
## 02 GNN 能做什么
### 2.1 Classification

GNN 能处理分类问题。例如说，我们现在有一个蛋白质分子的结构，我们想要训练出一个 classifier，让它告诉我们这个分子会不会导致突变，我们的 dataset 就是一个已经有 label 的 data，我们希望的是训练好的 classifier 可以知道任意一个它没有见过的分子 (unlabeled data) 是否也能引起突变。

![](./../S.%20Static/Images/GNN-01.png)
### 2.2 Generation

GNN 也可以完成 generation 任务，比如说我们喂给一个 generator 或者 decoder 一些分子结构，希望它合成一些类似的，具有其他特性的其他分子结构，比如说可以是更好合成的，成本更低的，效果更好的结构。

![](./../S.%20Static/Images/GNN-02.png)

### 2.3 other example

上面的例子里，输入和输出都是图，所以我们可以很轻易地理解为什么这些问题可以使用 GNN 来解决，接下来的例子可能会让你感到奇怪，但也确实是 GNN 的应用。

比如说，在一部侦探剧里，主角想找到真正的杀人凶手，这个问题也可以交给 GNN。我们可以这样想：剧中的人物作为节点输入，每个人物都有自己的职业、年龄、性别等等信息，这些就是节点的属性，同时我们知道，这些不同的人物之间是有关系的，这些关系也有可能是预测结果的重要依据，所以就可以把这些关系当作边，于是构成一张图，这样就有了 GNN 的输入。很显然，一般的神经网络只将人物作为孤立的节点进行考虑，从而忽略了人物之间的关联，而 GNN 弥补了这一缺陷，更加全面地考虑了这些因素。
## 03 GNN 的原理

有些时候，我们的输入未必是一张所有节点都已经有 label 的图，也许有些节点还没有 label，那么我们可以怎么进行训练呢？根据“近朱者赤近墨者黑”的原则，我们一般可以通过其有标签的邻居来推测一个节点的 label，也就是将其相连的邻居的信息进行汇聚。

![](./../S.%20Static/Images/GNN-03.png)

在这一过程中，训练用的 dataset 是人收集和输入的，但其中部分信息又需要机器自己去补全，这种方式具有半监督学习的特点。
## 04 GNN 的分类

上面已经我们说可以通过汇聚邻居节点的信息来推测 unlabeled 节点的 label，但是具体要怎么做？我们可以迁移 CNN 的方法，也就是使用卷积。

根据使用卷积方法的不同，GNN 可以分为两大类：[[Spatial-based GNN]] 和 [[Spectral-based GNN]]。

### 4.1 Spatial-based GNN

Spatial-based 方法就是将 [[CNN]] 的卷积操作推广到对图的处理上。我们说，在 CNN 中，每一层的卷积操作其实就是进一步提炼特征的过程，那么同样的，在 GNN 中，我们希望将上一层的图，经过卷积操作，输出经过进一步提炼的特征。

![](./../S.%20Static/Images/spatial-based-GNN-01.png)

上图中的 $h_n^i$ 表示在第 $n$ 个节点在第 $i$ 层的信息，经过卷积操作之后，我们需要把它更新成 $h_n^{i+1}$。这个过程具体的做法是，将该节点的邻居的特征 (neighbor feature) 聚合 (**aggregate**) 起来，来 update 下一层节点的状态 (hidden state)。

上面的过程只讨论了一种情况，那就是 GNN 的目标是预测每个节点的状态的情况；但是根据我们之前举的例子，GNN 的目标也可能是预测整张图的状态，例如说预测分子是否能引起突变，这是整个化学分子结构的 feature，那么这个时候我们就必须把图中每个节点的 feature 合起来，变成整个图的 feature (表示为 $h_G$)，这个过程我们叫作 **readout**，也就是把所有节点的 feature 集合起来代表整个图。

Aggregate 和 readout 是 GNN 中的重要环节，由于各种 GNN 的功能和应用不同，所以有些 model 可能没有 readout，但 aggregate 是必不可少的。

使用 spatial-based 方法的 GNN 有很多种，接下来列举几个比较经典的 model。

#### 4.1.1 NN4G：Neural Network for Graph

##### Embedding

首先，在输入层，我们将原始的图作为 NN4G 的输入，在 input layer 和 hidden layer 0 之间，我们一般会做一次 [[Embedding]]，直接用一个 embedding matrix ($\overline{w}_0$) 乘上节点的原始特征 ($x_n$)，得到该节点的新 feature ($h_n^0$)。即：
$$
h_n^0=\overline{w}_0·x_n
$$
##### Aggregate

在接下来的隐藏层之间，我们要做的就是 aggregate，那么 NN4G 是如何聚合一个节点的 neighbor feature 来更新这个节点的状态，也就是如何将 $h_n^i$ 变成 $h_n^{i+1}$ 的呢？

NN4G 采用的方法是将节点所有的 neighbor 全部加起来，再经过一个 transform 之后，将得到的东西再加上原本的 input feature。下图中展示了 $h_3^0\rightarrow h_3^1$ 的过程，公式就是：
$$
h_3^1=\hat{w}_{1,0}(h_0^0+h_2^0+h_4^0)+\overline{w}_1·x_3
$$

![](./../S.%20Static/Images/NN4G-01.png)
##### Readout

接下来来看一下 NN4G 是如何做 readout 的。一个 GNN 可以有很多层，假设如下图所示，现在的 NN4G 已经有 3 层了，我们可以将每一层的 node features 先加起来求平均，分别得到 $X_0,X_1,X_2$，再各自经过一个 transform，再相加得到一个 feature $y$，这个 $y$ 就是代表整个 graph 的 feature。

![](./../S.%20Static/Images/NN4G-02.png)

---
Q&S：
1. 为什么要选择相加？相加的好处是可以忽略节点数量不同的问题，这只是一种方法，不使用相加当然可以，还有很多奇奇怪怪的 model 采用了各种奇奇怪怪的方法。
2. 为什么不乘上权重再相加？当然可以，有些 model 就是这么干的，只不过 NN4G 没有这么做而已。

#### 4.1.2 DCNN：Diffusion-Convolution Neural Network

##### Embedding

DCNN 接收一张图，跟其他模型一样，在 hidden layer 0，它要对原始输入进行 embedding。它采用的方法是，对于任意一个节点 $n$，将与该节点距离为 1 的全部节点相加取平均，再乘上权重 $w_n^0$，得到该节点在 hidden layer 0 的特征 $h_n^0$，即：
$$
h_n^0=w_n^0·\operatorname{MEAN}(d(n,·)=1)
$$
下图以节点 3 为例展示了 embedding 的过程。

![](./../S.%20Static/Images/DCNN-01.png)
##### Aggregate

接下来讲解如何从 $h^0\rightarrow h^1$。在 CDNN 这个 model 中，aggregate 的方法和 embedding 的方法是差不多的，只不过从 hidden layer 0 开始，往后逐渐放宽对距离的限制，一开始是 1，再后面就会变成 2、3、4……，经过 $k$ 层之后，每个节点都会掌握整个 neighborhood 的信息，也就是说，对于第 $l\ (l>0)$ 层，其 aggregate 的公式如下：
$$
h^l_n=w^l_n·\operatorname{MEAN}(d(n,·)=l+1)
$$
之后，我们将每一层的所有 node feature ($h_0^l\sim h_n^l$) 分别组合起来形成一个单独的一层 $H^l$，这样就会得到 $H^0\sim H^K$，我们将每一层的同一个节点的特征序列取出，也就是 $h_n^0\sim h_n^k$，让它们经过一个 transform 以后，得到一个新的 feature $y_n$，我们就用 $y_n$ 代表节点 $n$ 的最终 feature。

![](./../S.%20Static/Images/DCNN-02.png)

#### 4.1.3 DGC：Diffusion Graph Convolution

DGC 这个 model 其实跟 DCNN 很像，它们唯一的不同就只在最后一步，DCNN 是将 $h_n^0\sim h_n^k$ 提取出来变换来得到 $y_n$，而 DGC 只是将 $H^0\sim H^K$ 加起来，加完之后得到的新矩阵就是最终的特征矩阵。

![](./../S.%20Static/Images/DGC-01.png)

#### 4.1.4 MoNET：Mixture Model Networks

MoNET 这个 model 的特点是：
1. 定义了对节点距离的度量
2. 使用了加权和而不是简单相加
##### Distance Measure

对于 $x$ 和 $y$ 两个节点，MoNET 的距离定义如下：
$$
\operatorname{u}(x,y)=(\frac{1}{\sqrt{\operatorname{deg}(x)}},\frac{1}{\sqrt{\operatorname{deg}(y)}})^T
$$
其中，$\operatorname{deg}$ 是节点的度 (degree)，$T$ 是指转置。

##### Aggregate

求出 $\operatorname{u}(x,y)$ 之后，以下是第 $l$ 层下，有 $k$ 个邻居的节点 $x$ 的特征计算公式：
$$
h_x^l=\sum_{i=0}^kw(\hat{\operatorname{u}}(x,i))\times h_i^l
$$
其中，$\hat{\operatorname{u}}(x,i)$ 是 $\operatorname{u}(x,i)$ 经过一个 transform 之后得到的值，然后将它丢到一个神经网络 $w(·)$ 中去求得权重，再用求得的权重乘上节点 $i$ 的特征 $h_i^l$，将节点 $0\sim n$ 与 $x$ 的加权和算出来，就能得到节点 $x$ 在聚合了 neighbor feature 之后的新 feature。

#### 4.1.5 GraphSAGE

关于 GraphSAGE，这里只做简单介绍。这个 model 有意思的点在于它的 aggregate 方法，除了一般的 mean、max-pooling，它还引入了 [[LSTM]]，也就是长短期记忆。

![](./../S.%20Static/Images/GraphSAGE-01.png)

LSTM 是用于处理序列数据的，这类数据的特点是具有顺序，但是一般来讲，图中的邻居节点不应该是有顺序的，那这样做可行吗？答案是可以的，正因为没有顺序，模型在学习过程中可能就会发现数据并不具有顺序，于是就会忽略掉顺序因素，然后学到一个比较好的 representation。

但并不是说用了 LSTM 效果就一定会好，以下是使用四种不同 aggregate 方法得出的 GraphSAGE 的性能比较：

![](./../S.%20Static/Images/GraphSAGE-02.png)

我们可以看到，GraphSAGE-LSTM 未必是最优的。

#### 4.1.6 GAT：Graph Attention Networks

这个 model 的特点是：它在做 aggregate 的时候要算加权和 (weighted sum)，而这里的权值 weight 不是事先给定的，而是由 model 自己计算出来的，计算的方法就是 attention。

在 GAT 中有一个 energy 的概念，它描述了某一个邻居节点对某一节点的重要性，也就是可以当作 weight 来使用。对邻居节点做 attention，就是计算出所有邻居节点的 energy，将 energy 作为 weight 以此来计算加权和，得到节点在下一层的 hidden representation。

![](./../S.%20Static/Images/GAT-01.png)

#### 4.1.7 GIN：Graph Isomorphism Network

GIN 是一个理论，它提供了关于 GNN model 有效性的理论证明，在此之前的 model 都是大家拿来就用，并没有去深究其为什么会有效的原因。GIN 给出了一个结论，那就是在 update 的时候，我们最好采用下面的方式去做 update：

$$
h_v^{(k)}=\operatorname{MLP}^{(k)}\left(\left(1+\epsilon^{(k)}\right) \cdot h_v^{(k-1)}+\sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)
$$

我们逐层解析上面这个公式。首先，这个公式推荐我们使用相加的方法来 aggregate 邻居节点的信息，之所以要这样做，而不选择 max-pooling、min-pooling，是因为下面的例子：

![](./../S.%20Static/Images/GIN-01.png)

图中不同颜色的圈代表不同的值。在 (a) 例子中，两个图中节点的全部值都是一样的，因而 max-pooling、mean-pooling 的结果也是一样的，但这就忽略了两张图的结构特征；(b) 中的两张图结构不一样，值不一样，但最大值一样，这就导致 max-pooling 会失效；(c) 中两张图结构不一样、值不一样、最大值一样、平均值一样，这就导致 max-pooling 和 mean-pooling 同时失效。

在做完 summation 之后，再用某个 concept 乘上节点自身上一层的 hidden representation (在这个模型的 implementation 中写到，这里的 $\epsilon$ 应该让机器自己去学，但它还有提到，其实 $\epsilon=0$ 是没有什么太大区别的)，再加上邻居节点的信息，最后将结果丢进一个 $\operatorname{MLP}$ (multi-layer perception)，这样就可以得到很好的效果。

### 4.2 Spectral-based GNN

