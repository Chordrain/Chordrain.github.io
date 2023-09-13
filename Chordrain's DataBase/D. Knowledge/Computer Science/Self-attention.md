#deepLearning #cs #neuralNetwork 

# Self-attention

## 01 Sophisticated Input

不论是在预测问题还是在影像处理上，我们都会假设输入的向量长度是一定的，但如果是不一定的呢？

比如说在翻译问题中，输入是句子，我们可以采用 one-hot encoding 或 word embedding 的方式将句子转换成向量，而句子是不定长的，也就是说输入的向量是不定长的，这就与我们之前所谈论的情况不一样了。

或者说更复杂一些，输入的向量不仅不定长，而且数量也不确定，如语音辨识，输入是很长一段语音，这段语音中包含了很多句子，机器需要把这些句子拆分出来，然后进行翻译，这种情况下，句子的数量和长度都不是确定的。

还有一些更复杂的数据结构，比如说图，例如社交网络、分子结构等，这些信息也可以看作是一组向量，所有这些情况就构成了更加复杂的输入 (more sophisticated input)，那么我们要如何处理这些输入呢？

## 02 What's the output?

除了更加复杂的输入，输出也分很多种情况。

对于词性分析问题，它的目标是将一句话中的每个单词的词性进行分类，这个时候，每一个向量都有一个 label；再比如语音辨识问题，机器需要把一段语音中的很多个 frame 转换成对应词；还有图的例子，例如在一个社交网络中，机器需要对图中的人进行分类，这些都是输入与输出等长的例子，也就是 N-to-N 的情况。

另一种情况是，对于一整个输入，机器只需要输出一个 label 就好了，例如情感分析，输入是一个句子向量，机器需要对句子中所表露出的情感进行判断，究竟是正面还是负面；再比如语者辨认，根据输入的语音判断是哪个人讲的，这些都是 N-to-1 的情况。

还有一种情况则更复杂，那就是输出不确定的情况。例如说翻译，输入的文本长度跟输出的文本长度并不存在必然联系，输出的长度应该由机器自己决定，这种情况下就变成了 N-to-N’。

## 03 Backdraws of FCNN

在讨论完了所有输入和输出的情况之后，我们来看看 FCNN 有什么缺点。在学过 FCNN 和 CNN 之后我们知道，FCNN 是 bias 最小的神经网络，它几乎可以拟合出任何函数，但是这表明它就是无敌的吗？未必，我们来看看下面这个例子。

$$
\text{Please translate: I saw a saw.}
$$

在上面这个翻译任务中出现了两个相同的单词 $saw$，显然这两个 $saw$ 的意思是不一样的，但对于 FCNN 来讲，它们没有任何差别，也就是说，FCNN 无法考虑上下文的关系，当它处理同一个单词 $saw$ 的时候，它并不会输出两个不同的答案。

那么难道 FCNN 就无法处理这种情况了吗？当然是可以的，不过要改变一下方法。我们可以设置一个 window，这个 window 包含了要翻译的词以及其前后的上下文，window 越大，其包含的上下文信息就越多。如果我们要 FCNN 考虑整个句子的长度，那么我们就必须开一个足够大的 window 把整个句子都盖住。但正如前面所讨论的，输入的长度可能是不定的，如果要这么做，我们就必须提前调查一下所有句子中最长的句子的长度，然后把 window 的大小设置成这个长度才有可能盖住所有句子。然而这样做，不仅会使参数增多，使运算量增大，还容易 overfitting。

那么到底有没有更好一点的做法呢？当然有，这就是接下来要讲的 self-attention。

## 04 Intro to Self-attention

针对上面的问题，self-attention 的做法是，将输入的向量转换成另一个向量，这些转换后的向量是考虑了上下文之后的向量，然后再将这些向量送给 FCNN 就行了。

![](S.%20Static/Images/Self-attention-01.png)

当然，self-attention 不止可以用一次，可能经过第一次 self-attention 之后 FCNN 已经有了输出，然后我们再用一次 self-attention 再转换一次，然后再交给另一个 FCNN 处理，最后得到结果。

现在我们已经知道 self-attention 做的主要工作就是将一个向量转换成一个考虑了所有向量之后的向量，也就是如下图所示的过程：

![](S.%20Static/Images/Self-attention-02.png)

上图中的 $b^i$ 就是由下层的 $a^i$ 在经过 self-attention 考虑了 $\sum_{i=1}^4a^i$ 之后得到的结果，那么这一步具体要怎么做呢？

我们现在以 $b^1$ 为例讲述求解 $b$ 的过程。我们要做的是，在 $a^1\sim a^4$ 整个序列中找出与 $a^1$ 有关的，能作为 $a^1$ 的 label 的判断依据的向量，为此，我们需要给两个序列之间的关联性一个度量，取名叫 $\alpha$，那么接下来的问题就是如何计算 $\alpha$。

## 05 Relevance α

$\alpha$ 的计算方式主要有两种：dot product 和 additive。

### 5.1 Dot-product

Dot-product 是最常用的方法。它的做法是：

将两个向量输入，将两个向量分别乘上不同的矩阵 $W^q,W^k$，得到两个新向量 $q$ 和 $k$，再将这两个新向量进行点乘，得到的结果就是这两个向量的关联度 $\alpha$。Dot-product 的公式是：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T)V
$$

**注意：上面的公式已经是注意力机制的完整公式，只有 $QK^T$ 这一部分是 dot-product。**

### 5.2 Scaled dot-product

>补充于 2023 年 9 月 6 日

Scaled dot-product 其实就是对 dot-product 进行了放缩，最后除了个常量。这个常量记作 $d_k$，其值等于矩阵的维度。这样做的原因在论文中是这样解释的：

>We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$.

当运算的矩阵的维度过高时，可能会导致点乘的结果过大或过小，这样的话，后续将结果放进 softmax 的时候会导致其分布到函数的两端 (softmax 的函数图像的两端梯度都很小)，梯度比较小，可能会出现梯度消失的问题。除以 $\sqrt{d_k}$ 进行放缩，可以使结果分布到函数图像的中间。Scaled dot-product 的公式是：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

**注意：上面的公式已经是注意力机制的完整公式，只有 $\frac{QK^T}{\sqrt{d_k}}$ 这一部分是 scaled dot-product。

### 5.3 Additive

Additive 的做法的第一步和 dot-product 是一样的：将两个向量输入，将两个向量分别乘上不同的矩阵 $W^q,W^k$，得到两个新向量 $q$ 和 $k$。但是接下来不是要将它们做点乘，而是相加，经过一个 activation function，再经过一个 transform，最终得到 $\alpha$。

下面是这两种方法的图示：

![](S.%20Static/Images/Self-attention-03.png)

## 06 Attention

在知道了如何计算 $\alpha$ 之后，我们选择使用 dot-product 方法，继续来看如何求解 $b^1$。

首先，根据 dot-product 的方法，这里我们要求 $a^1$ 和剩下向量的关系，所以我们应该先去求 $q^1$，这个值我们称为 query。根据公式，$q^1=W^qa^1$。

接下来，我们应该去求 $k$，也就是用 $W^k$ 去乘上剩下的所有向量。这个 $k$ 被我们称作是 key。

在求出 $q^1$ 和 $k^2, k^3, k^4$ 之后，将它们分别进行点乘，就能得到 $\alpha_{1,2}$，$\alpha_{1,3}$，$\alpha_{1,4}$，这个 $\alpha$ 也有一个名字，叫作 attention score。

但其实，我们还会用 $q^1$ 去乘上自己得到 $\alpha_{1,1}$，这也很好理解，毕竟你要考虑 $a^1$ 和其他向量的关系，也不能忘了 $a^1$ 与自身的关联。

在算出所有的 $\alpha_{1,i}$ 之后，我们会将它们放进一个 soft-max 里进行转换得到 $\alpha^\prime_{1,i}$：

$$
\alpha^\prime_{1,i}=\frac{\operatorname{exp}(\alpha_{1,i})}{\sum_j\operatorname{exp}(\alpha_{1,j})}
$$

这一步其实是 normalization，至于为什么是 soft-max，只能说这是先人尝试之后的结果，你当然也可以使用 ReLu 之类的函数，但只不过说经过前人的尝试，soft-max 的效果最好。

>补充于 2023 年 9 月 9 日：也许你有疑问为什么一定要将 α 放进 softmax 或其他类似的函数里，这是因为，α 代表了 key 和 query 之间的相似性，注意力机制的本质是关注相似性高的，而忽略相似性低的。最后对所有信息进行整合时，我们其实是根据 α 的大小进行加权聚合，相似性高的向量权重就大一点，所以我们需要将所有的 α 放进 softmax 中，以此来得到权重。简单来讲，计算 α 的目的就是为了得到权重，而后面计算出的 v 是每个向量的价值。

在做完这些工作之后，我们就终于可以来计算 $b^1$ 了。首先，我们需要将 $a^1\sim a^4$ 都乘上一个矩阵 $W^v$，得到 $v^1\sim v^4$，然后根据下面的公式就能得到 $b^1$ 了：

$$
b^1=\sum_i\alpha_{1,i}^\prime v^i
$$

讲完了全部过程，我们再来回顾一下。很显然，如果 $a^1$ 与其中某一个向量 $a^i$ 的关联度最大，那么 $\alpha_{1, i}^\prime$ 就会很大，这就会使得最终的 $b^1$ 与 $v^i$ 更加接近，也就是对 $a^i$ 的 attention 最大，这也就得到了考虑了上下文的向量。

Self-attention 里其实并没有做太多的工作，它需要通过 dataset 学习的其实仅仅只有三个参数：$W^q$，$W^k$ 和 $W^v$。除此之外的所有参数都是人为设置好的。

如果你想知道以上所有过程的代码思路 (仅仅是思路) 或者说矩阵运算的技巧，可以参考[这个视频](https://www.bilibili.com/video/BV1Wv411h7kN?p=39)。

## 07 Multi-head Self-attention

Self-attention 其实还有很多变体，其中一个在今天应用非常广泛的模型就是 multi-head self-attention。

Multi-head self-attention 的想法是，事物与事物之间的关联性有时候是多方面的，当考虑不同的方面时，两个事物的关联性可能就是不同的，所以我们需要不止一种 $\alpha$。考虑多少种 $\alpha$，就有多少个 head。

如果这样考虑的话，根据正常的 self-attention 的做法，原本只需要对每个 $a^i$ 计算 $q^i, k^i, v^i$，现在则还需要针对每个 $q^i$ 计算 $q^{i, 1}, q^{i, 2}...$，也就是考虑多方面的关联性。那既然 $q$ 现在有多个，那么 $k$ 和 $v$ 也肯定要有多个。那至于怎么进一步得到 $q^{i, 1}, q^{i, 2}...k^{i, 1}, k^{i, 2}...v^{i, 1}, v^{i, 2}...$，其实是用更多的 $W^q, W^k, W^v$ 来乘上原来的 $q,k,v$，也就是：

$$
\begin{aligned}
& \boldsymbol{q}^{i, \mathbf{1}}=W^{q, 1} \boldsymbol{q}^i \\
& \boldsymbol{k}^{i, 2}=W^{k, 2} \boldsymbol{k}^i\\
& \boldsymbol{v}^{i, 2}=W^{v, 2} \boldsymbol{v}^i
\end{aligned}
$$

所以，其实 multi-head self-attention 要做的事情和 self-attention 是一样的，只不过现在有多个 head，所以要每个 head 都做一遍独立的 self-attention 而已，最后你能得到多个 $b^{i, j}$，那接下来你可能会把这些 $b$ 都连接起来形成一个矩阵，然后将其乘上另一个矩阵 $W^o$，得到最终的 $b^i$。

## 08 Positional Encoding

不知道你看到这里有没有发现 self-attention 的一个缺陷？Self-attention 似乎只在考虑 attention，也就是向量与向量之间的关联，却漏掉了一个很重要的信息 —— 那就是“位置 (position)”！例如某个向量是排在序列的最前面还是最后，它是完完全全没有考虑的，而位置信息很明显，在很多任务中都是很重要的，尤其是对于文字处理而言，比如说，动词出现在句首的概率比较低，那么如果一个词出现在句首，它可能是动词的概率就比较低。

所以怎么办呢？这就是 positional encoding 这项技术的作用，它可以把向量的位置信息给“塞进去”。它给序列中的每一个位置都设定了一个 vector，称为 $e^i$，不同的位置都有一个专属的 $e$。我们要做的事情是，将这个 $e^i$ 加到 $a^i$ 上面去，就结束了。没错，就这么简单。

那么这个 $e$ 是如何确定的呢？在 *Attention is All You Need* 论文中，$e$ 是人为规定 (hand-crafted) 的，他们是使用 sin、cos 这些神奇的函数来得到 $e$ 的，至于可不可以用其他函数，答案当然是可以，positional encoding 目前还是一个尚待研究的问题，所以你用什么都是没问题的；当然，$e$ 也可以是通过 data 学习出来的。

## 09 Self-attention v.s. CNN

Self-attention 同样是可以用于处理图像的。我们知道图像信息也是向量，使用 self-attention 我们就可以去考虑每个 pixel 之间的关联度，让机器自己去筛选出一张图片中重要的信息。当今，使用 self-attention 处理图片已然不是什么很新鲜的事情，那么就会有一个问题：self-attention 和 CNN 孰优孰劣？

其实，我们可以将 self-attention 当作是一个复杂版的 CNN，而 CNN 是 self-attention 的简化版。我们知道，CNN 每次都只考虑一个 perceptive filed 的信息；而 self-attention 则是通过 pixel 与每个 pixel 之间的关联度，来自动筛选出值得关注的 filed，这就好像是在说 self-attention 的 perceptive filed 是自己学习出来的。

在 [On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/abs/1911.03584) 这篇论文里，作者用数学方法讨论了 CNN 和 self-attention，并且证明了只要参数设置正确，self-attention 完全可以变成 CNN。也就是说，CNN 只是 self-attention 的一种特例。

那么究竟 CNN 和 self-attention 谁更加好呢？显然通过我们上面的讨论，CNN 的 bias 比 self-attention 更大，也就是 CNN 的弹性更加小，能拟合出的函数更加少。而我们也知道，弹性大的网络在面对较小的数据量时很容易 overfitting；弹性小的网络在面对更大的数据量的时候则很难再学到新的东西。也有学者对此做过专门研究，最后发现，当数据量较少时，CNN 的准确率是能超过 self-attention 的，但当数据量到达 100M 级别的时候，CNN 就被 self-attention 超过了。所以 CNN 和 self-attention 到底谁更好是依你的数据量来定的。

## 10 Self-attention for Graph

Self-attention 也是可以用在 graph 上的，例如下面这张图：

![](S.%20Static/Images/Self-attention-04.png)

当我们需要去给节点 1 做 label 的时候，我们就可以去考虑它的邻居节点，然后使用 self-attention 对其与邻居节点之间的关联性进行考量，而对于那些不与节点 1 相连的节点则代表我们已经人为地帮机器排除了彼此的联系，所以就不需要机器再去考虑了。这其实就是 [[GNN]] 的一种。

## 11 Attention & Self-attention

我们上面介绍的是 self-attention，注意：self-attention (自注意力机制) 和 attention (注意力机制) 是不一样的！我们可以用一个例子来说明注意力机制是什么。以淘宝搜索商品为例，淘宝要做的是，将我们输入的关键词与其数据库内的商品相关联，其本质其实和 self-attention 很像，都是找关联性强的，忽略关联性弱的，但不同的是，在这个例子中，我们的 $q,k,v$ 来自 source (商品) 和 target (关键词)，它们位于 transfomer 架构的两端；而在上面举的机器翻译的例子中，source 是输入的中文，target 是要求解的英文翻译，它们位于 transfomer 的两端，但 $q,v,k$ 只来自 source。