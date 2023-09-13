#paperReading #drugRepositioning #cs #GNN
# 01 摘要

在这项研究中，我们提出了一种新的方法，称为 DRHGCN (基于异构信息融合图卷积网络的药物重新定位)，以发现某种疾病的潜在药物。为了充分利用不同领域 (即药物-药物相似网络、疾病-疾病相似网络和药物-疾病关联网络) 的不同拓扑信息，我们首先设计了域间和域内特征提取模块，通过对网络进行图卷积运算来学习药物和疾病的嵌入，而不是简单地将三个网络整合成一个异构网络。然后，我们将域间嵌入和域内嵌入并行融合，得到更具有代表性的药物和疾病嵌入。最后，我们引入了一种层关注机制来组合来自多个图卷积层的嵌入，以进一步提高预测性能。我们发现，DRHGCN 在四个基准数据集上取得了优异的性能 (平均 AUROC 为 0.934，平均 AUPR 为 0.539)，优于目前的方法。重要的是，我们对 DRHGCN 预测的候选药物进行了分子对接实验，提供了几种新的批准用于阿尔茨海默病 (如苯托品) 和帕金森病 (如三苯基和氟哌啶醇) 的药物。
# 02 研究方法

该论文中提出的模型分为 3 个子网络，分别完成不同的功能。
## 2.1 网络构建

令 $G$ 为一个包含了 $N$ 种药物和 $M$ 种疾病的 drug-disease 相似性网络，它的邻接矩阵为一个 $N\times M$ 大小的矩阵 $A$，其元素的取值为 0 或 1，$A_{i, j}=0$ 表示第 $i$ 种药物与第 $j$ 种疾病无关或未知，否则有关。

我们使用 $G_r$ 来表示一个 drug-drug 相似性网络，它的邻接矩阵是一个 $N\times N$ 的矩阵 $A^r$，它由药物相似性矩阵 $S^r$ 组成。特别地，如果第 $j$ 种药物 $r_j$ 是第 i 种药物 $r_i$ 基于矩阵 $S^r$ 的距离为 $k$ 的邻居节点，则 $A^r_{i,j}=S^r_{i,j}$；否则，$A^r_{i,j}=0$. $\text{Top}k$ 表示每种药物或疾病的 k 近邻的数量。

类似地，我们令 $G_d$ 表示一张 disease-disease 相似性网络，它的邻接矩阵是一个 $M\times M$ 的矩阵 $A^d$，它由疾病相似性矩阵 $S^d$ 组成，特别地，如果第 $j$ 种疾病 $d_j$ 是第 $i$ 种疾病 $d_i$ 基于矩阵 $S^d$ 的距离为 $k$ 的邻居节点，则 $A^d_{i, j}=S^d_{i,j}$；否则，$A^d_{i, j}=0$.
## 2.2 模型结构
### 2.2.1 Encoder

DRGCN 的 encoder 层使用 [[GCN]] 来提取药物与疾病的 [[Embedding|embeddings]]，基于 $S^r$ 和 $S^d$ 来分别得到 drug 和 disease 的 intra-domain embeddings，基于 $A$ 来得到 drug-disease 的 inter-domain embeddings。最后将三者混合得到最终的 drug 和 disease 的 embedding 表示。

首先，初始化 drugs 和 diseases 的 embeddings 如下：

$$
H^0=\left[\begin{array}{l}
H_r^0 \\
H_d^0
\end{array}\right]=\left[\begin{array}{cc}
S^r & 0 \\
0 & S^d
\end{array}\right] \in \mathbb{R}^{(N+M) \times(N+M)}
$$

Intra-domain 特征提取模型定义如下：

$$
\hat{H}^{l+1}=\left[\begin{array}{c}
\hat{H}_r^{l+1} \\
\hat{H}_d^{l+1}
\end{array}\right]=\left[\begin{array}{c}
\operatorname{GCN}\left(A^r, H_r^l, W_r^l\right) \\
\operatorname{GCN}\left(A^d, H_d^l, W_d^l\right)
\end{array}\right]
$$

其中，$\hat{H}_r^{l+1} \in \mathbb{R}^{N \times k}$ 是第 $l$ 层输出的 drug intra-domain features；$\hat{H}_d^{l+1} \in \mathbb{R}^{M \times k}$ 是第 $l$ 层输出的 disease intra-domain features。

$\operatorname{GCN}\left(A, H, W\right)$ 是卷积操作，其公式如下：

$$
\operatorname{GCN}(\mathrm{A}, \mathrm{H}, \mathrm{W})=\sigma\left(D^{-\frac{1}{2}} \mathrm{~A} D^{-\frac{1}{2}} \mathrm{HW}\right)
$$

其中，$D=\operatorname{diag}\left(\sum_j A_{i j}\right)$ (以 $\Sigma_jA_{ij}$ 为对角线上元素的对角矩阵)，$\sigma(\cdot)$ 是一个 ReLu 激活函数。

受 [[BGNN]] 的启发，DRHGCN 采用 bilinear aggregator (BA) 和传统的 GCN aggregator (AGG) 来提炼疾病和药物的 inter-domain feature。BA 采用 element-wise product 来保留节点之间的 interaction 和过滤 discrepant information。

对于药物 $r_i$，其 inter-domain feature 的定义如下：

$$
\tilde{H}_{r_i}^{l+1}=\sigma\left(\alpha^l \frac{\sum_j\left(H_{d_j}^l W^l \odot H_{r_i}^l W^l\right) A_{i j}}{\sum_j A_{i j}}+\left(1-\alpha^l\right) \frac{\sum_j H_{d_j}^l W^l A_{i j}}{\sum_j A_{i j}}\right)
$$

对于药物 $d_j$，其 inter-domain feature 的定义如下：

$$
\tilde{H}_{d_j}^{l+1}=\sigma\left(\alpha^l \frac{\sum_i\left(H_{r_i}^l W^l \odot H_{d_j}^l W^l\right) A_{i j}}{\sum_i A_{i j}}+\left(1-\alpha^l\right) \frac{\sum_i H_{r_i}^l W^l A_{i j}}{\sum_i A_{i j}}\right)
$$

之后，我们将 intra-domain 和 inter-domain features 聚合在一起：

$$
\left[\begin{array}{a}
H_r^{l+1}\\
H_d^{l+1}
\end{array}\right]=
\left[\begin{array}{a}
\hat{H}_r^{l+1}+\tilde{H}_r^{l+1}\\
\hat{H}_d^{l+1}+\tilde{H}_d^{l+1}
\end{array}\right]
$$

其中，$H_r^{l+1}$ 代表药物的 fusion features，$H_d^{l+1}$ 代表疾病的 fusion features。

由于深层的 GCN 容易出现 over-smoothing 的问题，所以我们在每个模块的输入和输出之间增加了 residule skip connection，于是有：

$$
H^{l+1}=\hat{H}^{l+1}+\tilde{H}^{l+1}+H^l
$$

这里的 $H^{l+1}$ 就是第 $l$ 层的输出 embeddings。

我们还在每一个卷积层中加入了 attention mechanism 来进一步提高预测的准确性，药物和疾病的最终 embeddings 定义如下：

$$
\left[\begin{array}{a}
H_R\\
H_D
\end{array}\right]=
\sum_{l=1}\beta^lH^l
$$

其中，$\beta$ 是由神经网络自己学习的参数，初始值为 $1/L$。药物和疾病的最终 embeddings 等于每一层输出的 embeddings 乘上该层的 $\beta$ 的和。

### 2.2.2 Decoder

为了重构药物与疾病之间的关联，我们的 decoder $f(H_R,H_D)$ 定义如下：

$$
\hat{A}=f(H_R,H_D)=\operatorname{sigmoid}(H_RH_D^T)
$$

其中，$\hat{A}\in\mathbb{R}^{N\times M}$ 是概率矩阵。药物 $r_i$ 与疾病 $d_j$ 的关联性由矩阵 $\hat{A}$ 的第 $(i, j)$ 个元素给出。

### 2.2.3 Optimization

在数据集中，已知的疾病-药物关系的数量远少于未知的疾病-药物关系，所以，我们的模型采用加权二值交叉熵来作为损失函数：

$$
\operatorname{loss}=-\frac{1}{N\times M}(\gamma\times\sum_{(i,j)\in S^+}\log\hat{A}_{ij}+\sum_{(i,j)\in S^-}(1-\log\hat{A}_{ij}))
$$
其中，$S^+$ 表示所有已知疾病-药物关联的集合；$S^-$ 则是未知疾病-药物关联的集合。平衡因子 $\gamma$ 用来减少数据的不平衡性的影响，其公式是：

$$
\gamma=\frac{|S^-|}{|S^+|}
$$

其中，$|S^-|$ 和 $|S^+|$ 分别表示 $S^-$ 和 $S^+$ 中疾病-药物对的数量。