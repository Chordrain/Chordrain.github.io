#machineLearning 
# 01 概述

<div class="row" style="display: flex; justify-content: space-between; padding: 10px 5px 5px;">
	<div class="column" style="width: 20%;">
		<img src="D:\24159\Documents\Obsidian\Library\笔记库\专业课\DeepLearning\attachment\knn-01.png">
	</div>
	<div class="column" style="width: 75%;">
		想象一个分类问题：假设在一张图中有 A 和 B 这两种点，现在新增一个点，如何判断这个点是属于 A 还是属于 B。<br><br>
		比较直观的想法是，这个<u>点离哪堆点更近，它就属于哪种点</u>。<br><br>
		那么如何衡量这里的远近？
	</div>
</div>

<div class="row" style="display: flex; justify-content: space-between; padding: 10px 5px 5px;">
	<div class="column" style="width: 75%;">
		我们的想法是：选择离它最近的 3 个点，观察这 3 个点中哪种点占据多数，这个点就属于哪种点。<br><br>
		依照这一想法，我们能画出右图，发现在离该点最近的 3 个点中，B 点占据了多数，所以按照少数服从多数的原则，该点应该属于 B 点。
	</div>
		<div class="column" style="width: 20%;">
		<img src="D:\24159\Documents\Obsidian\Library\笔记库\专业课\DeepLearning\attachment\knn-02.png">
	</div>
</div>

* 上述过程就是 k 近邻法。k 近邻法有 3 个要素：
	1. 距离远近：**距离度量**
	2. 参考点数量：**k 值**
	3. 服从多数：**分类决策规则**

* 于是乎，我们可以将 k 近邻法总结为两步：
	1. 根据距离度量和 k 值计算 N<sub>k</sub>(x)
	2. 根据分类决策规则计算类别 y

# 02 距离度量

## 2.1 欧式距离

那么我们用何种指标来度量两个点之间的距离呢？最常用的度量方法就是**欧氏距离**，其公式如下：
$$
d_{12}=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}
$$
当然，上面的公式只是对二维情况的讨论，如果扩展到多维的情形则如下：
$$
d_{12}=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$
## 2.2 曼哈顿距离

曼哈顿街区的分布呈现网格状，如果你要从一个十字路口开到另一个十字路口，那你要走的距离肯定不会是直线距离，而是如下图红、黄、蓝线所示：

![](S.%20Static/Images/knn-03.png)

此时，两点之间的距离公式就变成了
$$
d_{12}=|x_1-x_2|+|y_1-y_2|
$$
扩展到多维情况下就是
$$
d_{12}=\sum_{k=1}^n|x_{1k}-x_{2k}|
$$
## 2.3 切比雪夫距离

在国际象棋中，国王可以随意移动到周围 8 个格子，那么如果国王要从 $(x_1, y_1)$ 走到 $(x_2, y_2)$ 则最少需要走多少步？这个距离就叫做切比雪夫距离。显然在这种情况下，距离的计算公式就变成了
$$
d_{12}=\max(|x_1-x_2|, |y_1-y_2|)
$$
扩展到多维空间的情况下就是
$$
d_{12}=\max(|x_{1i}-x_{2i}|)
$$
## 2.4 闵可夫斯基距离

闵可夫斯基距离简称闵氏距离，它是一类距离的统称，上面讲到的欧氏距离、曼哈顿距离和切比雪夫距离都是闵氏距离的一种，其公式如下：
$$
d_{12}=\sqrt[p]{\sum_{k=1}^{n}|x_{1k}-x_{2k}|^p}
$$
* 其中 p 是一个变参数
	1. 当 p=1 时，就是曼哈顿距离；
	2. 当 p=2 时，就是欧式距离；
	3. 当 p=∞时，就是切比雪夫距离 (因为此时相对小的值趋于无穷小，大的值趋于无穷大)；

* 闵氏距离都有两个明显的**缺点**：
	1. 未考虑各个分量的量纲，也就是将“单位”相同看待了。例如这里有三个二维的样本 (身高 cm，体重 kg)：a (180, 50), b (190, 50), c (180, 60)，经过计算，a 和 b 的闵氏距离等于 a 和 c 的闵氏距离，但实际上身高的 10 cm 不能和体重的 10 kg 画上等号；
	2. 同时，它未考虑到各个分类的分布 (期望、方差) 可能是不同的。
## 2.5 标准化欧式距离

标准化欧式距离是针对欧式距离的缺点进行的一种改进，即先对每个分量进行标准化，再计算其欧式距离。假设样本集 X 的均值为 m，标准差为 s，则标准化变量的公式为：
$$
X^*=\frac{X-m}{s}
$$
标准化欧氏距离的公式为：
$$
d_{12}=\sqrt{\sum_{k=1}^{n}(\frac{x_{1k}-x_{2k}}{s_k})^2}
$$
## 2.6 余弦距离

几何中，夹角余弦可用来衡量两个向量方向的差异；机器学习中，借用这一概念来衡量样本向量之间的差异。在二维空间中，向量 A (x<sub>1</sub>, y<sub>1</sub>) 与向量 B (x<sub>2</sub>, y<sub>2</sub>) 的夹角余弦公式为：
$$
\cos\theta=\frac{x_1x_2+y_1y_2}{\sqrt{x_1^2+y_1^2}\sqrt{x_2^2+y_2^2}}
$$
则在 n 维空间中，两个样本点的夹角余弦为：
$$
\cos\theta=\frac{\sum_{k=1}^{n}x_{1k}x_{2k}}{\sqrt{\sum_{k=1}^nx_{1k}^2}\sqrt{\sum_{k=1}^nx_{2k}^2}}
$$
# 03 k 值选择

