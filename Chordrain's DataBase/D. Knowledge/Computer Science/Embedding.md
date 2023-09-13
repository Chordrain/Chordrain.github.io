#concept #cs #deepLearning 

# 什么是 embedding

在神经网络的相关论文中，我们经常会遇到 embedding 的概念，那么它到底是什么？

要想理解 embedding，就得先从 embed 说起。其实说白了，embed 就是将我们客观世界中存在的实体的性质转变成计算机可识别、可处理的语言的过程。例如一张图片，人看到的就是色彩，计算机要的则是数字，所以我们就得寻找方法将图片信息转换成计算机认得的信息，例如使用矩阵；再比如地形，地形涉及的信息量就更大的了，有海拔、气候、交通等等，如何将这些信息存储到电脑中，也是一个需要我们思考的 embed 过程。

embedding 是 embed 的名词，既可以指 embed 这一过程，也可以指数据的表示形式 (representation)。一般来讲，当客观世界的实体存储到计算机之后，人就再也看不懂了，人看不懂没有关系，计算机能看懂就行，于是我们可以对这些数据进行进一步提炼和加工，这就是神经网络在做的事。通过矩阵乘法，我们能对数据进行降维；同时也可以升维，构造更复杂的特征；此外，有些数据的特征矩阵可能会很稀疏，例如 NLP 中对语言文字的处理，将稀疏矩阵变为稠密矩阵也是 embedding 的用途之一，这些构造的结果也称为 embedding。

所以，简单理解，embedding 就是现实实体数据的低维表示，也是稀疏矩阵向稠密矩阵的变换过程。要想 model 的结果正确，就得首先确保数据的表示形式正确且妥当，model 以何种方式进行 embedding 是非常重要的。
# Reference

* [【CSDN】深度学习中 Embedding 的解释](https://blog.csdn.net/yuanmiyu6522/article/details/120930840)
* [【CSDN】什么是 embedding？]( https://blog.csdn.net/weixin_44493841/article/details/95341407 )