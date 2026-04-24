---
title: "[COMPT 0x02] 非确定性/封闭性/正则表达式→FA"
date: 2025-12-25
draft: false
math: true
categories:
  - 可计算性理论
tags: []
---

本节课的主题：

* 非确定性
* 封闭性
* 正则表达式→FA

## NFA

上一次课讲到 DFA 并不足以证明拼接和星运算下的封闭性，这一节课，我们将引入非确定有限状态自动机（Nondeterministic Finite Automata, NFA）来解决这一问题。

DFA 的每一步转移都是当前状态遇到一个输入字符然后去到一个**唯一确定**的后继状态；而 NFA 不一样，在 NFA 中，一个状态在读入同一个字符时，可转移的后继状态可能存在多个，甚至还允许出现 **$\epsilon$-转移**（$\epsilon$-transition）——即在不输入任何字符的情况下转移到下一个状态。

对于上述关于 NFA 的描述，我们可以想象到，输入同一个字符串，对应的转移路径也可能是多样的，也许有些路径能到达接受状态，有些不能，那么这种情况下算 NFA 接受了这个字符串，还是拒绝了呢？规定：**只要存在一条能到达接受状态的路径，就说这个 NFA 接受了这个字符串。**

下面给出 NFA 的形式化定义，一个 NFA 由一个五元组 $(Q,\Sigma,\delta,q_0,F)$ 定义，其中：

- $Q$：状态集
- $\Sigma$：字符集（字母表）
- $\delta$：转移函数，$\delta:Q\times\Sigma_\epsilon\rightarrow \mathcal{P}(Q)=\\{R|R\subseteq Q\\}$
- $q_0$：初始状态
- $F$：接受状态集

可以看到，NFA 的定义和 DFA 的定义的唯一区别就是转移函数 $\delta$。在 NFA 中，由于存在 $\epsilon$-转移，所以可输入字符集为 $\Sigma_\epsilon=\Sigma\cup\epsilon$。此外，由于每个状态在输入同一个字符的情况下可能存在多个转移状态，因此，$Q\times\Sigma_\epsilon$ 应该映射到 $Q$ 的幂集，即在状态 $q_i$ 下读入一个字符 $w$，存在 $(q_i,w)\rightarrow\\{q_j,q_{j+1},\dots\\}$.

NFA 模型并不对应任何物理机器，其在现实世界并不存在，引入它仅仅是因为它将有助于我们的证明。可是上一节课我们讨论的范畴是 DFA，正则的定义是能被 DFA 识别，难道 NFA 也行？没错，因为 NFA 能被转换成 DFA，所以为了严谨，我们需要先证明对于任何一个 NFA，都存在一个与之等价的 DFA，再继续使用 NFA 证明正则运算的封闭性。

## NFA→DFA

<font style="background: #FEEDD5;">📜定理</font>：如果存在 NFA 能识别语言 $A$，则 $A$ 是正则语言。

要证明这个定理，我们只需要证明对于每个 NFA $M$，都存在一个对应的 DFA $M^\prime$，使得当 $M$ 能识别 $A$ 时，$M^\prime$ 也能识别 $A$。换言之，我们需要找到将 NFA 转化成 DFA 的方法。由于 $\epsilon$-转移会增加转化的难度，我们这里只考虑不含 $\epsilon$-转移的 NFA，含 $\epsilon$-转移的情况如果感兴趣可以自行了解（编译原理其实有讲）。先定义 $M=\\{Q,\Sigma,\delta,q_0,F\\}$，$M^{\prime}=\\{Q^{\prime},\Sigma,\delta^{\prime},q_0^{\prime},F^{\prime}\\}$。具体的步骤为：

1. 初始时 $Q^{\prime}=\varnothing$
2. 将 $q_0$ 加入到 $Q^{\prime}$
3. 对于 $Q^\prime$ 中每一个状态， 分别找出每个输入下的转移状态集合，对于每个不存在于 $Q^\prime$ 中的集合，加入 $Q^\prime$。
4. 重复步骤 3，直到覆盖所有转移。
5. $F^{\prime}$ 为 $Q^\prime$ 中所有含有 $F$ 的状态。

详细过程和例子可以看 [How to convert NFA to DFA](https://www.educative.io/answers/how-to-convert-nfa-to-dfa) 这篇文章。下面给出 $M^\prime$ 的形式化定义：

* $Q^\prime=\mathcal{P}(Q)$
* $\delta^\prime(R,a)=\\{q|q\in\delta(r,a)\text{ for some }r\in R\\}$($R\in Q^\prime$)
* $q_0^\prime=\\{q_0\\}$
* $F^\prime=\\{R\in Q^\prime|R\cap F\neq\varnothing\\}$

<font style="background: #FEEDD5;">🏹练习</font>：在上述转换过程中，假设 $M$ 中状态的个数为 $n$，$M^\prime$ 中有多少个状态？答案是 $2^n$，从定义也能看出，$Q^\prime$ 是 $Q$ 的幂集，幂集的大小就是 $2^n$。当然在大多数情况下，真实构造出的 $M^\prime$ 并没有那么多状态，因为很多状态并无法到达。

既然知道了 NFA 是可以转化为 DFA 的了，那我们就可以放心地用 NFA 证明封闭性了。

## 并操作下的封闭性

<font style="background: #FEEDD5;">📜定理</font>：如果 $A_1$ 和 $A_2$ 是正则语言，那么 $A_1\cup A_2$ 也是正则语言。

虽然 $\cup$ 下的封闭性我们在上一节课已经讨论过了，但不妨用 NFA 重新证明一下，见识一下 NFA 的威力。在上节课的证明方法中，我们使用组合状态对的方法来合并两个的 DFA。有了 NFA 之后就不需要这么麻烦了，我们可以直接增加一个初始状态，再让这个初始状态通过两个 $\epsilon$-转移直接连接到另外两个 DFA 的初始状态，最后将原来的初始状态改为非初始状态就大功告成了（如下图所示）。

<center><img src="/images/Software Theory Fundamentals/02-01.jpg" alt="02-01" style="zoom: 50%;" /></center>

仔细观察这个 NFA，你会发现它和我们之前用状态对构造出的 DFA 是等价的。

## 拼接操作下的封闭性

<font style="background: #FEEDD5;">📜定理</font>：如果 $A_1$ 和 $A_2$ 是正则语言，那么 $A_1\circ A_2$ 也是正则语言。

考虑拼接两个 DFA $M_1$ 和 $M_2$，在没有 NFA 的情况下，这个过程无法完成，但现在有了 NFA，这一步就变得相当简单了，只需要把前一个 DFA 的接受状态通过 $\epsilon$-转移连接到后一个 DFA 的初始状态就完成了。

<center><img src="/images/Software Theory Fundamentals/02-02.jpg" alt="02-02" style="zoom: 50%;" /></center>

（*注意上图中的 $q_{1,1}$ 不应该保持为接受状态，这里画错了）

## ∗操作下的封闭性

<font style="background: #FEEDD5;">📜定理</font>：如果 $A$ 是正则语言，那么 $A^* $ 也是正则语言。

$A^* $ 是 $A$ 和自己进行零次或多次拼接，这就要求当自动机 $M^\prime$ 读完一个 $A$ 中的字符串之后，还能从头开始继续读入下一个 $A$ 中的字符串。那思路就很简单了，只需要增加一条从接受状态到原初始状态的 $\epsilon$-转移即可。此外，要注意 $A^* $ 还包括了 $\epsilon$，为了能让 NFA 直接接受 $\epsilon$，我们设置一个新的初始状态（这个初始状态同时也是接受状态），并增加从该初始状态到原初始状态的 $\epsilon$-转移，这样就保证了 NFA 能接受 $\epsilon$。

<center><img src="/images/Software Theory Fundamentals/02-03.jpg" alt="02-03" style="zoom: 50%;" /></center>

<font style="background: #FEEDD5;">🏹练习</font>：在上述转换过程中，如果 $M$ 有 $n$ 个状态，那么 $M^\prime$ 有多少状态？答案是 $n+1$，这个很简单，因为我们只是增加了一个初始状态而已。

## 正则表达式→NFA

通过证明三种正则运算下的封闭性，我们知道了正则语言经过正则运算得到的仍然是正则语言。我们将以上三种证明中用到的构造 FA 的方法称为封闭构造（Closure Construction）。但不要忘记我们的目标是证明 “正则表达式 $\Rightarrow$ FA”，现在让我们回到这个主题。其实要证明这一点，我们只需要找到将正则表达式转换为 NFA 的方法。

我们将正则表达式分为原子（atomic）表达式和复合（composite）表达式。我们首先尝试将原子表达式转换为 NFA，之后再尝试把复合表达式转换为 NFA。由于我们已经证明了三种正则运算的封闭性，因此后者的工作我们已经完成了，无非是将封闭性的证明再写一遍。

原子表达式一共有三种，由于它们的结构很简单，因此可以很容易转化为 NFA：

1. $R=a\text{ for }a\in\Sigma$

    <center><img src="/images/Software Theory Fundamentals/02-04.jpg" alt="02-04" style="zoom:50%;" /></center>

2. $R=\epsilon$

    <center><img src="/images/Software Theory Fundamentals/02-05.jpg" alt="02-05" style="zoom:50%;" /><center>

3. $R=\varnothing$

    <center><img src="/images/Software Theory Fundamentals/02-06.jpg" alt="02-06" style="zoom:50%;" /><center>

这样就完成了原子表达式的 NFA 转化。至于复合表达式的 NFA 转化，只需要重复我们在封闭性证明过程中用到的封闭构造就可以完成了。

由于 NFA 能被转换为 DFA，而能被 DFA 识别的语言是正则语言，可知 NFA 能识别的语言也是正则语言。而现在我们又成功把正则表达式转换成 NFA，就说明正则表达式描述的语言一定也是正则语言，于是得出下述定理：

<font style="background: #FEEDD5;">📜定理</font>：如果 $R$ 是一个正则表达式，并且 $A=L(R)$，那么 $A$ 是正则语言。

以上，我们完成了“正则表达式 $\Rightarrow$ FA”的证明。下一节课，我们将证明“FA $\Rightarrow$ 正则表达式”，完成我们的目标。
