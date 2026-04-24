---
title: "[COMPT 0x01] 有限状态自动机/正则表达式"
date: 2025-12-24
draft: false
math: true
categories:
  - 可计算性理论
tags: []
---

## 课程介绍

本笔记对应的课程是 MIT 的计算理论（Theory of Computation），原课程本来分为可计算性理论和复杂度理论两部分，而笔者所在学校开的这门课则由可计算性理论和软件形式化理论两部分组成，其中可计算性理论部分完全参考了 MIT 的课程，本笔记也仅收录可计算性理论部分。

这门课的名称为可计算性理论（Computability theory），可计算性理论从理论层面回答究竟什么样的问题能用计算机（算法）解决。现实中的计算机是一个相当复杂的物体，为了能从数学上讨论它，我们将计算机抽象为一些简单的模型，来帮助我们从不同角度理解计算机。这些抽象模型要比实际上的计算机简单得多，并且在能力以及模拟真实计算机的方法上也各不相同。我们将先从几个受限的模型讲起，这些模型无法完全模拟通用计算机的能力，但却有助于我们理解这门学科和我们所讨论的问题。最后我们会拓展到真正的通用计算机模型，也就是图灵机。

---

本节课的主题：

- 课程介绍
- FA 
- 正则表达式

## FA

让我们从一个非常简单的模型——有限状态自动机（Deterministic Finite Automata, FA）开始谈起。

一个 FA（记作 $M$）的计算流程如下：输入字符串 $s$，$M$ 的状态从初始状态 $q_0$（下标也可以从 1 开始记作 $q_{1}$）开始，之后每读入一个字符，便根据转移函数 $\delta$ 转移到下一个状态。当读完所有字符时，如果 $M$ 处于接受状态，则说明字符串 $s$ 被 $M$ 接受；否则，说明字符串 $s$ 被 $M$ 拒绝。

下面的例子展示了一个自动机 $M_1$，其中，单环圆圈表示一个状态，双环圆圈表示一个接受状态，箭头表示一次状态转移，箭头上的字母表示下一个读入的字符。

<center><img src="/images/Software Theory Fundamentals/01-01.jpg" alt="01-01" style="zoom:50%;" /></center>

对于 $M_1$，显然它能接受 01101，但无法接受 00101。事实上，如果你仔细观察它的结构就会发现，接受状态 $q_3$ 只有一个前驱 $q_2$，$q_2$ 也只有一个前驱 $q_1$，要从 $q_1$ 转移到 $q_2$，就必须读入一个 1，随后必须再立刻读入一个 1 才能达到 $q_3$。也就是说，$M_1$ 会接受所有包含 11 的字符串，我们定义其为 $A=\\{w|w\text{ contains substring 11}\\}$。对于任意字符串 $s\in A$，$M_1$ 都可以接受。**这种情况下，我们说 $M_1$ 能识别 $A$，或者说 $A$ 是 $M_1$ 的语言，记做 $A=L(M_1)$。**

下面给出 FA 的形式化定义。一个 FA  $M$ 由一个五元组 $(Q,\Sigma,\delta,q_0,F)$ 定义，其中：

- $Q$：状态集
- $\Sigma$：字符集（字母表）
- $\delta$：转移函数，$\delta:Q\times\Sigma\rightarrow Q$
- $q_0$：初始状态
- $F$：接受状态集

以 $M_1$ 为例，其定义为：

* $Q=\\{q_1,q_2,q_3\\}$

* $\Sigma=\\{0,1\\}$

* $F=\\{q_3\\}$

* $q_0=q_1$

* $\delta=$
    $$
    \begin{array}{c|cc}
    \text{} & \text{0} & \text{1} \\
    \hline
    q_{1} & q_{1} & q_{2} \\
    q_{2} & q_{1} & q_{3} \\
    q_{3} & q_{3} & q_{3} \\
    \end{array}
    $$

## 字符串和语言

由于我们已经提到了字符串和语言，所以我们有必要认识它们的定义。

字符串是**有限**字符序列（无限长的字符串不属于本课程的讨论范畴）；语言是有限或无限的字符串集合；空字符串 $\epsilon$ 是长度为 0 的字符串；空语言 $\varnothing$ 是不含任何字符串的字符串集合。（注意 $\epsilon\notin\varnothing$）

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📌定义</font>：$M$ 能接受字符串 $w=w_1w_2...w_n(\forall i,w_i\in\Sigma)$，如果存在一个状态序列 $r_0,r_1,r_2,\dots,r_n\in Q$ 满足：

* $r_0=q_0$
* $r_i=\delta(r_{i-1},w_i)$ for $1\le i\le n$
* $r_n\in F$

下面三种说法等价：

* $L(M)=\\{w|M\text{ accepts }w\\}$
* $L(M)$ 是 $M$ 的语言
* $M$ 可识别 $L(M)$

## 正则语言

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📌定义</font>：一个语言是**正则（regular）**的，如果存在某个 FA 能识别它。

以 $A$ 为例，由于 $A$ 能被 FA  $M_1$ 识别，因此 $A$ 是正则语言。

下面来看两道例题。第一道题是，给定语言 $B=\\{w|w\text{ has an even number of 1s}\\}$，$B$ 是否是正则的？要证明一个语言是正则的很简单，只要构造一个能识别它的 FA 就行了。对于 $B$，我们仅用两个状态就可以识别它，对应 FA 如下图所示：

<center><img src="/images/Software Theory Fundamentals/01-02.jpg" alt="01-02" style="zoom: 50%;" /></center>

第二道题是，给定语言 $C=\\{w|w\text{ has equal numbers of 0s and 1s}\\}$，问 $C$ 是否是正则语言？在这个语言中，我们需要确保 0 和 1 的数量相等，我们可以使用状态来记录数量，但 $C$ 并没有对 0 和 1 的数量进行限制，这意味着我们需要无数多个状态才能枚举每一种情况，显然，FA 是做不到的，因此 $C$ 不是正则语言。严谨的形式化证明需要用到泵原理，而这是后面的内容，举这个例子只是为了说明——并不是所有语言都能被 FA 识别，这类语言被称为**非正则语言**。

## 正则表达式

正则语言之间存在三种运算，我们称之为**正则运算**（Regular Operations）。设 $A$，$B$ 为两种语言，则有：

* 并（Union）: $A\cup B=\\{w|w\in A\text{ or }w\in B\\}$
* 拼接（Concatenation）: $A\circ B=\\{xy|x\in A\text{ and }y\in B\\}=AB$ (注意 $AB\neq BA$.)
* 星（Star）: $A^{* }=\\{x_1\dots x_{k}|\text{each }x_{i}\in A\text{ for }k\ge0\\}$ (例：令 $A=\\{good,boy\\}$, 有 $A^* =\\{\epsilon,good,boy,goodgood,boyboy,goodboy,goodgoodboy\dots\\}$. 注意 $\epsilon\in A^*$ 恒成立.)

**正则表达式**（Regular Expressions, REs）就是仅用上述三种运算构造出的恰好能被 FA 识别的语言集合的代数写法。正则表达式提供了除 FA 之外的描述一种语言的方法。例如，我们可以用正则表达式表示之前提到的语言 $A$：$\Sigma^* 11\Sigma^* $。下面我们提出一个课程目标，接下来的课程内容将围绕这个目标展开。

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">🎯目标</font>：证明 FA 与正则表达式是等价的。

要证明这件事并不简单，因为等价是双向的。要证明两者等价，我们既要证明 FA 具有正则表达式的表达能力，也要证明正则表达式具有 FA 的表达能力。要证明这两件事，我们只需要找到一个方法，使两者能够相互转化，即对任何一个 FA 而言，这里都存在一个与之等价的正则表达式，反之亦然。我们首先证明正则表达式 $\Rightarrow$ FA。为了证明这一点，我们需要学习有关正则运算的封闭性（Closure Properties）的概念[^1]。

一个集合在某个运算下是封闭的，是指对集合施加这个运算之后，结果仍然落在这个集合中。放在这门课的语境里，就是说，对正则语言施加正则运算之后，得到的仍然是正则语言。下面我们将依次讨论每种运算下的封闭性。

## 并操作下的封闭性

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：如果 $A_1$ 和 $A_2$ 是正则语言，那么 $A_1\cup A_2$ 也是正则语言。

证明这个定理就是证明 $A_1\cup A_2$ 是正则语言，思路就是构造一个 FA  $M$，使其能识别 $A_1\cup A_2$。凭空构造这样的自动机是很困难的，我们可以把问题转换成合并两个自动机 $M_1$ 和 $M_2$，其中 $M_{1}$ 能识别 $A_{1}$，$M_{2}$ 能识别 $A_{2}$。合并的具体方法可以看这篇文章：[Union Process in DFA](https://www.geeksforgeeks.org/theory-of-computation/union-process-in-dfa/)。整体思路还是比较好理解的，就是将两个自动机的初始状态合并为一个新的初始状态对，用一个二元组 $(q_{1,0},q_{2,0})$ 表示，其中第一个下标指出了当前状态来自于哪个 FA，第二个下标就是当前状态在原 FA 中的编号；读入字符时，按照两个自动机的转移函数，转移到对应的状态对。假设 $q_{1,0}$ 读入 1 后转移到 $q_{1,1}$，$q_{2,0}$ 读入 1 后维持不变，那么就有 $(q_{1,0},q_{2,0})\overset{1}{\rightarrow}(q_{1,1},q_{2,0})$。以此类推，继续输入各种不同的字符组合，直到覆盖尽可能多的状态对为止。最后再用普通状态表示这些状态对，例如用 $q_{0}^{\prime}$ 表示 $(q_{1,0},q_{2,0})$，就得到了一个符合标准形式的、合并后的自动机 $M$。

下面给出构造好的 $M$ 的形式化定义：

- $Q=Q_1\times Q_2=\\{(q,r)|q\in Q_1\text{ and }r\in Q_2\\}$
- $s_0=(q_0,r_0)$
- $\delta((q,r),\alpha)=(\delta_1(q,\alpha),\delta_2(r,\alpha))$
- $F=(F_1\times Q_2)\cup(Q_1\times F_2)$

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">🏹练习</font>：如果 $M_1$ 和 $M_2$ 分别有 $k_1$ 和 $k_2$ 个状态，那么合并后得到的 $M$ 有几个状态？答案是 $k_1\times k_2$。这很好理解，两个自动机中的状态组合为状态对，即状态之间求笛卡尔积，一共就有 $k_1\times k_2$ 种。但在实际中可能并没有这么多，并不是所有状态对都是合法（可达）的。

## 拼接操作下的封闭性

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：如果 $A_1$ 和 $A_2$ 是正则语言，那么 $A_1\circ A_2$ 也是正则语言。

同样的，我们要构造一个自动机来识别 $A_1\circ A_2$。因为是拼接操作，所以这一次我们考虑将两个自动机拼接起来，一个直观的想法是：直接将前面一个自动机的接受状态连接到后一个自动机的起始状态（如下图所示），但中间的转移条件是什么？

<center><img src="/images/Software Theory Fundamentals/01-03.jpg" alt="01-03" style="zoom: 50%;" /></center>

一个直观的想法是，从 $M_{1}$ 的接受状态到 $M_{2}$ 的初始状态之间的转移不应该需要任何输入，但当前我们讨论的自动机都是**确定有限状态自动机**（Deterministic Finite Automata, DFA），而 DFA 是不允许出现空转移的！因此 DFA 的能力并不足以证明拼接和星运算的封闭性。下一节课，我们将介绍更加强大的非确定有限状态自动机（Non-deterministic Finite Automata, NFA），使用 NFA 可以大大降低证明的难度。

[^1]:Closure 在集合论中一般翻译为闭包，这里我翻译为封闭，因为讨论的是运算的性质。
