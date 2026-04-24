---
title: "[COMPT 0x08] 不可判定性的证明"
date: "2026-01-03 18:00:00"
draft: false
math: true
categories:
  - 可计算性理论
tags: []
---

在上节课我们留了一些问题，我们说 $EQ_{CFG}$ 和 $A_{TM}$ 是不可判定的，但是没有给出证明。这一节课，我们将学习如何证明语言的不可判定性（undecidability）。我们将使用对角化方法（the diagonalization method）来证明。但在学习对角化方法之前，我们需要先学习一些集合论的知识。

## 无限的大小

集合存在有限集和无限集之分。比较有限集的大小很简单，但是无限集呢？是否可以比较无限集的大小？虽然这一点听起来很不可思议，但在现代集合论创始人康托尔的理论中，无限集的大小确实是可以比较的。

让我们回忆一下高中数学知识。我们知道函数的本质是一种映射，记为 $f$，它将一个集合 $A$ 中的元素映射到另一个集合 $B$。如果对于一个确定的 $x \in A$ 存在一个唯一确定的 $f(x) \in B$，那么就称这个映射为**单射**。如果对于 $B$ 中的每一个 $y$，都至少存在一个 $x$ 使得 $f(x) = y$，那么就称这个映射为**满射**。如果一个映射既是单射，也是满射，那就称这个映射为**双射**。

康托尔提出了一种比较两个无限集合大小是否相等的方法，就是看能否建立起两个集合之间的双射。若能，则两个集合大小相等；否则，大小不等。

问：令 $\mathbb{N} = \\{ 1,2,3,\dots \\}$，$\mathbb{Z} = \\{ \dots,-2,-1,0,1,2,\dots \\}$，比较 $\mathbb{N}$ 和 $\mathbb{Z}$ 的大小。

虽然看起来 $\mathbb{Z}$ 中的元素要比 $\mathbb{N}$ 多，但反直觉的是，两者大小其实相等。我们可以建立如下映射关系：
$$
\begin{array}{c|c}
n & f(n) \\ \hline
1 & 0 \\
2 & 1 \\
3 & -1 \\
4 & 2 \\
5 & -2 \\
\vdots & \vdots \\
\end{array}
$$
第一列 $n$ 来自 $\mathbb{N}$，而第二列 $f(n)$ 覆盖整个 $\mathbb{Z}$，这张表可以无限延伸下去，构成一个双射，因此 $\mathbb{N}$ 和 $\mathbb{Z}$ 大小相等。

问：令 $\mathbb{Q}^{+} = \\{ m / n \mid m,n \in \mathbb{N} \\}$，证明 $\mathbb{Q}^{+}$ 和 $\mathbb{Z}$ 大小相等。

依然十分反直觉，但我们确实能做到在 $\mathbb{Q}^{+}$ 和 $\mathbb{Z}$ 之间建立双射，不过需要一些技巧。让我们先把 $\mathbb{Q}^{+}$ 中的元素列成一张二维表格，表格的行号代表分子，列号代表分母：
$$
\begin{array}{c|ccccc}
\mathbb{Q}^{+} & 1 & 2 & 3 & 4 & \dots \\ \hline
1 & 1/1 & 1/2 & 1/3 & 1/4 & \dots \\
2 & 2/1 & 2/2 & 2/3 & 2/4 & \dots \\ 
3 & 3/1 & 3/2 & 3/3 & 3/4 & \dots \\ 
4 & 4/1 & 4/2 & 4/3 & 4/4 & \dots \\
\dots & \dots & \dots & \dots & \dots & \dots \\
\end{array}
$$
如果我们按行去映射，比如说 $1/1 \to 1$，$1/2 \to 2$，$1/3 \to 3$……那第二行及其之后的元素将永远不会轮到；同理，如果按列去映射，第二列及其之后的元素将永远不会轮到。康托尔想出了一种巧妙的折线法，从左上角的元素开始，按斜线枚举，即构建如下映射：
$$
\begin{array}{c|cc}
n & f(n) \\ \hline
\mathbb{N} & \mathbb{Q}^{+} \\
1 & 1/1 \\ 
2 & 2/1 \\ 
3 & 1/2 \\
4 & 3/1 \\
5 & 1/3 & \cancel{2/2} \\
\dots & \dots \\
\end{array}
$$
注意在枚举过程中要去掉重复的分数，例如 1/1 和 2/2 就是重复的，否则不满足单射。最终，按照这种折线枚举法，所有 $\mathbb{Q}^{+}$ 中的元素都能被映射到，由此一来我们就构建了 $\mathbb{N}$ 和 $\mathbb{Q}^{+}$ 之间的双射，证明两者大小相等。

## 可数集

通过上面两个例子，我们可以总结出一个规律，那就是如果两个集合能被枚举，那就可以建立两者的双射。这种可以被枚举的性质被称为可数（countable），下面给出可数的定义。

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📌定义</font>：一个集合是可数的，当且仅当它是有限集合，或者它的大小与 $\mathbb{N}$ 相等。

要证明一个无限集合是可数的，就要拿它与 $\mathbb{N}$ 做比较，$\mathbb{N}$ 代表了最基础、最小等级的无限。那如果一个集合不可数呢？我们如何证明它是不可数的？

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：实数集 $\R$ 是不可数集合。

现在，我们尝试证明 $\R$ 是不可数的。这就要引出我们一开始讲到的对角化方法了！

我们假设 $\R$ 是可数的，并尝试枚举它，就会得到下面这张表格：
$$
\begin{array}{c|c}
n & f(n) \\ \hline
\mathbb{N} & \R \\
1 & 2.718281828 \dots \\ 
2 & 3.141592653 \dots \\ 
3 & 0.000000000 \dots \\
4 & 1.414213562 \dots \\
5 & 0.142857242 \dots \\
6 & 0.207879576 \dots \\
7 & 1.234567890 \dots \\
\dots & \dots \\
\end{array}
$$
而我们能证明，不论如何枚举，一定存在一个数 $x \in \R$ 且 $x$ 不存在于这张表中，而我们不仅能证明 $x$ 存在，甚至还能找出这个 $x$！让我们来看看如何做到。

我们暂且忽略整数部分，只关注小数部分。毕竟举出一个小数部分不存在于表中的数也能完成证明。我们将应用对角化方法构造出那个不存在于表中的 $x$。首先，我们聚焦于第一个数字的小数后第一位，我们能看到这一位是 $7$，因此，我们选取一个不等于 $7$ 的数字作为 $x$ 的小数点后第一位数，得到 $x = 0.8$，如此一来就保证了 $x$ 与表格中第一个数绝对不相等；下面我们来看表格中第二个数字的小数后第二位，我们能看到这一位是 $4$，因此，我们选取一个不等于 $4$ 的数字作为 $x$ 的小数点后第二位数，得到 $x = 0.85$，如此一来就保证了 $x$ 与表格中第二个数绝对不相等；以此类推，我们不断重复这一过程：令 $x$ 的第 $i$ 位小数与表格中第 $i$ 个数的第 $i$ 位小数不相等，最终将得到一个实数 $x$，它与表格中任何一个数都不相等，即它不存在于表中。因此表格建立的映射不满足满射的关系，证明 $\R$ 不可数！整个过程如下所示。
$$
\begin{array}{c|c}
n & f(n) \\ \hline
\mathbb{N} & \R \\
1 & 2.\dot{7}18281828 \dots \\ 
2 & 3.1\dot{4}1592653 \dots \\ 
3 & 0.00\dot{0}000000 \dots \\
4 & 1.414\dot{2}13562 \dots \\
5 & 0.1428\dot{5}7242 \dots \\
6 & 0.20787\dot{9}576 \dots \\
7 & 1.234567\dot{8}90 \dots \\
\dots & \dots \\
\end{array}
\quad \Rightarrow \quad
x=0.8516182 \dots \text{ is not on the list}
$$
上面的结论告诉我们实现自然数集到实数集的满射是不可能的，由此可见，实数集比自然数集要大。

---

下面我们将建立以上内容与我们之前所学内容的联系。令 $\mathcal{L}$ 为所有语言的集合，我们有如下定理：

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：所有语言的集合 $\mathcal{L}$ 是不可数集合。

我们可以通过建立一个 $\mathcal{L}$ 到 $[0,1)$ 上的实数集的双射来证明这一点。为方便起见，我们令 $\Sigma=\\{ 0, 1\\}$，那么 $L(\mathcal{L})=\Sigma^{* }$。我们按字典序枚举出 $L(\mathcal{L})$ 中所有的字符串：
$$
\begin{array}{c|ccccccccc}
\Sigma^{* } & \epsilon & 0 & 1 & 00 & 01 & 10 & 11 & 000 & \dots  \\ \hline
&
\end{array}
$$
那么对于任意一个语言 $A \in \mathcal{L}$，其只不过是覆盖了 $\Sigma^{* }$ 中的全部或部分字符串，例如：
$$
\begin{array}{c|ccccccccc}
\Sigma^{* } & \epsilon & 0 & 1 & 00 & 01 & 10 & 11 & 000 & \dots  \\ \hline
A \in \mathcal{L} &  & 0 &  & 00 & 01 & & & & \\ \hline
&
\end{array}
$$
在表格的第三行中，我们用 0 表示某个字符串 $w \in \Sigma^{* }$ 不属于 $A$，用 1 表示某个字符串 $w \in \Sigma^{* }$ 属于 $A$，最终将得到一个 $A$ 的二进制编码，如下所示：
$$
\begin{array}{c|ccccccccc}
\Sigma^{* } & \epsilon & 0 & 1 & 00 & 01 & 10 & 11 & 000 & \dots  \\ \hline
A \in \mathcal{L} &  & 0 &  & 00 & 01 & & & & \dots \\ \hline
f(A) & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 0 & \dots
\end{array}
$$
由此我们建立了一个 $f$，它能将 $\mathcal{L}$ 中的语言 $A$ 映射为一个二进制序列。如果我们把这个二进制序列看作是一个二进制小数，即 $f(A) \to 0.f(A)$，那不就得到了从 $A$ 到 $[0,1)$ 上的实数集的双射吗？！而实数集又是不可数的，从而得证 $\mathcal{L}$ 是不可数的。

我们从这个例子中能得到一个有趣的洞察：虽然 $\mathcal{L}$ 是不可数的，但字符串集合 $\Sigma^{* }$ 却是可数的。这个洞察将有助于我们证明下面这个定理：

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：所有图灵机的集合 $\mathcal{M}$ 是可数集合。

上面这条定理告诉我们图灵机的种类是可数的，如何证明？我们说我们可以用自然语言来描述一个图灵机，这里的描述也不过是一个字符串，那么所有图灵机的集合就可以表示为所有这些字符串的集合，表示为 $\\{ \langle M \rangle \mid M \text{ is a Turing Machine} \\}$。这个集合无非就是 $\Sigma^{* }$ 的一个子集，所以它也是可数的，证毕。

上面的结论揭示了这样一件事：语言的集合是不可数的，而图灵机的集合是可数的，这意味着语言的数量比图灵机的数量要多，因此，这里必然存在一些语言，它们没有对应的图灵机来判定它们，因此下面这条定理成立。

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：必定存在一些语言是不可判定的。

## 不可判定性证明

下面我们将证明上节课遗留下来的问题：证明 $A_{TM} = \\{ \langle M,w \rangle \mid M \text{ is a TM and } w \in L(M) \\}$ 是不可判定的。

我们仍使用反证法。假设 $A_{TM}$ 是可判定的，那么这里必定存在一个判定器 $H$，它能判定输入字符串 $w$ 是否能被 $M$ 接受：
$$
H \text{ on } \langle M, w \rangle =
\begin{cases}
\text{Accept} &\text{ if } M \text{ accepts }w,\\
\text{Reject} &\text{ if not}.
\end{cases}
$$
接下来的任务就是证明这个 $H$ 不可能存在，我们将构造一个图灵机 $D$ 来找出矛盾。注意，$D$ 要做的事情可能看起来非常诡异，但也正是这种反常的例子才能让我们找到矛盾所在。我们先给出 $D$ 的描述：
$$
\begin{aligned}
D = &\text{“On input }\langle M \rangle \\
&\text{1. Simulate } H \text{ on input } \langle M, \langle M \rangle \rangle.\\
&\text{2. Accept if } H \text{ rejects; Reject if }H\text{ accepts.”}\\
\end{aligned}
$$
简单来说，$D$ 的输入为一个图灵机的描述 $\langle M \rangle$，它会将这个图灵机 $M$ 和它的描述 $\langle M \rangle$ 输入给 $H$，然后做出与 $H$ 相反的判断。而 $H$ 会接受 $\langle M \rangle$，当且仅当 $M$ 接受 $\langle M \rangle$，因此，如果我们将 $\langle M \rangle$ 输入 $D$，会得到：
$$
D \text{ accepts } \langle M \rangle \text{ iff } M \text{ doesn't accept }\langle M \rangle.
$$
而如果我们给 $D$ 输入 $\langle D \rangle$，那就会得到下面这种诡异的局面：
$$
D \text{ accepts } \langle D \rangle \text{ iff } D \text{ doesn't accept }\langle D \rangle.
$$
这是一个明显的悖论，由此我们找到了矛盾，证明这种 $H$ 不可能存在，因此 $A_{TM}$ 不可被判定。

这其实是一种对角化方法的应用，我们可以构造下面这张表格：
$$
\begin{array}{c|cccccc}
 & \langle M_{1} \rangle & \langle M_{2} \rangle & \langle M_{3} \rangle & \langle M_{4} \rangle & \dots & \langle D \rangle \\ \hline
M_{1} & acc & rej & acc & rej & \dots & \\
M_{2} & acc & acc & acc & acc & \dots & \\ 
M_{3} & rej & rej & rej & rej & \dots & \\ 
M_{4} & rej & rej & acc & acc & \dots & \\
\dots & \dots & \dots & \dots & \dots & \dots & \\
D &  & & & &  & \\
\end{array}
$$
表格的行代表图灵机，列代表输入，中间的每个格子代表对应图灵机在对应输入下的行为，表中的数据是随意捏造的，仅做示例，我们现在要填补对后一行的数据。我们已经知道，$D$ 会做出与 $M$ 被输入自身时相反的判断，例如表中，$M_{1}$ 输入 $\langle M_{1} \rangle$ 时会接受，那么当 $D$ 输入 $\langle M_{1} \rangle$ 时就会拒绝，以此类推。也就是说，我们只需要将对角线上的结果取反，就是 $D$ 的结果。那么就有：
$$
\begin{array}{c|cccccc}
 & \langle M_{1} \rangle & \langle M_{2} \rangle & \langle M_{3} \rangle & \langle M_{4} \rangle & \dots & \langle D \rangle \\ \hline
M_{1} & acc & rej & acc & rej & \dots & \\
M_{2} & acc & acc & acc & acc & \dots & \\ 
M_{3} & rej & rej & rej & rej & \dots & \\ 
M_{4} & rej & rej & acc & acc & \dots & \\
\dots & \dots & \dots & \dots & \dots & \dots & \\
D & rej & rej & acc & rej & \dots  & ? \\
\end{array}
$$
发现了吗？最右下角也在对角线上，因此要填入与自己相反的值，由此产生了悖论。所以说，这也是一种对角化方法的应用。

---

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜定理</font>：如果 $A$ 和 $\overline{A}$ 都是图灵可识别的，那么 $A$ 是图灵可判定的。

这里的 $\overline{A}$ 意思是 $A$ 的补集。我们假设 $M_{1}$ 能识别 $A$，$M_{2}$ 能识别 $\overline{A}$，那么我们能构造出判定器 $T$ 来判定 $A$：
$$
\begin{aligned}
T = &\text{“On input } w \\
&\text{1. Run } M_{1} \text{ and } M_{2} \text{ on } w \text{ in parallel until one accepts.}\\
&\text{2. If } M_{1} \text{ accepts then accept;}\\
&\quad\text{If } M_{2} \text{ accepts then reject.”}\\
\end{aligned}
$$
由于 $A$ 和 $\overline{A}$ 是互补的关系，所以对于任意输入 $w$，$M_{1}$ 和 $M_{2}$ 中至少有一个能跑出结果，从而得到 $T$ 是可停机的，证明了 $A$ 是图灵可判定的。

事实上，如果把 $T$ 的行为反一下，变成：
$$
\begin{aligned}
T^{\prime} = &\text{“On input } w \\
&\text{1. Run } M_{1} \text{ and } M_{2} \text{ on } w \text{ in parallel until one accepts.}\\
&\text{2. If } M_{2} \text{ accepts then accept;}\\
&\quad\text{If } M_{1} \text{ accepts then reject.”}\\
\end{aligned}
$$
你会发现，$T^{\prime}$ 就是 $\overline{A}$ 的判定器，也就是说 $\overline{A}$ 也是图灵可判定的。这也证明了语言的**可判定性在补集操作下是封闭的**。

<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜推论</font>：$\overline{A_{TM}}$ 是不可识别的。

有了上面那条定理，这条推论就很容易证明了：由于我们已经证明了 $A_{TM}$ 是不可判定但却是可识别的，因此 $\overline{A_{TM}}$ 必定是不可识别的，不然就会推出矛盾。

值得注意的是，虽然可识别性在补操作下不是封闭的，但在交、并、拼接、星操作下都是封闭的。感兴趣可以自己证明。

## 规约方法

除了上面的方法，我们还可以使用规约的方法来证明不可判定性。所谓规约，就是把问题转换成一个已经解决的问题，通过得出与已有结论相悖的结果证明不可判定性。来看下面的例子。

这个语言定义了著名的停机问题：$HALT_{TM}=\\{ \langle M, w \rangle \mid M \text{ halts on input } w\\}$。停机问题即判定图灵机在输入 $w$ 上是否能停机的问题，这个问题是不可判定的，我们将使用规约方法来证明它的不可判定性。

首先，我们假设 $HALT_{TM}$ 是可判定的，令 $R$ 为 $HALT_{TM}$ 的判定器。那么我们将可以通过 $R$ 来证明 $A_{TM}$ 是可判定的（而我们知道这是错的）。

我们将构造一个判定器 $S$ 来判定 $A_{TM}$，怎么做？首先，$R$ 能够判定一个字符串 $w$ 是否能在图灵机 $M$ 上停机，那么我们可以先用它来测试一下，看看输入的 $w$ 是否能在 $M$ 上停机，如果能停机，如果不能就拒绝，如果能，那事情就有意思了，我们就可以直接返回 $M$ 的判定结果，这样使得 $S$ 成为了一个判定器，其能判定 $A_{TM}$。$S$ 的描述如下：
$$
\begin{aligned}
S = &\text{“On input } w \\
&\text{1. Use } R \text{ to test if } M \text{ halts on } w \text{. If not, reject.}\\
&\text{2. Simulate } M \text{ on } w.\\
&\text{3. Accept if } M \text{ accepts;}\\
&\quad\text{Reject if } M \text{ rejects.”}\\
\end{aligned}
$$
由于我们已经知道 $A_{TM}$ 是不可判定的，上述推论与我们现有知识相悖，因此 $R$ 不可能存在，所以 $HALT_{TM}$ 是不可判定的。
