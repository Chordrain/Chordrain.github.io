---
title: "[COMPT 0x04] 上下文无关文法/下推自动机/CFG↔PDA"
date: 2025-12-28
draft: false
math: true
categories:
  - 可计算性理论
tags: []
---

本节课的主题：

- 上下文无关文法
- 下推自动机
- CFG ↔ PDA

## CFG

上一节里我们已经简单介绍了 CFG，也介绍了一个例子 $G_1$，如下左所示。可以发现，$G_1$ 中存在两条规则，它们的左值是一样的，所以我们可以使用一种简写形式，如下右所示。
$$
\begin{aligned}
G_1:\quad S & \rightarrow 0S1\\
S & \rightarrow R\\
R & \rightarrow \epsilon
\end{aligned}
\begin{aligned}
\quad\quad\quad
\text{Shorthand:}\quad S & \rightarrow 0S1 | R\\
R & \rightarrow \epsilon\\
\\
\end{aligned}
$$
由上下文无关文法生成的语言称为上下文无关语言（Context Free Language, CFL），反过来，有如下定义：

<font style="background: #FEEDD5;">📌定义</font>：$A$ 是一个 CFL，当存在一个 CFG $G$ 使得 $A=L(G)$。

我们可以用一棵树来表示一个字符串的推导过程，下图展示了使用 $G_1$ 生成字符串 $0011$ 的过程：

<center><img src="/images/Software Theory Fundamentals/04-01.jpg" alt="04-01" style="zoom:80%;" /></center>

<font style="background: #FEEDD5;">📌定义</font>：一个 CFG $G$ 是一个四元组 $(V,\Sigma,R,S)$，其中：

* $V$：变量的有限集合
* $\Sigma$：终结符的有限集合
* $R$：规则的有限集合，规则的形式为 $V\rightarrow(V\cup\Sigma)^*$
* $S$：开始变量的有限集合

下面是几种常用写法：

* $u\Rightarrow v$：表示在 $G$ 中存在一条从 $u$ 到 $v$ 的替换规则
* $u\overset{*}{\Rightarrow}v$：表示 $u$ 可以通过一系列替换规则最终变成 $v$
* $L(G)=\\{w|w\in\Sigma^* \text{ and }S\overset{*}{\Rightarrow} w\\}$

以防你疑惑什么叫上下文无关：所谓上下文无关，就是指在生成字符串时，它不需要管前面和后面的字符是什么，只需要不断用替换规则进行替换就行了。而下面这个例子就展示了什么叫上下文有关：
$$
\begin{aligned}
B & \rightarrow 0B1|\epsilon\\
B1 & \rightarrow 1B
\end{aligned}
$$
这显然不是一个 CFG，关注到第二条规则 $B1 \rightarrow 1B$，它要求当 $B$ 下文出现一个 1 时才能替换，这就是上下文有关了。而 CFG 应当只考虑变量本身。

下面我们来看另一个例子：
$$
\begin{aligned}
G_2:\quad E & \rightarrow E+T|T\\
T & \rightarrow T\times F|F\\
F & \rightarrow (E)|a
\end{aligned}
$$
我们可以用 $G_2$ 来推导出很多式子，例如 $a+a\times a$，下面是这个式子的解析树（parse tree）：

<center><img src="/images/Software Theory Fundamentals/04-02.jpg" alt="04-02" style="zoom:80%;" /></center>

解析树其实蕴含了一些隐藏信息。你可以发现，乘法永远会比加法的优先级更高，因为它的深度更深，计算上肯定是它更先被计算。

我们来对比一下 $G_2$ 和 $G_3$：
$$
\begin{aligned}
G_2:\quad E & \rightarrow E+T|T\\
T & \rightarrow T\times F|F\\
F & \rightarrow (E)|a
\end{aligned}
\quad\quad\quad
\begin{aligned}
G_3:\quad E & \rightarrow E+E|E\times E|(E)|a\\\\\\
\end{aligned}
$$
如果你尝试一下的话，就会发现 $G_2$ 和 $G_3$ 是等价的，它们能识别同一种语言，也就是有 $L(G_1)=L(G_2)$。但这是否说明它们没有区别？不是！如果你仔细的话就会发现，$G_3$ 是没有优先级的！例如同样是生成 $a+a\times a$，$G_3$ 的解析树有两种：

<div style="display: flex; flex-direction: row; justify-content: center;"><img src="/images/Software Theory Fundamentals/04-03.jpg" alt="04-03" style="zoom:80%;" /><img src="/images/Software Theory Fundamentals/04-04.jpg" alt="04-04" style="zoom:80%;" /></div>

可以看到，在 $G_3$ 中，先算加法或先算乘法都是可以的，并没有优先级。我们说，如果一个字符串对应两棵不同的解析树，那么这个 CFG 就是有二义（ambiguous）的。从这两棵树可以看出，$G_3$ 是有二义的。在编程语言里，二义性是不允许出现的，任何一门编程语言都需要消除二义性，例如 Python 使用缩进，C 使用花括号。

正则表达式可以转换为 CFG，例如下面这个例子：
$$
\begin{aligned}
0\Sigma^*1\\\\\\\\\\\\
\end{aligned}
\quad\quad\quad
\begin{aligned}
S & \rightarrow 0B1\\
B & \rightarrow BB|0|1|\epsilon\\\\
S & \rightarrow A1\\
A & \rightarrow 0B\\
B & \rightarrow BB|0|1|\epsilon\\
\end{aligned}
$$
对于正则表达式 $0\Sigma^*1$（$\Sigma=\\{0,1\\}$），存在两个与之对应的 CFG，这说明正则表达式与对应的 CFG 不唯一。而且我们能发现，只要我们想，**我们总能让规则的右边最多剩下两个变量**。例如在上例中，第一个 CFG 的第一条规则 $S\rightarrow0B1$ 的右边存在三个变量，而通过将 $0B$ 单独提出来用另一个中间变量 $A$ 来替代，就让右边只剩下了 $A1$ 两个变量。

需要注意的是，虽然正则表达式都可以转换为 CFG（也可以说所有正则语言都可以用 CFG 表示），但并不是 CFG 生成的语言都是正则语言。这一点也证明了 CFG 是比正则表达式更强大的模型。

## PDA

既然正则表达式对应的更强大的模型是 CFG，那么 FA 呢？是否存在一个自动机和 CFG 一样强大？有的兄弟，有的，隆重介绍**下推自动机**（Pushdown Automata, PDA）。

PDA 类似于 NFA，但要比 NFA 多一个外置装置——**下推栈**（Pushdown Stack）。PDA 每走一步，可以做以下操作：

* 读取：
    * 读取输入字符
    * 读取自身状态
    * 读取并删除栈顶字符
* 操作：
    * 修改自身状态
    * 往栈中添加一个字符

PDA 可以用这个栈来存储额外信息，以此来决定当前要不要转移，要转移到哪里，因此 PDA 要比一般的 FA 更加强大。

<font style="background: #FEEDD5;">📌定义</font>：一个 PDA 由一个六元组 $(Q,\Sigma,\Gamma,\delta,q_0,F)$ 定义，其中：

* $Q$：状态集

* $\Sigma$：输入的字符集

* $\Gamma$：栈的字符集

* $\delta$：$Q\times\Sigma_{\epsilon}\times\Gamma_{\epsilon}\rightarrow\mathcal{P}(Q\times\Gamma_{\epsilon})$

    ​	例：$\delta(q,a,c)=\\{(r_1,d),(r_2,e)\\}$

* $q_0$：初始状态

* $F$：接受状态集

回忆我们之前讨论的语言 $D=\\{0^k1^k|k\ge0\\}$，我们证明过它是非正则的，FA 无法无限计数，因此 FA 无法表达它，但 PDA 不一样。我们可以用 PDA 的栈来进行计数，具体的操作是：

1. 每次读到 0，就压入栈，直到读到 1；
2. 每次读到 1，就从栈顶推出一个 0；
3. 输入结束时，如果栈为空，进入接受状态。

再来考虑 $B=\\{ww^{R}|w\in\\{0,1\\}^*\\}$（$w^R$ 表示 $w$ 的反转字符串），这里我们需要使用到 PDA 的非确定性，我们这样使用栈：

1. 应用非确定性，每次可以选择以下其中一种操作：
    1. 读入字符并压入栈；
    2. 读入字符，并将其与栈顶字符比较，如果不一样就拒绝；
2. 输入结束时，如果栈为空， 进入接受状态。

非确定性再结合栈，这赋予了 PDA 作弊般的表达能力。比如说，我们可以用 PDA 来识别任何正则表达式所表达的语言。怎么做？在一开始，PDA 一个字符都不读，它只是通过 $\epsilon$-转移进行非确定的状态转移。每次转移，它都通过正则表达式的规则生成字符，并将字符压入栈。结合 PDA 的非确定性，我们可以让 PDA 不断地去尝试构建各种各样的字符串，直到覆盖全部可能的字符串，压入栈中。待生成完成后，开始读入字符，并和栈顶元素进行比较，如果读到最后栈刚刚好为空，进入接受状态；如果没有任何一条路径能接受输入字符串，那就拒绝。

但 PDA 也不是万能的，例如语言 $\\{ww|w\in\\{0,1\\}^*\\}$，PDA 无法表达它，这一点我们之后会证明，然后引出另一种全能的模型。

## CFG→PDA

CFG 和 PDA 其实是等价的，接下来我们将会学习两者的相互转换。

<font style="background: #FEEDD5;">📜定理</font>：如果 $A$ 是一个 CFL，那么存在某个 PDA 能识别它。

证明上述定理的思路，就是把 $A$ 对应的 CFG 转换成 PDA。但是怎么做？你可能会觉得棘手的是选择哪条规则进行替换，但其实这一点可以用非确定性解决。真正有难度的部分在于如何跟踪中间结果，那这部分就是栈的工作。PDA 依赖非确定性去猜测使用哪条规则进行替换，使用栈保存中间结果，待栈中只剩下终结符，去比较栈中内容与输入字符串是否相等，相等就进入接受状态，不相等就拒绝。 

这件事说起来很简单，但做起来会有一些微妙的问题。我们说使用栈来记录中间状态，意思是指记录推导过程。我们以 $G_2$ 为例，回忆一下推导 $a+a\times a$ 的过程，我们可以预见栈中内容的变化过程为：
$$
\begin{aligned}
E
\end{aligned}
\quad\Rightarrow\quad
\begin{aligned}
E\\
+\\
T
\end{aligned}
\quad\Rightarrow\quad
 \begin{aligned}
T\\
+\\
T\\
\times\\
F
\end{aligned}
\quad\Rightarrow\quad
\begin{aligned}
F\\
+\\
F\\
\times\\
a
\end{aligned}
\quad\Rightarrow\quad
\begin{aligned}
a\\
+\\
a\\
\times\\
a
\end{aligned}
$$
但是这里存在一个问题：栈只允许对栈顶进行操作，我们无法直接操作栈顶以下的字符。好在这个问题是可以解决的：我们每次只替换栈顶的变量，一旦栈顶的变量变成终结符，我们就立马把它从栈顶弹出去和输入字符串进行匹配，匹配成功后，继续替换栈中的剩余变量。通过这种方式，我们无需违反栈的性质就能完成替换。

按照这样的思路，我们将从 CFG 转换为 PDA 的步骤总结如下：

1. 将开始变量压入栈顶；
2. 如果栈顶是：
    1. 变量：使用规则替换；
    2. 终结符：弹出，然后与下一个输入字符进行匹配；
3. 当栈为空时，接受。

以 $G_2$ 为例，假设输入字符串为 $a+a\times a$，下面将是栈中字符变化的过程：
$$
\begin{aligned}
E
\end{aligned}
\Rightarrow
\begin{aligned}
E\\
+\\
T
\end{aligned}
\Rightarrow
 \begin{aligned}
F\\
+\\
T\\
\end{aligned}
\Rightarrow
\begin{aligned}
T\\
+\\
T\\
\end{aligned}
\Rightarrow
\begin{aligned}
a\\
+\\
T\\
\end{aligned}
\overset{\text{ pop }}{\Rightarrow}
\begin{aligned}
+\\
T\\
\end{aligned}
\overset{\text{ pop }}{\Rightarrow}
\begin{aligned}
T\\
\end{aligned}
\Rightarrow
\begin{aligned}
T\\
\times\\
F\\
\end{aligned}
\Rightarrow
\begin{aligned}
F\\
\times\\
F\\
\end{aligned}
\Rightarrow
\begin{aligned}
a\\
\times\\
F\\
\end{aligned}
\overset{\text{ pop }}{\Rightarrow}
\begin{aligned}
\times\\
F\\
\end{aligned}
\overset{\text{ pop }}{\Rightarrow}
\begin{aligned}
F\\
\end{aligned}
\Rightarrow
\begin{aligned}
a\\
\end{aligned}
\overset{\text{ pop }}{\Rightarrow}
\begin{aligned}
\varnothing
\end{aligned}
$$

## 总结

<font style="background: #FEEDD5;">📜定理</font>：$A$ 是一个 CFL，当且仅当存在某个 PDA 能识别 $A$。

这条定理比上一条要更强，它说明了 PDA 和 CFG 是完全等价的。我们已经证明了 CFG $\Rightarrow$ PDA，但我们将不会去证明 PDA $\Rightarrow$ CFG，因为这个证明会有点复杂，并不作为课程要求，如果有兴趣可以自行了解。

最后总结一下我们已经学习过的几种模型以及它们之间的关系。

FA 包括 DFA、NFA、GNFA，它们都能识别正则语言，我们称之为正则语言的识别器（recognizers）；而 PDA 能识别 CFL，我们称之为 CFL 的识别器。正则表达式能生成正则语言，我们称之为正则语言的生成器（generators）；而 CFG 能生成 CFL，我们称之为 CFL 的生成器。

正则语言是一种 CFL，CFL 包括了正则语言，因为 PDA 比 FA 要更加强大，肯定能识别正则语言。

尽管我们说 PDA 比 FA 更强，但其实这几类模型都属于弱模型，这里存在一些其他它们无法识别的语言。在下一节，我们将介绍强模型，这将是贯穿我们剩余课程的主题，也是这门课的重点。虽然这些弱模型的表达能力不足，但学习它们有助于我们理解语言，为学习图灵机做好铺垫。

## 课后题

$A.$ **令 $E=\\{a^ib^j|i\neq j\text{ and } 2i\neq j\\}$，证明 $E$ 是 CFL。**

条件 $i \neq j$ 且 $2i \neq j$ 意味着在 $i, j \ge 0$ 的坐标平面上，我们需要避开两条直线：$j = i$ 和 $j = 2i$。

这两条直线将第一象限分成了三个区域：

1. 区域 1： $j < i$ （即 $a$ 的数量比 $b$ 多）
2. 区域 2： $i < j < 2i$ （即 $b$ 的数量在 $a$ 的 1 倍到 2 倍之间，且不等于边界）
3. 区域 3： $j > 2i$ （即 $b$ 的数量比 $a$ 的两倍还要多）

因此，$E = L_1 \cup L_2 \cup L_3$，其中：

- $L_1 = \\{a^i b^j \mid i > j\\}$
- $L_2 = \\{a^i b^j \mid i < j < 2i\\}$
- $L_3 = \\{a^i b^j \mid j > 2i\\}$

我们需要为这三个语言分别构造 CFG。

$1.$ 对于 $L_1 = \\{a^i b^j \mid i > j\\}$

这个语言表示 $a$ 至少比 $b$ 多一个。我们可以先生成相等数量的 $a$ 和 $b$，然后在左侧添加额外的 $a$。

$$
S_1 \to aS_1b \mid aS_1 \mid a
$$
$2.$ 对于 $L_3 = \\{a^i b^j \mid j > 2i\\}$

这个语言表示对于每一个 $a$ 至少对应两个 $b$，且最后还要多出至少一个 $b$。

$$
S_3 \to aS_3bb \mid S_3b \mid b
$$
$3.$ 对于 $L_2 = \\{a^i b^j \mid i < j < 2i\\}$

这是最关键的部分。条件 $i < j < 2i$ 意味着：

- $j$ 必须大于 $i$（至少包含一对 $a \to bb$）
- $j$ 必须小于 $2i$（至少包含一对 $a \to b$）

我们可以通过组合“一个 $a$ 匹配一个 $b$”和“一个 $a$ 匹配两个 $b$”的规则来构造。为了保证严格不等式，文法必须确保这两种匹配规则都至少被用到一次。文法构造如下：
$$
\begin{aligned}
S_2 & \to aS_2b \mid aS_2bb \mid aAbb\\
A &\to aAb \mid aAbb \mid ab
\end{aligned}
$$
由于我们已经为 $L_1, L_2, L_3$ 分别构造了上下文无关文法，因此它们都是 CFL。

令 $S$ 为原语言 $E$ 的开始符号，我们可以通过引入以下产生式来合并它们：

$$
S \to S_1 \mid S_2 \mid S_3
$$
$E$ 对应的完整 CFG：
$$
\begin{aligned}
S &\to S_1 \mid S_2 \mid S_3\\
S_1 &\to aS_1b \mid aS_1 \mid a\\
S_2 & \to aS_2b \mid aS_2bb \mid aAbb\\
A &\to aAb \mid aAbb \mid ab\\
S_3 &\to aS_3bb \mid S_3b \mid b
\end{aligned}
$$
由于存在一个 CFG 能识别 $E$，因此语言 $E$ 是一个上下文无关语言。证毕。

