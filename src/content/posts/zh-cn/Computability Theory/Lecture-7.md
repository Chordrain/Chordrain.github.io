---
title: "[COMPT 0x07] DFAs和CFGs的判定问题"
date: 2026-01-03
draft: false
math: true
categories:
  - 可计算性理论
tags: []
---

这节课，我们将使用图灵机来解决真实的问题。

## DFAs的接受问题

令 $A_{DFA} = \\{ \langle B,w \rangle \mid B \text{ is a DFA and } B \text{ accepts } w \\}$，证明 $A_{DFA}$ 是图灵可判定的。

证明：构造一个图灵判定器 $D_{A-DFA}$ 来判定 $A_{DFA}$，下面是对 $D_{A-DFA}$ 的描述：

$$
\begin{aligned}
D_{A-DFA} = &\text{“On input }s\\
&\text{1. Check that }s\text{ has the form }\langle B,w \rangle \text{ where }B\text{ is a DFA and }w\text{ is a string; reject if not.}\\
&\text{2. Simulate the computation of }B \text{ on } w.\\
&\text{3. If } B \text{ ends in an accept state then accept. If not then reject.”}\\
\end{aligned}
$$
相信大家都能看懂 $D_{A-DFA}$ 是如何判定 $A_{DFA}$ 的。由于 DFA 肯定能在有限时间内给出结果，因此 $D_{A-DFA}$ 总是能停机。

考虑到大家可能会疑惑 $D_{A-DFA}$ 长什么样，为了让大家更加理解，我们给出下面这张图来可视化这个 $D_{A-DFA}$。

<center><img src="/images/Software Theory Fundamentals/07-01.png" alt="07-01" /></center>

可以看到，$B$ 是以一个 DFA 的标准形式化定义给出的，$\langle B,w \rangle$ 就是把 $B$ 的形式化定义和 $w$ 拼接到一起组成了一个字符串。$D_{A-DFA}$ 运行时就根据这个字符串模拟出这样一个 $B$。此外，我们知道为了模拟一个 DFA 的运行，还需要记录当前状态和头指针在 $w$ 上的位置，于是我们再加一个纸带记录这两个信息，这无伤大雅，毕竟我们已经知道多带图灵机和单带图灵机是等价的。

 ## NFAs的接受问题

令 $A_{NFA} = \\{ \langle B,w \rangle \mid B \text{ is a NFA and } B \text{ accepts } w \\}$，证明 $A_{NFA}$ 是图灵可判定的。

证明思路和 DFAs 的接受问题是一模一样的，但有一点细微的差别。注意 NFA 中存在着 $\epsilon$-转移，这可能会使得 NFA 陷入无限循环，导致构造出的图灵机 $D_{A-NFA}$ 无法停机。解决方法其实是很简单的，将 NFA 转换为 DFA 就行了。

构造一个图灵判定器 $D_{A-NFA}$ 来判定 $A_{NFA}$，下面是对 $D_{A-NFA}$ 的描述：
$$
\begin{aligned}
D_{A-NFA} = &\text{“On input }s\\
&\text{1. Convert NFA }B\text{ to equivalent DFA }B^{\prime}.\\
&\text{2. Check that }s\text{ has the form }\langle B,w \rangle \text{ where }B\text{ is a NFA and }w\text{ is a string; reject if not.}\\
&\text{3. Simulate the computation of }B^{\prime} \text{ on } w.\\
&\text{4. If } B^{\prime} \text{ ends in an accept state then accept. If not then reject.”}\\
\end{aligned}
$$
实际上还能再简单一点，我们可以看到 2、3、4 步其实和之前的 $D_{A-DFA}$ 是一样的，因此我们可以这么做：
$$
\begin{aligned}
D_{A-NFA} = &\text{“On input }s\\
&\text{1. Convert NFA }B\text{ to equivalent DFA }B^{\prime}.\\
&\text{2. Run TM }D_{A-DFA}\text{ on input }\langle B^{\prime},w \rangle.\\
&\text{3. Accept if }D_{A-DFA} \text{ accepts.}\\
&\quad\text{Reject if not.”}
\end{aligned}
$$
将另一个图灵机作为子程序在图灵机中进行调用是完全合法的，要习惯这种用法，这将对我们解题提供极大便利。

## DFAs的空集问题

令 $E_{DFA}=\\{ \langle B \rangle \mid B \text{ is a DFA and } L(B) = \varnothing \\}$，证明 $E_{DFA}$ 是图灵可判定的。

这个问题其实就是在问我们如何判定一个 DFA 所能识别的语言是空语言。DFA 中是没有 $\epsilon$-转移的，所以，只要 DFA 中存在着从初始状态到接受状态的路径，那 $L(B)$ 就肯定不为空；反之，如果不存在这样一条路径，$L(B)$ 就为空。要检查是否存在这样的路径，就要用到路径搜索算法了。路径搜索算法可太多了，我们这里选择 BFS 作为示例。下面是我们构造的图灵判定器 $D_{E-DFA}$：
$$
\begin{aligned}
D_{E-DFA} = &\text{“On input }\langle B \rangle \\
&\text{1. Mark start state.}\\
&\text{2. Repeat until no new state is marked:}\\
&\quad\text{Mark every state that has an incoming arrow from a previously marked state.}\\
&\text{3. Accept if no accept state is marked.}\\
&\quad\text{Reject if some accept state is marked.”}
\end{aligned}
$$
考虑这样一个问题：我们是否可以选择将所有可能的字符串都输入 $B$，来检测 $L(B)$ 是否为空？答案是不行。因为我们这里是要证明 $E_{DFA}$ 是图灵可判定的，而这种方法会使得 $D_{E-DFA}$ 可能无法停机，那就无法证明 $E_{DFA}$ 是图灵可判定的了。

## DFAs的等价问题

令 $EQ_{DFA} = \\{ \langle A,B \rangle \mid A \text{ and } B \text{ are DFAs and } L(A) = L(B) \\}$，证明 $EQ_{DFA}$ 是图灵可判定的。

这是一个有趣的问题。解决这个问题可以用到我们之前所说的调用子程序图灵机的思想，至于是哪个图灵机，这里给个提示：就是上面的 $D_{E-DFA}$。希望大家先自己想一想。

我们的思路是这样的：如果 $L(A) = L(B)$，那么 $L(A)$ 和 $L(B)$ 的对称差（两个集合不相交的部分）应该为 $\varnothing$，所以我们去构造一个 DFA $C$，并且 $L(C)=(L(A) \cap \overline{L(B)}) \cup (\overline{L(A)} \cap L(B))$。那下一步就只需要调用 $D_{E-DFA}$ 来检测 $C$ 是否识别的是空语言就好了。

下面给出我们构造的图灵判定器 $D_{EQ-DFA}$：
$$
\begin{aligned}
D_{EQ-DFA} = &\text{“On input }\langle A,B \rangle \\
&\text{1. Construct DFA }C\text{ where }L(C)=(L(A) \cap \overline{L(B)}) \cup (\overline{L(A)} \cap L(B)).\\
&\text{2. Run }D_{E-DFA}\text{ on }\langle C \rangle. \\
&\text{3. Accept if }D_{E-DFA} \text{ accepts.}\\
&\quad \text{Reject if }D_{E-DFA} \text{ rejects.”}
\end{aligned}
$$
<font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">🏹练习</font>：令 $EQ_{REG} = \\{ \langle A,B \rangle \mid A \text{ and } B \text{ are REs and } L(A) = L(B) \\}$，能否证明 $EQ_{REG}$ 是图灵可判定的？这道题的答案显而易见，我们已经知道了正则表达式和 DFA 是等价的，既然我们能解决 DFAs 的等价问题，那么正则表达式的等价问题也能迎刃而解。

## CFGs的接受问题

令 $A_{CFG} = \\{ \langle G,w \rangle \mid G \text{ is a CFG and } w \in L(G) \\}$，证明 $A_{CFG}$ 是图灵可判定的。

事实上，要证明这件事还真不是件容易的事情。一个 CFG 中可能存在无穷无尽的推导路径，如果要去枚举每一种可能，那么我们构造出的图灵机就不可能停机。

要解决这个问题，我们需要认识一个概念——乔姆斯基范式（Chomsky Normal Form, CNF）。CNF 只允许出现下面这两种形式的规则：
$$
\begin{aligned}
A & \to BC\\
B & \to b
\end{aligned}
$$
CNF 保证了：

1. <font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜引理 1</font>：任何一个 CFG 都可以转换成 CNF。
2. <font style="border-radius: .2rem; background-color: #FFEBDD; color: #FF970A;">📜引理 2</font>：如果 $H$ 是一个 CNF 并且 $w \in L(H)$，那么每个对 $w$ 的推导的长度都是 $2 | w | - 1$ 步。

让我们来关注引理 2。我们不需要知道为什么会有 $2 | w | - 1$，我们只需要知道它给出了对推导路径长度的限制，这使得遍历所有的推导成为可能。CNF 中，每一步推导只有有限的可选规则，而总推导步数已经被限制成固定值 $2|w|-1$，这就变成了一个遍历“有限深度、有限分支的树”问题，因此推导路径是可枚举的。

下面给出构造的图灵判定器 $D_{A-CFG}$：
$$
\begin{aligned}
D_{A-CFG} = &\text{“On input }\langle G,w \rangle \\
&\text{1. Convert }G\text{ into CNF. }\\
&\text{2. Try all derivations of length } 2|w|-1 . \\
&\text{3. Accept if any generate }w.\\
&\quad \text{Reject if not.”}
\end{aligned}
$$
这里我们还能给出一个推论：每一个 CFL 都是图灵可判定语言。

这一点其实是很直观的。我们说 CFL 的定义就指出了：假设 $A$ 是一个 CFL，那么一定存在一个 CFG $G$ 能生成 $A$。使用这个 $G$，我们就能构造的图灵判定器 $M_{G}$ 来判定一个字符串是否属于 CFL $A$：

$$
\begin{aligned}
M_{G} = &\text{“On input } w \\
&\text{1. Run } D_{A-CFG} \text{ on } \langle G, w \rangle.\\
&\text{2. Accept if } D_{A-CFG} \text{ accepts.}\\
&\quad \text{Reject if it rejects.”}
\end{aligned}
$$
这里你可能会疑问，我们如何找到这个 $G$？答案是找不到，我们也不需要关注找不找的到，我们只需要关注其存在性即可。

出于同样的原因，我们可以得到结论：$A_{PDA}$ 肯定也是图灵可判定的，因为 PDA 和 CFG 是等价的。

## CFGs的空集问题

令 $E_{CFG}=\\{ \langle G \rangle \mid G \text{ is a CFG and } L(G) = \varnothing \\}$，证明 $E_{CFG}$ 是图灵可判定的。

要去检查一个 CFG 是否生成空语言，其实就看它最终能否把开始变量替换为终结符。要验证这一点，我们可以从终结符开始倒推：首先标记 CFG 中全部的终结符，假设非终结符 $A$ 能产生终结符 $a$，那么就将 $A$ 进行标记，被标记的非终结符也可以视作终结符，然后继续向前标记。如果最终 CFG 中的开始变量未被标记，那就说明 CFG 生成的是空语言，否则不是。

下面给出构造的图灵判定器 $D_{E-CFG}$：
$$
\begin{aligned}
D_{E-CFG} = &\text{“On input }\langle G \rangle \\
&\text{1. Mark all occurrences of terminals in }G.\\
&\text{2. Repeat until no new variables are marked:}\\
&\quad\text{Mark all occurrences of variables }A\text{ if}\\
&\quad A \to B_{1} B_{2} \dots B_{k} \text{ is a rule and all } B_{i} \text{ were already marked.}\\
&\text{3. Reject if the start variable is marked.}\\
&\quad\text{Accept if not.”}
\end{aligned}
$$

## CFGs的等价问题

令 $EQ_{CFG} = \\{ \langle G,H \rangle \mid G \text{ and } H \text{ are CFGs and } L(G) = L(H) \\}$，能否证明 $EQ_{CFG}$ 是图灵可判定的？

思路有两种，一种是找到 $G$ 和 $H$ 对应的 PDA，然后分别把对方的字符串输入进对方的 PDA，看是否能完全接收，但这样的话图灵机就不可停机了，因此这条路不行；另一种思路是效仿 DFAs 的等价问题的思路，我们去构造对称差，但可惜的是，CFL 在交集和补集的运算下是不封闭的，这导致我们无法用同样的思路去证明。那怎么办？

直接给出结论：$EQ_{CFG}$ 是图灵不可判定的。具体的证明我们将在下节课给出。

不仅如此，判断一个 CFG 是否具有二义性也是不可判定的。可能大家原本都以为 CFG 是一个很简单的东西，但没想到在这些问题上它居然这么棘手。

## TMs的接受问题

令 $A_{TM} = \\{ \langle M,w \rangle \mid M \text{ is a TM and } w \in L(M) \\}$，能否证明 $A_{TM}$ 是图灵可判定的？

$A_{TM}$ 也是图灵不可判定的，关于这一点的证明将在下节课给出。但是值得注意的是，虽然它是图灵不可判定，但它却是图灵可识别的，下面让我们来尝试证明这一点。

下面的图灵识别器 $U$ 能识别 $A_{TM}$：

$$
\begin{aligned}
U = &\text{“On input }\langle M,w \rangle \\
&\text{1. Simulate }M\text{ on input }w.\\
&\text{2. Accept if }M\text{ halts and accepts}.\\
&\text{3. Reject if }M\text{ halts and rejects}.\\
&\text{4. Reject if }M\text{ never halts.”}\\
\end{aligned}
$$
这就是用一台图灵机模拟另一台图灵机的例子。但是注意，Michael Sipser 教授指出我们不应该在图灵机的算法里写“如果图灵机没有停机，那也拒绝”这种话（$U$ 中的第 4 步），因为图灵机的停机问题是不可判定的，我们没办法知道图灵机是还未跑出结果，还是在不断循环。正确的写法是“如果 $M$ 卡死/循环，$U$ 也永远循环”。

## 课后题

$A.$ **假设 $C$ 是一种语言。证明 $C$ 是图灵可识别的，当且仅当存在一个图灵可判定的语言 $D$ 使得 $C = \\{ x \mid \exists y, \langle x,y \rangle \in D \\}$。**

先证明：存在一个图灵可判定的语言 $D$ 使得 $C = \\{ x \mid \exists y, \langle x,y \rangle \in D \\}$ $\Rightarrow$ $C$ 是图灵可识别的。要证明这一点，我们就要构造一个图灵机 $R_{C}$ 来识别 $C$。假设 $D_{D}$ 是 $D$ 的判定器。注意这里的 $y$ 是可以枚举的，这将是我们的切入点：
$$
\begin{aligned}
R_{C} = &\text{“On input } w \\
&\text{1. For all }y \in \Sigma^{* }\text{, simulate }D_{D}\text{ on input }\langle w,y \rangle, \\
&\quad\text{until } D_{D}\text{ halts and accepts some }y.\\
&\text{2. Accept if }D_{D}\text{ halts and accepts.”}\\
\end{aligned}
$$
下面证明：$C$ 是图灵可识别的 $\Rightarrow$ 存在一个图灵可判定的语言 $D$ 使得 $C = \\{ x \mid \exists y, \langle x,y \rangle \in D \\}$。我们要构造一个图灵判定器来识别 $D$，也就是要构造一个图灵可判定的语言 $D$ 能和 $C$ 满足这样的关系，这里面的难点在于如何让图灵机停机。其实可以直接把 $y$ 设定为运行时间，说如果 $y$ 步内运行出来了就接受，否则就拒绝，那就保证了一定能跑出结果，因此，假设 $R_{C}$ 是识别 $C$ 的图灵识别器，我们可以这样构造 $D$ 的判定器 $D_{D}$：
$$
\begin{aligned}
D_{D} = &\text{“On input } \langle x,y \rangle \\
&\text{1. Simulate }R_{C}\text{ on input } x \text{ within }y\text{ steps},\\
&\text{2. Accept if }R_{C}\text{ halts and accepts.}\\
&\text{3. Reject if }R_{C}\text{ halts and rejects.}\\
&\text{4. Reject if }R_{C}\text{ does not halt within }y\text{ steps.”}\\
\end{aligned}
$$

---

$B.$ **令 $E = \\{ \langle M \rangle \mid M \text{ is a DFA that accepts some string with more 1s than 0s} \\}$，证明 $E$ 是图灵可判定的。（提示：CFL）**

使用 CFG 可以构造一个文法来生成 1 比 0 多的字符串，并令对应的语言为 $A$。若 $M$ 能识别 1 比 0 多的字符串，那肯定有 $A \cap L(M) \neq \varnothing$，另外，正则语言和 CFL 的交集仍然是 CFL（见 Lecture 5），也就是说，问题转变为了判断 CFL 是否为空，我们可以用到之前的构造 $D_{E-CFG}$。我们可以构造图灵判定器 $D_{E}$ 来识别语言 $E$：
$$
\begin{aligned}
D_{E} = &\text{“On input } \langle M \rangle \\
&\text{1. Simulate }D_{E-CFG}\text{ on input } \langle G \rangle \text{ where } L(G) = A \cap L(M).\\
&\text{2. Accept if }D_{E-CFG}\text{ halts and accepts.}\\
&\text{3. Reject if }D_{E-CFG}\text{ halts and rejects.”}
\end{aligned}
$$
