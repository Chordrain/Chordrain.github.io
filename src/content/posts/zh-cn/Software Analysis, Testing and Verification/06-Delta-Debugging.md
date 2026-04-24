---
title: "[SATV 0x06] Delta Debugging"
date: 2025-10-10
draft: false
math: true
categories:
  - 软件分析测试与验证
tags:
  - deltaDebugging
---

Once we have reproduced a program failure, we must find out what’s relevant:

* Does failure really depend on 10,000 lines of code?
* Does failure really require this exact schedule of events?
* Does failure really need this sequence of function calls?

Answering these questions helps simplify a test case. But why simplify?

* Ease of communication: a simplified test case is easier to communicate
* Easier debugging: smaller test cases result in smaller states and shorter executions
* Identify duplicates: simplified test cases subsume several duplicates

Let's look at a concrete bug report in the Mozilla bug database. Consider this HTML page.

```html
<td align=left valign=top>
<SELECT NAME="op sys" MULTIPLE SIZE=7>
<OPTION VALUE="All">All<OPTION VALUE="Windows 3.1">Windows 3.1<OPTION VALUE="Windows 95">Windows 95<OPTION VALUE="Windows 98">Windows 98<OPTION 
VALUE="Windows ME">Windows ME<OPTION VALUE="Windows 2000">Windows 2000<OPTION VALUE="Windows NT">Windows NT<OPTION VALUE="MacSystem 7">Mac 
System 7<OPTION VALUE="Mac System 7.5">Mac System 7.5<OPTION VALUE="Mac System 7.6.1">Mac System 7.6.1<OPTION VALUE="Mac System 8.0">Mac System 
8.0<OPTION VALUE="Mac System 8.5">Mac System 8.5<OPTION VALUE="Mac System 8.6">Mac System 8.6<OPTION VALUE="Mac System 9.x">Mac System 
9.x<OPTION VALUE="MacOS X">MacOS X<OPTION VALUE="Linux">Linux<OPTION VALUE="BSDI">BSDI<OPTION VALUE="FreeBSD">FreeBSD<OPTION 
VALUE="NetBSD">NetBSD<OPTION VALUE="OpenBSD">OpenBSD<OPTION VALUE="AIX">AIX<OPTION VALUE="BeOS">BeOS<OPTION VALUE="HP-UX">HP-UX<OPTION 
VALUE="IRIX">IRIX<OPTION VALUE="Neutrino">Neutrino<OPTION VALUE="OpenVMS">OpenVMS<OPTION VALUE="OS/2">OS/2<OPTION VALUE="OSF/1">OSF/1<OPTION 
VALUE="Solaris">Solaris<OPTION VALUE="SunOS">SunOS<OPTION VALUE="other">other</SELECT>
</td>
<td align=left valign=top>
<SELECT NAME="priority" MULTIPLE SIZE=7>
<OPTION VALUE="--">--<OPTION VALUE="P1">P1<OPTION VALUE="P2">P2<OPTION VALUE="P3">P3<OPTION VALUE="P4">P4<OPTION VALUE="P5">P5</SELECT>
</td>
<td align=left valign=top>
<SELECT NAME="bug severity" MULTIPLE SIZE=7>
<OPTION VALUE="blocker">blocker<OPTION VALUE="critical">critical<OPTION VALUE="major">major<OPTION VALUE="normal">normal<OPTION 
VALUE="minor">minor<OPTION VALUE="trivial">trivial<OPTION VALUE="enhancement">enhancement</SELECT>
</tr>
</table>
```

Loading this page using a certain version of Mozilla's web browser and printing it causes a segmentation fault. Somewhere in this HTML input is something that makes the browser fail. But how do we find it?

How do we go from that large input to this simple input?

```html
<SELECT>
```

## 01 Binary Search

What do we as humans do in order to minimize test cases?

One possibility is that we might use a binary search, cutting the test case in two and testing each half of the input separately. We could even iterate this procedure to shrink the input as much as possible. Even better, binary search is a process that can be easily automated for large test cases!

Let's see how this application of binary search might work.

Proceeding by binary search, we throw away half the input and see if the output is still wrong. If it is, repeat this operation; if not, go back to the previous state and discard the other half of the input.

Now let's see an example. Assuming that the original input that triggers the bug is:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-01.png)

After testing, the output is reported to be wrong, so we discard half of the original input and test again:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-02.png)

The testing still reports the same error, so we repeat the process:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-03.png)

It is evident from the testing result that the output is correct, indicating that this particular input is not the root cause of the bug. We therefore revert to the previous state and attempt the alternative half of the input.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-04.png)

The testing result indicates that this half is the root cause. However, we are yet to ascertain whether this is the minimal bug-triggered input, so it is necessary to continue splitting and testing it.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-05.png)

As demonstrated in the figure above, both halves successfully pass the test, indicating that the previous input has been sufficiently minimized. In other words, once streamlined, the input can no longer result in the error. This is the simplified input we are looking for.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-06.png)

Using binary search, we can identify the following code that is triggered by a bug:

```html
<SELECT NAME="priority" MULTIPLE SIZE=7>
```

The next step is to identify the specific part of the code that triggers the bug. In this regard, we can still use binary search, but with finer granularity. The following graph illustrates the process and provides the result.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-07.png)

Finally, we identify the source of the issue:

```html
<SELECT>
```

## 02 Delta Debugging

Let's generalize the binary search procedure a bit. Instead of requiring ourselves to divide inputs strictly in half at each iteration, we could allow ourselves more granularity in dividing our input.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Delta-Debugging-08.png)

The process of dividing the input will result in an increase in the granularity. The impact of input granularity:

|            Input granularity             | Finer  | Coarser |
| :--------------------------------------: | :----: | :-----: |
| Chance of finding a failing input subset | Higher |  Lower  |
|          Progress of the search          | Slower | Faster  |

 Now we give the formal description of the delta-debugging algorithm.

Inputs and Failures:

* Let $R$ be the set of possible inputs
* $r_{P}\in R$ corresponds to an input that passes
* $r_{F}\in R$ corresponds to an input that fails

Changes:

* We can go from one input $r_{1}$ to another input $r_{2}$ by a series of changes. A change $\delta$ is a mapping $R\rightarrow R$ which takes one  input and changes it to another input.
* A change $\delta$ can be decomposed to a number of **elementary changes** $\delta_{1},\delta_{2},\dots,\delta_{n}$ where $\delta=\delta_{1}\circ\delta_{2}\circ\dots\circ\delta_{n}$ and $(\delta_{i}\circ\delta_{j})(r)=\delta_{j}(\delta_{i}(r))$.
* We have a set of changes $c_{F}=\{\delta_{1},\delta_{2},\dots,\delta_{n}\}$ such that $r_{F}=(\delta_{1}\circ\delta_{2}\circ\dots\circ\delta_{n})(r_{P})$. Each subset of $c$ of $c_{F}$ is a test case.

Testing test cases:

* Given a test case $c$, we would like to know if the input generated by applying changes in $c$ to $r_{P}$ causes the same failure as $r_{F}$.
* We define the function $\mathrm{test}: \mathrm{Powerset}(c_{F})\rightarrow\{P,F\}$ such that, given $c=\{\delta_{1},\delta_{2},\dots,\delta_{n}\}\subseteq c_{F}$, $\mathrm{test}(c)=F$ iff $(\delta_{1}\circ\delta_{2}\circ\dots\circ\delta_{n})(r_{P})$ is a failing input.

Minimizing test case:

* Goal: find the smallest test case $c$ such that $\mathrm{test}(c)=F$
* A failing test case $c\subseteq c_{F}$ is called the global minimum of $c_{F}$ if $\forall c^{\prime}\subseteq c_{F},\lvert c^{\prime}\rvert<\lvert c\rvert\Rightarrow\mathrm{test}(c^{\prime})\neq F$.
* The global minimum is the smallest set of changes which will make the program fail. Finding the global minimum may require performing an exponential number of tests.

##  03 Search for 1-minimal Input

Find a set of changes that cause the failure, but removing any single change causes the failure to go away. This is **1-minimality**.

* A failing test case $c\subseteq c_{F}$ is called a **local minimum** of $c_{F}$ if $\forall c^{\prime}\subset c,\mathrm{test}(c^{\prime})\neq F$.

* A failing test case $c\subseteq c_{F}$ is **n-minimal** if $\forall c^{\prime}\subset c,\lvert c\rvert-\lvert c^{\prime}\rvert\le n\Rightarrow\mathrm{test}(c^{\prime})\neq F$.

* A failing test case is **1-minimal** if $\forall\delta_{i}\in c,\mathrm{test}(c-\{\delta_{i}\})\neq F$.

Here we present a naive algorithm for finding a 1-minimal subset of $c$: if $\forall\delta_{i}\in c,\mathrm{test}(c-\{\delta_{i}\})\neq F$, then $c$ is 1-minimal else recurse on $c-\{\delta\}$ for some $\delta\in c,\mathrm{test}(c-\{\delta\})=F$. This algorithm can be described in pseudocode as follows:

```pseudocode
function find_1_minimal(c):
    for δ in c:
        c_prime = c - {δ}  // 创建一个移除了 δ 的新集合
        if test(c_prime) == F:
            return find_1_minimal(c_prime)  // 递归调用，继续简化
    return c  // 如果无法移除任何元素，返回当前集合，它是 1-minimal
```

Asymptotic analysis: In the worst case, we have to remove one element from the set per iteration, which is potentially $N+(N-1)+(N-2)+\dots$. This is $\Omicron(N^2)$.

Could we improve this process? The answer is yes. It is silly to remove one element at a time. We can attempt to divide the change set into two initially, and increase the number of subsets if we are unable to make progress. Should we get lucky, the search will converge quickly. This is the **minimization algorithm**.

The minimization algorithm finds a 1-minimal test case:

* It partitions the set $c_{F}$ to $\Delta_{1},\Delta_{2},\dots,\Delta_{n}$ - $\Delta_{1},\Delta_{2},\dots,\Delta_{n}$ are pairwise disjoint, and $c_{F}=\Delta_{1}\cup\Delta_{2}\cup\dots\cup\Delta_{n}$. Specially, when $n=2$, $\Delta_{1}=\nabla_{2}$ and $\Delta_{2}=\nabla_{1}$.
* Define the complement of $\Delta_{1}$ as $\nabla_{i}=c_{F}-\Delta_i$.
* It starts with $n=2$.
* It tests each test case defined by each partition and its complement, and reduces the test case if a smaller failure inducing set is found, otherwise it refines the partition (i.e. $n:=n*2$).

Steps of the minimization algorithm:

1. Start with $n=2$ and $\Delta$ as the test set
2. Test each $\Delta_{1},\Delta_{2},\dots,\Delta_{n}$ and each $\nabla_{1},\nabla_{2},\dots,\nabla_{n}$
3. There are three possible outcomes:
   1. Some $\Delta_{i}$ causes failure: Go to step 1 with $\Delta=\Delta_{i}$ and $n=2$.
   2. Some $\nabla_{i}$ causes failure: Go to step 1 with $\Delta=\nabla_{i}$ and $n=n-1$[^1].
   3. No test causes failure: If granularity can be refined, go to step 1 with $\Delta=\Delta$ and $n=n*2$; otherwise, done, found the 1-minimal subset.

Asymptotic analysis: Worst case is still quadratic, when it subdivides until each set is of size 1 (reduce to the naive algorithm). But luckily, for a single failure, it converges in $\Omicron(\log N)$ (binary search).

In practice, it may be necessary to customize and adapt the delta debugging algorithm to each significant system in order to exploit system knowledge; that is, it is not a plug-and-play algorithm.

---

Quiz: A program crashes when its input contains 42. Fill in the data in each iteration of the minimization algorithm assuming character granularity.

| Iteration |  n   | $\Delta$ | $\Delta_{1},\Delta_{2},\dots,\Delta_{n},\nabla_{1},\nabla_{2},\dots,\nabla_{n}$ |
| :-------: | :--: | :------: | :----------------------------------------------------------: |
|     1     |  2   |   2424   |                              24                              |
|     2     |  4   |   2424   |                     2,4,242,224,424,244                      |
|     3     |  3   |   242    |                         2,4,24,42,22                         |
|     4     |  2   |    42    |                             4,2                              |

[^1]:This is to keep the granularity (search length) consistent, for $\lvert\nabla_{i}\rvert+1=\lvert\nabla_{i}\rvert+\lvert\Delta_{i}\rvert=n$.
