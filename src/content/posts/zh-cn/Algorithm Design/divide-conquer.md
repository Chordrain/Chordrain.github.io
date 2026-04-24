---
title: "[ALDE 0x04] 分治法"
date: 2023-12-23
draft: true
math: true
categories:
  - 算法设计与分析
tags:
  - 分治
note: 本系列笔记是为算法设计与分析课程的复习而写的，涉及知识点并不完备，也仅为应付考试。
---

分治算法没有太多好讲的，多做几道经典例题就能应付考试了。这里拿三道作业题（数据库查询、逆序对、银行卡账户）练练手就够用了。

<!--more-->

## 01 数据库查询

> 存在两个数据库，每个数据库中有 $n$ 个整数，且没有重复的数；数据库可供查询，输入一个 $k$，它会返回该数据库中第 $k$ 小的数；要求使用尽可能少的查询次数，找出这 $2n$ 个整数中第 $n$ 小的数。

假设我们能直接看到两个数据库中的序列，且都已排好序，分别令其为 $A$ 和 $B$，我们定义函数 $\mathrm{Find}(A,B,k)$，该函数返回寻找序列 $A$ 和序列 $B$ 中第 $k$ 小的数。

初始时 $k=n$，函数内部分别查询 $A$ 和 $B$ 的第 $\lfloor k/2 \rfloor$ 小的数，有：
$$
a = A [\lfloor k/2 \rfloor],b = B [\lfloor k/2 \rfloor]
$$
**情况一**：$a \lt b$

由于 $a$ 小于 $b$，可知 $A$ 中第 $\lfloor k/2 \rfloor$ 小的数比 $B$ 中第 $\lfloor k/2 \rfloor$ 小的数还要小，因此 $A[1,\dots,\lfloor k/2 \rfloor]$ 都不是第 $k$ 小的数，可以直接丢弃，因此递归调用 $\mathrm{Find}(A[\lfloor k/2 \rfloor + 1,\dots,n],B,k - \lfloor k/2 \rfloor)$。

**情况二**：$a \gt b$

由于 $b$ 小于 $a$，可知 $B$ 中第 $\lfloor k/2 \rfloor$ 小的数比 $A$ 中第 $\lfloor k/2 \rfloor$ 小的数还要小，因此 $B[1,\dots,\lfloor k/2 \rfloor]$ 都不是第 $k$ 小的数，可以直接丢弃，因此递归调用 $\mathrm{Find}(A,B[\lfloor k/2 \rfloor + 1,\dots,n],k - \lfloor k/2 \rfloor)$。

**情况三**：$a=b$

题设已经说不存在重复数，因此不需要考虑。但如果没有这个条件，遇到这种情况其实就是在说第 $k$ 小的数就是 $a$，可以直接返回。

**递归出口**：$k = 1$

此时，直接返回 $\min(A[1],B[1])$。

以上，我们就成功使用分治法完成了在知道两个有序序列的情况下，查找第 $k$ 小的数。但现在我们看不到数据库中的数，所以我们需要把 $\mathrm{Find}$ 的前两个参数从序列改成查询的起始位置的偏移量，即 $\mathrm{Find}(i,j,k)$，其返回在 $A$ 中从第 $i$ 小的数开始，和 $B$ 中第 $j$ 小的数开始的数中，第 $k$ 小的数。整题的代码如下所示。

```python
# A.find(i) 返回数据库A中第i小的数
# B.find(i) 返回数据库B中第i小的数
def Find(i, j, k):
    if k == 1:
        return min(A.find(i), B.find(j))
    t = k // 2
    a = A.find(t + i)
    b = B.find(t + j)
    if a < b:
        return Find(t + i + 1, j, k - t)
    else:
        return Find(i, t + j + 1, k - t)
```

## 02 逆序对

> 给定含有 $n$ 个数的数组，求数组中逆序对的个数。

数逆序对的过程可以在归并排序中完成。在归并过程中，左右两个子数组已经排好序，可以直接统计逆序对的个数，下面是一个例子：
$$
\begin{aligned}
\text{Left: }& \overset{\downarrow}{2} \  4 \  7 \  8 \\
\text{Right: }& \overset{\downarrow}{3} \ 5 \\
\end{aligned}
$$
一开始，两个指针分别指向 2 和 3。比较之后发现，$2 \lt 3$，因此 $(2,3)$ 并不构成一个逆序对。右移左指针：
$$
\begin{aligned}
\text{Left: }& 2 \  \overset{\downarrow}{4} \  7 \ 8 \\
\text{Right: }& \overset{\downarrow}{3} \ 5 \\
\end{aligned}
$$
此时，两个指针分别指向 4 和 3。比较之后发现，$4 \gt 3$，因此 $(4,3)$ 构成一个逆序对，并且由于已经排好序，因此，4 之后的数字肯定也都大于 3，都能与 3 构成逆序对，因此，这里我们已经找到了 3 个逆序对，而且左边的序列里已经不存在能和 3 构成逆序对的数了，因此右指针右移：
$$
\begin{aligned}
\text{Left: }& 2 \  \overset{\downarrow}{4} \  7 \ 8 \\
\text{Right: }& 3 \  \overset{\downarrow}{5}
\end{aligned}
$$
此时，两个指针分别指向 4 和 5。比较之后发现，$4 \lt 5$，因此 $(4,5)$ 不构成一个逆序对，左指针右移：
$$
\begin{aligned}
\text{Left: }& 2 \ 4 \ \overset{\downarrow}{7} \ 8 \\
\text{Right: }& 3 \  \overset{\downarrow}{5}
\end{aligned}
$$
此时，两个指针分别指向 7 和 5。比较之后发现，$7 \gt 5$，因此 $(7,5)$ 构成一个逆序对，并且由于已经排好序，因此，7 之后的数字肯定也都大于 5，都能与 5 构成逆序对，因此，这里我们又找到了 2 个逆序对，而且左边的序列里已经不存在能和 5 构成逆序对的数了，同时右指针已经到达边界，无法继续右移，所以此次查找结束，共找到 $3+2=5$ 个逆序对。

整体的代码如下：

```python
def Merge(left, right):
    i, j = 0, 0
    inversions = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            inversions += len(left) - i
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions

def MergeSort(arr):
    if len(arr) <= 1:
        return arr, 0
	mid = len(arr) // 2
	left, left_inversions  = MergeSort(arr[:mid])
	right, right_inversions = MergeSort(arr[mid:])
    merged, mid_inversions = Merge(left, right)
	return merged, left_inversions + right_inversions + mid_inversions
```

现在把题目改一下：

> 定义重要逆序对的概念为 $(a_{i},a_{j})$，其中 $i \lt j$ 且 $a_{i} \gt a_{j} \times 2$。求一个序列中重要逆序对的数量。

其实思路跟上面一道题是一样的，但是要做一点小改动。在数重要逆序对时，右指针应该在检测到 $a_{i} \gt a_{j} \times 2$ 时才右移；但在归并时，右指针是在 $a_i \gt a_{j}$ 时就该右移了。所以在改动后的版本中，我们不能在归并的同时进行计数，而是要把这一步单独拿出来：

```python
def Merge(left, right):
    i, j = 0, 0
    inversions = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] > right[j] * 2:
            inversions += len(left) - i
            j += 1
        else:
            i += 1
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions

def MergeSort(arr):
    ...
```

## 03 银行卡账户

> 给定 $n$ 张银行卡，每张银行卡都属于一个账户，但不知道具体的账户是什么。现有一台测试器（equivalence tester）可以检测两张银行卡是否属于同一个账户。给出一个 $O(n\log n)$ 算法判断这堆银行卡中是否存在超过 $n/2$ 张银行卡属于一个账户，要求使用测试器的次数尽可能少。

使用分治法解决这道题，我们先定义函数 $\mathrm{FindMajority}(\text{cards})$，该函数返回占多数的那张卡以及它的账户出现的次数。

在函数内部，我们对这堆银行卡进行均分，分为左右两部分，直到不可再分，即 $\mathrm{len}(\text{cards})=1$ 为止，此时肯定满足条件，返回这唯一一张卡和它出现的次数；回到上一层，此时要对左边和右边的结果进行合并，如何做？由于函数会返回在这堆银行卡中占多数的卡，因此，我们只需要用测试器测试一下这两张卡是否属于同一张，如果是就可以继续返回这张卡，以及它在左右两边出现的次数的和；否则，我们需要把左边返回的那张卡（如果存在的话），与右边堆里所有的卡进行测试，看看是否在两堆中占多数，如果是，返回它；否则，用同样的方法，再看右边返回的那张卡（如果存在的话）是不是占多数，如果是，返回它；否则说明这里不存在占多数的卡，返回 $\text{None}$。

你可能会问，有没有可能合并后，占多数的账户并不一定就是左右两边返回的那张卡呢？没有可能！因为左右两边的卡基本上是均分的，且总数固定，在已经有卡占据其中一堆的大多数时，不可能存在后来居上者。

于是，我们可以写出下面的代码：

```python
# assumption: len(cards) > 0
def FindMajority(cards):
    if len(cards) == 1:
        return cards[0], 1
    t = len(cards) // 2
    left, left_count   = FindMajority(cards[:t])
    right, right_count = FindMajority(cards[t:])
    if left is not None and right is not None and equivalence_tester(left, right):
        return left, left_count + right_count
    if left is not None:
        count = 0
        for c in cards[t:]:
            if equivalence_tester(c, left):
                count += 1
	left_count += count
    if left_count > t:
        return left, left_count
    if right is not None:
        count = 0
        for c in cards[:t]:
            if equivalence_tester(c, right):
                count += 1
	right_count += count
    if right_count > t:
        return right, right_count
    return None, 0

def MajorityExists(cards):
    card, count = FindMajority(cards)
    return card is not None
```

注：其实这道题还有 $O(n)$ 的解法，但不是分治思路。