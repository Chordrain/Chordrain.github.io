---
title: "[KATA 0x01] Codeforces Round 1009 (Div. 3)"
date: 2025-04-12
draft: false
math: true
categories:
  - 算法笔记
tags: []
---

赛事🔗：[Codeforces Round 1009 (Div. 3)](https://codeforces.com/contest/2074)。

记录第一次打 codeforces 周赛，虽然大家都说 div 3 算是比较适合新手入门了，但其实也蛮考验编程功底的，思维量对于新手来讲还是大的。最终战绩是做了四道题，我太菜了……不论如何写个题解，锻炼一下脑力。

## A. Draw a Square

原题🔗：[A. Draw a Square](https://codeforces.com/contest/2074/problem/A)

题目大意：给定四个坐标，形如 $(-l,0), (r,0), (0,-d), (0,u)$，问它们是否能构成一个矩形。

这就是 div 3 的难度吗？爱了爱了。代码：

```cpp
#include <iostream>
using namespace std;

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t; cin >> t;
    while (t--) {
        int l, r, d, u;
        cin >> l >> r >> d >> u;
        if (l == r && r == d && d == u) cout << "Yes" << '\n';
        else cout << "No" << '\n';
    }
    return 0;
}
```

## B. The Third Side

题目🔗：[B - The Third Side](https://codeforces.com/contest/2074/problem/B)

题目大意：给定一个整数数组 $a$，你要不断地从中选出两个数字 $a_{i},a_{j}$，并再找到一个整数 $x$，使得以它们为长度的三条边能构成一个三角形，然后将 $a_{i},a_{j}$ 从 $a$ 中删除，再把 $x$ 加入 $a$，如此循环往复，问你 $a$ 中剩下的最后一个数的最大值为多少。

智商检测题，最大的数字肯定是 $x$。由于三角形边长的性质就是两边之和必定大于第三条边，因此 $x$ 要取最大的话，必定是 $a_{i}+a_{j}-1$，只要每次都这样取，最后剩下的数字肯定会最大。但是真去模拟就太麻烦了，可以想到，由于每一轮都有 $x=a_{i}+a_{j}-1$，最终的 $x$ 将会把所有元素都加一遍，又由于每轮都要减 1，所以还得减去数组长度，即 $x_{\max} = \Sigma a_{i} - |a|$，

代码：

```cpp
#include <iostream>
#include <vector>
using namespace std;

void solve() {
    int n; cin >> n;
    vector<int> a(n);
    for (int& _a : a) {
        cin >> _a;
    }
    if (n == 1) {
        cout << a[0] << endl;
        return;
    }
    int ans = 0;
    for (int& _a : a) {
        ans += _a;
    }
    ans -= a.size() - 1;
    cout << ans << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t; cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```

## C. XOR and Triangle

原题🔗：[C. XOR and Triangle](https://codeforces.com/contest/2074/problem/C)

题目大意：题目给出一个 $x$ （$x \ge 2$），要求我们找到一个 $y$（$x \gt y \gt 0$），使得三条长度分别为 $x$、$y$ 和 $x \oplus y$ 的边能组成一个三角形。

这道题 分值 1400。能组成一个三角形的条件是任意两边之和大于第三条边，则我们找到的 $y$ 必须满足：

1. $x + y \gt x \oplus y$
2. $x + (x \oplus y) \gt y$
3. $y + (x \oplus y) \gt x$

异或操作是不进位加法，$x$ 和 $y$ 又是正数，所以肯定有 $0 \lt x \oplus y \le x + y$，又 $x \gt y$，条件 2 天然成立，所以主要看条件 1 和 3。异或和加法就差在进位，这里存在一个公式：
$$
x + y = (x \oplus y) + 2 \times (x \land y)
$$
稍微解释一下这个公式：$x \land y$ 完美找出了所有的进位位，乘 $2$ 是因为进位要加到更高的位上去，相当于左移一位。

将公式代入条件 1，得到 $x \land y$ 必须大于 $0$，即 $x+y$ 必须产生进位，那就是要求 $x$ 和 $y$ 至少有一位都为 $1$。

将公式代入条件 3，得到 $y \gt x \land y$，这一点如何保证？仔细思考一下会发现，$\land$ 运算只有在 $1 \land 1$ 时才得 $1$，否则都为 $0$，也就是说在 $\land$ 运算下，对应的数位要么不变，要么变小，有 $x \land y \ge y$，所以我们只需要保证 $y \neq x \land y$ 就行了。而要保证这一点，就只要 $y$ 和 $x$ 在某一位上不一样就行了。再加上题目要求 $x \gt y$，所以更准确来讲，是要保证至少在某一位上，$x$ 为 $1$，$y$ 为 $0$。

整理一下，现在我们要保证下面两个新条件：

1. $x$ 和 $y$ 至少有一位都为 $1$（条件 1）；
2. $x$ 中为 $1$ 的某位上，$y$ 中那一位为 $0$（条件 3）。

那么接下来就是想一种通用算法来构造这个 $y$。但在此之前还要排出两种无解的情况：

1. $x=2^{n}$，此时 $x$ 中只有最高位为 $1$，其余都为 $0$，此时找不出任何 $y$ 能满足上面两条；
2. $x=2^{n}-1$，此时 $x$ 全为 $1$，也无法同时满足上面两条。

上面两种情况都应该直接输出 `-1`。至于构造 $y$ 的方法，我们可以想一个很直观的，直接取 $y=2^{k-1}-1$（其中 $k$ 为 $x$ 的二进制位数），即对于任意有解的 $x$，我们直接令 $y$ 的第 $k$ 位为 $0$，其余位都取 $1$，这样直接就保证了两个条件。

总之这就是这道题的思路了，下面是代码：

```cpp
#include <iostream>
using namespace std;

void solve() {
    int x; cin >> x;
    if ((x & (x - 1)) == 0 || (x & (x + 1)) == 0)
        cout << -1 << '\n';
	else {
        int k = 32 - __builtin_clz(x);
        cout << (1 << (k - 1)) - 1 << '\n';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t; cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```

## D. Counting Points

原题🔗：[D. Counting Points](https://codeforces.com/contest/2074/problem/D)

题目大意：给定 $n$ 个圆心在 x 轴上的圆，每个圆的半径表示为 $r_{i}$，所有圆的半径之和为 $m$，问这些圆覆盖了多少二维坐标系上的整数坐标点。

分值 1400。说实话这道题比 C 题要简单点。这道题的难点主要在于如何处理重叠部分，我采取的是扫描线法——枚举每一个落在圆内的 x 坐标，然后计算在穿过这个坐标的竖线上有多少个落在圆内的点。具体的做法就是计算竖线与每个圆的相交形成的弦的长度，这个长度取下底就是 y 坐标的最大值，则这条弦上的点的个数就是 $y*2+1$。

显然这个过程中可以优化的点就是如何找覆盖了 x 坐标的圆，我采取的策略是哈希，key 就是被覆盖的 x 的集合，value 是圆的编号集合。另外还要注意数据范围，这道题得用 `long long` 才能过最后一个测试用例。

具体的代码如下：

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>
using namespace std;

void solve() {
    long long n, m, ans = 0; cin >> n >> m;
    vector<long long> c(n), r(n);
    unordered_map<long long, vector<long long>> active;
    for (int i = 0; i < n; i++) {
        cin >> c[i];
    }
    for (int i = 0; i < n; i++) {
        cin >> r[i];
        for (int j = c[i] - r[i]; j <= c[i] + r[i]; j++) {
            active[j].emplace_back(i);
        }
    }
    for (const auto& pair : active) {
        long long x = pair.first, y = -1e18;
        vector<long long> circs = pair.second;
        for (const int& circ : circs) {
            y = max((long long)sqrt(r[circ] * r[circ] - (c[circ] - x) * (c[circ] - x)), y);
        }
        ans += y * 2 + 1;
    }
    cout << ans << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t; cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```

## E. Empty Triangle

原题🔗：[E. Empty Triangle](https://codeforces.com/contest/2074/problem/E)

题目大意：这是一道互动题。给定 $n$ 个坐标点，题目保证没有任何两个坐标点位置相同，没有任何三个坐标点共线。每次询问，你可以选中其中 3 个坐标点的编号，构成一个三角形，如果这个三角形覆盖了部分坐标点，系统将会从这些点中任选一个，返回给你它的编号；否则返回 `0`。在最多 75 次询问内，你要给出 3 个坐标点的编号，使得它们构成的三角形内部不覆盖任何一个点。

这道题的分值是 1600。从这道题开始就不是我这种业余选手能掺和的了，光是题面就吓死人了，第一次见到互动题。虽然是参考的别人的思路，但还是写写吧。

每次系统会返回给你一个在三角形内部的点，那么我们可以用这个点来作为新的顶点来继续构造三角形，这样就能使三角形的面积越来越小，小到一定程度时，肯定无法覆盖任何点。由于查询次数非常有限，我们需要让面积下降的速度尽可能快，而题目中写道系统具有适应性，也就是它会故意挑选那种刁钻的点，拖慢面积缩小的速度。比如说我们初始选择的三个点是 $i,j,k$，系统返回点 $p$，我们选择 $i,j,p$ 继续尝试，并且每次我们都保留 $i,j$，这样的话就很容易被针对。解决这一问题的方法就是**随机化**，每次丢弃一个随机的点，这样就不容易被系统针对。

下面是我的代码，值得注意的是随机生成器的初始化不要写在函数内部，因为创建生成器的开销巨大，放在函数内部会大大拖慢运行效率：

```cpp
#include <iostream>
#include <unordered_map>
#include <random>
#include <chrono>
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

int get_random(int r) {
    return uniform_int_distribution<int>(1, r)(rng);
}

int ask(int i, int j, int k) {
    cout << "? " << i << " " << j << " " << k << endl;
    int p;
    if (!(cin >> p) || p == -1) exit(0);
    return p;
}

void report(int i, int j, int k) {
    cout << "! " << i << " " << j << " " << k << endl;
}

void solve() {
    int n, random_num, i, j, k;
    cin >> n;
    // initialize i,j,k
    i = get_random(n);
    do {
        j = get_random(n);
    } while (i == j);
    do {
        k = get_random(n);
    } while (i == k || j == k);
    while (true) {
        int resp = ask(i, j, k);
        if (resp) {
            // discard a point arbitrarily
            switch (get_random(3)) {
                case 1:
                    i = resp;
                    break;
                case 2:
                    j = resp;
                    break;
                case 3:
                    k = resp;
            }
        } else {
            report(i, j, k);
            break;
        }
    }
}

int main() {
    int t;
    if (!(cin >> t)) return 0;
    
    while (t--) {
        solve();
    }
    
    return 0;
}
```

## F. Counting Necessary Nodes

原题🔗：[F. Counting Necessary Nodes](https://codeforces.com/contest/2074/problem/F)

题目大意：一个二维坐标系被多个不同大小的正方形（最小的正方形是 $1\times 1$ 的，四个小正方形构成更大的 $2^{2} \times 2^{2}$ 正方形，以此类推）分割,对于 $2^{k} \times 2^{k}$ 大小的瓷砖，它只能铺在左下角坐标为 $2^{k}$ 的倍数的位置。现在随便给你一块矩形区域，问你要构成这样的矩形，需要至少几个正方形。

从这题开始就是 2000 分以上的题了，好想成为算法高手啊🏳️～这道题类似于铺瓷砖，除了瓷砖形状有限制之外，瓷砖的位置也有限制，对于 $2^{k} \times 2^{k}$ 大小的瓷砖，它只能铺在左下角坐标为 $2^{k}$ 的倍数的位置。我们可以采取一种自顶向下的策略，先考虑最大的砖，再依次考虑比较小的砖，每次都从小砖的数量里减去大砖的数量，一层一层求解，直到计算到 $1\times 1$ 的砖为止。

这种策略为什么可行？因为地图最大就 $10^6$，当 $k=20$ 就能全覆盖了，也就是说最多推 21 层（包括 $k=0$）。对于每一层，我们都去放当前能放的砖的最大数量，即横坐标上能放下的数量乘上纵坐标上能放下的数量，这两个数量是可以直接计算得到的（我一开始去枚举结果超时了，蠢死了），不过最终结果还要减去上一层，也就是更大一点的砖的数量，我们只要不断把当前层的结果传递到下一层，重复以上操作，知道来到 $k=0$ 的一层就成功了。

代码：

```cpp
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

void solve() {
    int l1, r1, l2, r2, last_total = 0, ans = 0;
    cin >> l1 >> r1 >> l2 >> r2;
    int k = (int)log2(max(r1 - l1, r2 - l2)) + 1;
    while (k >= 0) {
        int len = 1 << k;
        int n1 = max(0, r1 / len - (l1 + len - 1) / len);
        int n2 = max(0, r2 / len - (l2 + len - 1) / len);
        ans += n1 * n2 - last_total * 4;
        last_total = n1 * n2;
        k--;
    }
    cout << ans << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t;
    if (!(cin >> t)) return 0;
    
    while (t--) {
        solve();
    }
    
    return 0;
}
```

## G. Game With Triangles: Season 2

原题🔗：[G. Game With Triangles: Season 2](https://codeforces.com/contest/2074/problem/G)

题目大意：给定一个正 $n$ 边形，每个顶点都有一个权重，选择三个顶点构造一个三角形，你将获得这三个顶点权重之积的分数，你可以构造多个三角形，但是要求这些三角形不能有任何重合（点和边重叠也不行），问你能取得的最大分数是多少。

一开始的思路是用二进制枚举每一种选点方案，按位与就能知道两种方案是否重复使用了某个顶点，剩下只要解决面积重叠的判定问题就行了，但很快发现不行，顶点最多有 400 个，枚举直接状态爆炸。一看题解是区间 DP，润了～

其实问题的核心在于面积重叠的判定，仔细一想会发现，每次构造出一个三角形，其实都是对多边形进行了一次空间划分。于是，我们可以考虑从小空间开始考虑，把小空间合并成大空间，最后扩展到整个多边形。

具体来讲，设置状态数组为 $F[i][j]$，表示选择 $i$ 和 $j$ 之间的顶点时，所能得到的最大分数。假设我们当前已经计算得出了 $i$ 到 $j-1$ 号顶点所围出的空间的最大分数 $F[i][j-1]$，下一步我们扩张右边界，得到 $F[i][j]$，那我们会遇到两种情况：

1. 加入新顶点 $j$ 之后 $F[i][j-1]$ 仍然是最大，所以 $F[i][j]=F[i][j-1]$；

2. 回退到更小的子空间中，例如 $i$ 到 $k$（$k \lt j-1$），腾出空间 $k+1$ 到 $j$，在这个空间中顶点 $j$ 可以构成一个新三角形使总分更上一层楼，即 $F[i][j]=F[i][k]+\text{NewTriangle}_{k+1,j,?}$。显然，要找到这个新三角形，我们还得找到除了 $j$ 以外的另外两个顶点，直接的做法是枚举，但那样的话内层循环又多出来 $O(n^{2})$ 的复杂度，这里考虑继续使用动态规划进行优化。

    维护一个辅助状态数组 $H[i][j]$，表示在必须选择 $i,j$ 号顶点的情况下，再选一个中间顶点 $k$（$i \lt k \lt j$）所能得到的最大分数，它的转移方程就很简单了，因为两个顶点已经确定，所以只需要枚举剩下一个顶点 $k$，再加上被 $k$ 分割出的子空间的最大分数 $F[i+1][k-1]$ 和 $F[k+1][j-1]$，取所有可能里的最大值即可。

    有了 $H$ 之后，我们只要枚举 $k$ 就行了，这样内层循环时间复杂度就降低到了 $O(n)$。

两个状态数组的转移方程为：
$$
H[i][j] = \max_{i < k < j} \left( a_i \cdot a_k \cdot a_j + F[i+1][k-1] + F[k+1][j-1] \right)\\
F[i][j] = \max \left( F[i][j-1], \max_{i \le k < j} (F[i][k] + H[k+1][j]) \right)
$$
如果遇到非法的状态，例如 $i \ge j$，或者 $j-i \lt 2$，那么直接设为 0 即可。

下面是代码：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void solve() {
    int n;
    cin >> n;
    
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    
    vector<vector<long long>> H(n + 1, vector<long long>(n + 1, 0));
    vector<vector<long long>> F(n + 1, vector<long long>(n + 1, 0));

    for (int r = 3; r <= n; r++) {  // 扩展右边界
        for (int l = r - 2; l > 0; l--) {   // 回退左边界，从小空间到大空间
            for (int k = l + 1; k <= r - 1; k++) {  // 选取中间顶点，除了两端不能选之外其他都能选
                H[l][r] = max(a[l] * a[k] * a[r] + F[l + 1][k - 1] + F[k + 1][r - 1], H[l][r]);
            }
            F[l][r] = F[l][r - 1];
            for (int k = l; k <= r; k++) {  // 选取可能的新三角形，可以贴合边界
                F[l][r] = max(F[l][k - 1] + H[k][r], F[l][r]);
            }
        }
    }
    
    cout << F[1][n] << "\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    
    return 0;
}
```











