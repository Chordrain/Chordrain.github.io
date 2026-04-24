---
title: "[KATA 0x02] Codeforces Round 1084 (Div. 3)"
date: 2025-04-12
draft: true
math: true
categories:
  - 算法笔记
tags: []
---

赛事🔗：[Codeforces Round 1084 (Div. 3)](https://codeforces.com/contest/2200)

最近在投某大厂实习，看了一下历年来的机试题感觉单刷力扣根本不够，又来刷 cf 了……感觉一年没登这个平台了（其实真的过了一年）。这次只做出来 4 道题，如果要是能稳定做出 1000-1500 分的题目，90% 的面试题应该就不在话下了。

## A. Eating Game

原题🔗：[A. Eating Game](https://codeforces.com/contest/2200/problem/A)

分值：800

题目大意：在一张环形的桌子上有 $n$ 位参赛者，$i$ 号参赛者面前有 $a_{i}$ 盘菜，轮到某位参赛者时，该名参赛者可以吃 $1$ 盘菜，然后轮到 $i\ \mathrm{mod}\ n + 1$ 号参赛者，如果参赛者面前没有菜了就跳过，谁最后吃完谁赢。可以从任意参赛者开始游戏，问一轮游戏里可能有多少赢家。

肯定是菜多的人赢啊，本质上就是统计数组 $a$ 中有几个最大值，代码：

```cpp
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

void solve() {
    int n, m = -1, cnt = 0;
    cin >> n;
    vector<int> a(n);
    for (int& ai : a) {
        cin >> ai;
        m = max(m, ai);
    }
    for (int& ai : a) {
        cnt += ai == m;
    }
    cout << cnt << '\n';
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

## B. Deletion Sort

原题🔗：[B. Deletion Sort](http://codeforces.com/contest/2200/problem/B)

分值：800

题目大意：给定一个含有 $n$ 个正整数的数组 $a$，如果 $a$ 是非递减的，那么游戏结束；否则，不断从 $a$ 中删除元素，直到 $a$ 是非递减的。问游戏结束时，$a$ 中至少剩几个元素。

智商检测题。问至少剩几个元素，那就是要删尽可能多的元素，假设在一个序列里发现了一个破坏非递减性质的元素为 $b$，$b$ 肯定大于前面的元素，小于后面的元素。正常人可能会删除后面的元素，那样就直接保证非递减了。但这道题的目标不是排序，而是删元素。我们可以把排序的目标扔一边，由于一旦排好了游戏就结束了，所以我们在删完尽可能多的元素之前都不能让游戏结束，所以，我们可以先把 $b$ 前面的元素一个个全删了，此时游戏不会结束，然后再把 $b$ 后面的一个个元素全删了，这样最后就只剩下了 $b$，答案为 1。没错，只要数组不是一开始就非递减的，那答案永远为 1.

这里唯一要注意的是不要自作聪明在输入还没读完的时候就直接输出答案，这样会导致输入输出错位。代码：

```cpp
#include <iostream>
#include <vector>
using namespace std;

void solve() {
    int n; cin >> n;
    vector<int> a(n);
    for (int& ai : a) {
        cin >> ai;
    }
    for (int i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1]) {
            cout << 1 << '\n';
            return;
        }
    }
    cout << n << '\n';
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

## C. Specialty String

原题🔗：[C. Specialty String](https://codeforces.com/contest/2200/problem/C)

分值：900

题目大意：给定一个初始只含有小写字母的字符串，游戏规则是，如果两个相同的字符之间全是星号，那就可以把这两个字符也替换成星号。这个操作可以反复无数次，直到再没有可替换的字符为止，此时如果整个字符串都只含星号，那就胜利；否则失败。给你一个这样的字符串，如果能胜利，就输出 `YES`，不能就输出 `NO`。

解法其实是模拟，核心是中心扩散。仔细思考就能明白，初始字符串中不存在任何星号，那么要能替换出星号，只有在有两个连续相同字符时才行。那么我先把字符串中所有连续的相同字符替换成星号，再遍历一遍，对所有星号的两边进行扩散，只要星号被两个相同字符包裹，那就把这两个字符替换成星号，继续扩散。最终，如果可能，那么整个字符串都会变成星号，输出 `YES`，否则输出 `NO`。

下面是代码：

```cpp
#include <iostream>
#include <string>
using namespace std;

void solve() {
    int n; cin >> n;
    string s; cin >> s;
    for (int i = 0; i < n - 1; i++) {
        if (s[i] == s[i + 1]) {
            s[i] = s[i + 1] = '*';
            i++;
        }
    }
    for (int i = 0; i < n; i++) {
        int j = i, k = i + 1;
        while (j >= 0 && s[j] == '*' && j--);
        while (k < n  && s[k] == '*' && k++);
        while (j >= 0 && k < n && s[j] == s[k]) {
            s[j--] = s[k++] = '*';
        }
    }
    for (int i = 0; i < n; i++) {
        if (s[i] != '*') {
            cout << "NO\n";
            return;
        }
    }
    cout << "YES\n";
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

其实这道题我尝试了好几遍，最开始写出的代码是这样的：

```cpp
void solve() {
    int n; cin >> n;
    string s; cin >> s;
    for (int i = 0; i < n - 1; i++) {
        if (s[i] == s[i + 1]) {
            s[i] = s[i + 1] = '*';
            int j = i, k = i + 1;
            while (j >= 0 && s[j] == '*' && j--);
            while (k < n  && s[k] == '*' && k++);
            while (j >= 0 && k < n && s[j] == s[k]) {
                s[j--] = s[k++] = '*';
            }
            i++;
        }
    }
    for (int i = 0; i < n; i++) {
        if (s[i] != '*') {
            cout << "NO\n";
            return;
        }
    }
    cout << "YES\n";
}
```

和最终版的区别是，最终版先把相同连续字符替换成星号，然后再去扩散；而这一版是两步同时做，结果 WA 了。后来我意识到，扩散过程可能会因为后续字符尚未被替换成星号而被破坏，于是我就把 `for` 重复了一遍，结果过了，证明了我的猜想，之后才改成了最终版的样子。让 AI 帮我找了个反例：`xaaybbyx`。这就是我说的情况，第一次扩散 `aa` 时，第一个 `y` 应该和 `bb` 后面的 `y` 匹配，第一个 `x` 应该和 `bby` 后面的 `x` 匹配，但因为 `bb` 是后面才会被替换的，所以匹配被隔断了。

## D. Portal

原题🔗：[D. Portal](https://codeforces.com/contest/2200/problem/D)

分值：1300

题目大意：给定一个含有 $n$ 个数字的数组，这个数组中存在两个传送门，你可以把一个传送门左边的元素移动到另一个传送门右边，也可以把一个传送门右边的元素传送到另一个传送门左边，这两种操作可以进行任意次。现在给你一个数组，和两个传送门的位置，问使用这两种操作，你能得到的字典序最小的数组是什么。

1000-1300 分这个区间的题都属于逻辑思维题，侧重于对题目的理解和观察，而不是对算法的熟练度。这么一看我脑子真是锈掉了，思考了好久……

我们根据传送门的位置把数组分成左中右（$L,M,R$）三部分，仔细寻思就会发现，两个传送门能实现：

1. $M$ 循环移动；
2. $L+R$ 左右平移。

所以我们要把 $L$ 和 $R$ 当作一个整体来考虑，只要 $M$ 和 $L+R$ 都尽可能保持字典序最小就行了。$M$ 考虑起来较为简单，找到最小的那个数，把它循环移动到 $M$ 的开头；$L+R$ 不能循环移动，因此顺序是固定的，那就要将 $L+R$ 中的数字，从左到右依次和 $M$ 中最小数做比较，找到那个分界点，即 $L+R$ 中第一个大于 $M$ 中最小值的数，把从这个数开始的所有数（包括分界点）都移到 $R$ 中去，最后把 $L,M,R$ 组装起来就行了。

下面是代码：

```cpp
#include <iostream>
#include <vector>
using namespace std;

void solve() {
    int n, x, y;
    cin >> n >> x >> y;
    vector<int> p(n);
    for (int& c : p) {
        cin >> c;
    }
    // 方便起见先把M和L+R单独提取出来
    vector<int> M(p.begin() + x, p.begin() + y);
    vector<int> LR;
    for (int i = 0; i < p.size(); ) {
        if (i == x) {
            i = y;
            continue;
        }
        LR.push_back(p[i]);
        i++;
    }
    int M_start = 0, LR_split = -1;
    // 找到M中的最小值
    for (int i = 0; i < M.size(); i++) {
        M_start = M[M_start] > M[i] ? i : M_start;
    }
    vector<int> _M, ans;
    // 把M_start循环移动到最开头并存储到_M中
    for (int i = M_start, r = 0; r < M.size(); r++, i++) {
        i = i % M.size();
        _M.push_back(M[i]);
    }
    // 找到LR中的分割点
    for (int i = 0; M.size() && i < LR.size(); i++) {
        if (LR[i] > M[M_start]) {
            LR_split = i;
            break;
        }
    }
    // 构造正确答案
    if (LR_split != -1) {
        ans.insert(ans.end(), LR.begin(), LR.begin() + LR_split);
        ans.insert(ans.end(), _M.begin(), _M.end());
        ans.insert(ans.end(), LR.begin() + LR_split, LR.end());
    } else {
        ans.insert(ans.end(), LR.begin(), LR.end());
        ans.insert(ans.end(), _M.begin(), _M.end());
    }
    // 输出
    for (const int& c : ans) {
        cout << c << ' ';
    }
    cout << '\n';
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

## E. Divisive Battle

原题🔗：[E. Divisive Battle](https://codeforces.com/contest/2200/problem/E)

分值：1500

题目大意：Alice 和 Bob 在玩一个游戏，游戏规则是，给定一个正整数数组，如果数组是非递减的，那么 `Bob` 获胜，否则，必须从数组中选一个数进行因数分解（不能是 1 和它自身），然后把它用这两个因数替换掉（顺序自选），如此反复，直到无数可分解为止，此时如果数组是非递减的，`Bob` 获胜，否则 `Alice` 获胜。`Alice` 先手，假定两人都采取最优策略行动，问谁能赢。

太难了😭根本不会做。其实一开始是意识到了的：只要数组中出现两个质数逆序对，那直接就可以宣判 Alice 赢了。但问题是我要如何知道分解之后会不会出现质数逆序对？如果要把一个数所有的质因数都找出来也太耗时了。想了很久都没想出来，最终放弃了。

一个数被分解到最后肯定都是质因数，一旦质因数被排成逆序对，那就没有挽回的余地了。题目说两人都会采取最优策略，那么对于 Alice 而言，只要是合数，她肯定会第一时间分解出它的最小质因数，然后把最小质因数放在前面，创造逆序对，这样她肯定就赢了。而 Alice 又是先手，所以只要数组不是一开始就排好了，那按理来讲怎么都是 Alice 赢。这是我一开始的想法，但其实不是，因为我漏掉了一种情况——如果一个数的所有质因数相同，比如说 8，这个数只有质因数 2，那么它怎么排都不会破坏非递减性质，这种情况下 Bob 仍有可能赢。所以算法流程应该是：

1. 判断数组是否一开始就是非递减的，是输出 `Bob`，不是转步骤 2；
2. 判断数组中是否存在数字有 2 个及以上的质因数，如果存在，那么先手的 Alice 肯定会赢，输出 `Alice`，否则转步骤 3；
3. 判断数组中是否存在数字只有单一质因数，如果存在，将它替换为这个数，再检查一遍数组是否非递减，是输出 `Bob`，不是输出` Alice`。

所以，其实我们关注的只是质因数的数量和最小的那个质因数。而事实上，只要我们能找出最小质因数，质因数总数就很好求了。找最小质因数可以用埃筛或者欧筛，这里我选择更简单的埃筛，也能过。找到了最小质因数之后，只要不断从原数中不断除以这个最小质因数（去重），直到除不尽为止，再去找除完之后的数的最小质因数，如此反复，就能求出一个数的质因数数量了。

下面是代码：

```cpp
#include <iostream>
#include <vector>
using namespace std;

vector<int> spf;

void init_spf(int limit) {
    spf.resize(limit + 1);
    for (int i = 1; i <= limit; i++) {
        spf[i] = i;
    }
    for (int i = 2; i * i <= limit; i++) {
        if (spf[i] == i) {
            for (int j = i * i; j <= limit; j += i) {
                if (spf[j] == j) {
                    spf[j] = i;
                }
            }
        }
    }
}

int count_distinct_prime_factor(int num) {
    int distinct = 0;
    while (num > 1) {
        int val = spf[num];
        distinct++;
        while (num % val == 0) {
            num /= val;
        }
    }
    return distinct;
}

bool is_non_decreasing(vector<int>& a) {
    if (a.size() < 2) {
        return true;
    }
    for (int i = 1; i < a.size(); i++) {
        if (a[i] < a[i - 1]) {
            return false;
        }
    }
    return true;
}

void solve() {
    int n;
    cin >> n;
    vector<int> a(n), prime_factor_count(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        prime_factor_count[i] = count_distinct_prime_factor(a[i]);
    }
    if (is_non_decreasing(a)) {
        cout << "Bob\n";
        return;
    }
    for (int cpf : prime_factor_count) {
        if (cpf >= 2) {
            cout << "Alice\n";
            return;
        }
    }
    for (int i = 0; i < prime_factor_count.size(); i++) {
        if (prime_factor_count[i] == 1) {
            a[i] = spf[a[i]];
        }
    }
    if (is_non_decreasing(a)) {
        cout << "Bob\n";
    } else {
        cout << "Alice\n";
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    init_spf(1e6);
    int t; cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```

## F. Mooclear Reactor 2

原题🔗：[F. Mooclear Reactor 2](https://codeforces.com/contest/2200/problem/F)

分值：1900

题目大意：Bessie 手上有 $n$ 个粒子，每个粒子可产生的能量记作 $x$，每个粒子还有一个限制 $y$，意思是当选择这个粒子来产能时，Bessie 最多还能再选 $y$ 个其他粒子。这里还有一个商店，商店里有 $m$ 个粒子，Bessie 可以从中购买一个，但是可以买了不用，问在**分别**购买商店中各个粒子的情况下，Bessie 最多可以产生多少能量。

如果没有这个商店那我还能尝试一下，有商店就算了。这里比较 tricky 的点在于数量限制，而我们可以去枚举选取的粒子总数 $k$。假设 $f(k)$ 代表在我选取 $k$ 个粒子时（不包括商店）所能得到的最大能量值，那么在没有商店的情况下，$\max(f)$ 就是答案。计算 $f$ 的方式其实很简单，由于只限制共存粒子的数量，而要拿的粒子数已经确定了，那么接下来就是能拿粒子（即 $k - 1 < y$ 的粒子）中能量尽可能大的我们都拿上，这一步就是贪心的思想，所以我们需要事先对自己手里的粒子按能量大小进行排序。

但是光有 $f$ 还不够，问题问的是在购买商店里某个粒子的情况下所能得到最大能量，所以我们不得不考虑商店里的每一个粒子。既然要加上商店里的粒子，那就必须给它腾出一个位置，则当我们讨论 $k$ 时，我们应该限制拿了 $k-1$ 个之后就不能再拿了，剩下那个位置留给商店。于是设置 $g(k)$，表示在要拿 $k$ 个粒子的情况下，只拿 $k-1$ 个手上的粒子时的最大能量值。计算方式跟 $f$ 基本一样，区别只是不能拿满 $k$ 个。

特别地，如果我要拿 $k$ 或 $k-1$ 个粒子，但是已经没有粒子可以拿了，注意要令 $f(k)=g(k)=-\infty$，表示这个方案不可行。

剩下的就是去枚举每一个商店里的粒子了，对于商店里的每一个粒子，我们去求：

1. 不使用这个粒子时的最大值，即 $\max(f(1),\dots,f(n))$；
2. 使用这个粒子时的最大值，即 $\max(g(1)+y,\dots,g(n+1)+y)$。

最终答案就是上面两者取较大的。下面是代码：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct particle {
    long long x;
    int y;
};

void solve() {
    int n, m;
    cin >> n >> m;
    vector<particle> bessie(n + 1), shop(m + 1);
    for (int i = 1; i <= n; i++) {
        cin >> bessie[i].x >> bessie[i].y;
    }
    for (int i = 1; i <= m; i++) {
        cin >> shop[i].x >> shop[i].y;
    }
    sort(bessie.begin() + 1, bessie.end(), [](const particle& p1, const particle& p2) {
        return p1.x > p2.x;
    });
    vector<long long> f(n + 1, 0), g(n + 2, 0);
    for (int k = 1; k < f.size(); k++) {
        int picked = 0;
        for (int i = 1; i < bessie.size() && picked < k; i++) {
            if (k - 1 <= bessie[i].y) {
                f[k] += bessie[i].x;
                picked += 1;
            }
        }
        if (picked != k) {
            f[k] = -1e18;
        }
    }
    for (int k = 1; k < g.size(); k++) {
        int picked = 0;
        for (int i = 1; i < bessie.size() && picked < k - 1; i++) {
            if (k - 1 <= bessie[i].y) {
                g[k] += bessie[i].x;
                picked += 1;
            }
        }
        if (picked != k - 1) {
            g[k] = -1e18;
        }
    }    
    for (int i = 1; i < shop.size(); i++) {
        long long ans = -1;
        for (int k = 1; k < g.size(); k++) {
            if (k - 1 <= shop[i].y) {
                ans = max(ans, shop[i].x + g[k]);
            }
        }
        for (int k = 1; k < f.size(); k++) {
            ans = max(ans, f[k]);
        }
        cout << ans << " ";
    }
    cout << '\n';
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

然后你就会惊奇地发现 TLE 了。是的，这道题用上面这种暴力解法是会超时的，得用线段树进行优化。但是呢，俺不会，润了。

## G. Operation Permutation

原题🔗：[G. Operation Permutation](https://codeforces.com/contest/2200/problem/G)

分值：2200

题目大意：





































