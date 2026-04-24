# Caeastro Astro Blog

这是一个基于 [Astro](https://astro.build) 重构的博客，视觉延续 Hugo `hugo-narrow` 主题的简洁质感，同时借助 **Cloudflare Pages + Pages Functions** 加入了：

- **GitHub 登录**（OAuth2）
- **文章点赞**（自己实现，数据存 Cloudflare KV，每个 GitHub 账号一票）
- **评论**（Giscus，走 GitHub Discussions，同样要登录 GitHub）
- 浅色 / 深色双主题，多端自适应，阅读进度条，侧栏 TOC，返回顶部等
- 分类 / 标签 / 归档 / 相关文章 / 上一页下一页 / RSS / Sitemap

> 💡 **目录结构一句话**：markdown 文章放在 `src/content/posts/<语言>/<分类>/<slug>.md`，图片放在 `public/images/...`，其余的你基本不用改。

---

## 目录

1. [快速开始](#快速开始)
2. [目录结构](#目录结构)
3. [写文章](#写文章)
  - [分类、标签、归档怎么工作](#分类标签归档怎么工作)
  - [Markdown 写作规则（继承自旧 Hugo 博客）](#markdown-写作规则继承自旧-hugo-博客)
  - [图片](#图片)
  - [公式](#公式)
  - [代码](#代码)
  - [多语言](#多语言)
4. [从旧 Hugo 博客迁移](#从旧-hugo-博客迁移)
5. [主题与外观](#主题与外观)
6. [评论（Giscus）](#评论giscus)
7. [点赞 + GitHub 登录（Cloudflare Pages Functions）](#点赞--github-登录cloudflare-pages-functions)
  - [本地调试](#本地调试)
  - [部署到 Cloudflare Pages](#部署到-cloudflare-pages)
8. [常见问题](#常见问题)

---

## 快速开始

```bash
# 1. 安装依赖 & 启动本地开发服务器（不含 likes / OAuth，纯前端预览）
./scripts/dev.sh
# → http://localhost:4321

# 2. 构建生产站点
./scripts/build.sh
# → 产物在 ./dist

# 3. 用 Wrangler 本地跑完整站点（含 likes、OAuth）
./scripts/preview.sh
# → http://localhost:8788
```

环境：macOS Apple Silicon、Node 20+（已在 Node 25 测过）。

> 第一次 `./scripts/dev.sh` 会自动 `npm install`。

---

## 目录结构

```
astro-blog/
├─ public/                       # 静态资源（迁移脚本会把 ../static/ 整个拷过来）
│  ├─ avatar.JPG
│  └─ images/
│     ├─ Algorithm Design/...
│     └─ Software Analysis.../...
├─ src/
│  ├─ config.ts                  # 站点标题、作者、Giscus、点赞等配置
│  ├─ content.config.ts          # Astro Content Collections schema
│  ├─ content/
│  │  ├─ posts/
│  │  │  ├─ zh-cn/<分类>/<slug>.md
│  │  │  └─ en/<分类>/<slug>.md
│  │  └─ pages/
│  │     ├─ about.zh-cn.md
│  │     └─ about.en.md
│  ├─ layouts/
│  │  ├─ BaseLayout.astro
│  │  └─ PostLayout.astro
│  ├─ components/                # Header / Footer / PostCard / TOC / Likes / Comments / ...
│  ├─ pages/                     # 路由
│  │  ├─ index.astro             # 中文首页
│  │  ├─ posts/
│  │  ├─ categories/
│  │  ├─ tags/
│  │  ├─ archives.astro
│  │  ├─ about.astro
│  │  ├─ rss.xml.ts
│  │  └─ en/                     # 英文版镜像（/en/*）
│  ├─ styles/global.css          # 迁移自 hugo-narrow 的排版与主题 tokens
│  └─ lib/                       # 工具函数 + 自定义 remark/rehype 插件
├─ functions/                    # Cloudflare Pages Functions（仅在 CF Pages / wrangler 下运行）
│  ├─ _lib/session.ts            # HMAC 签名的 session cookie + 工具
│  └─ api/
│     ├─ auth/
│     │  ├─ github/login.ts
│     │  ├─ github/callback.ts
│     │  ├─ logout.ts
│     │  └─ me.ts
│     └─ likes/[slug].ts         # GET 返回计数, POST 点赞/取消
├─ scripts/
│  ├─ dev.sh
│  ├─ build.sh
│  ├─ preview.sh                 # Wrangler 本地跑完整站点（含 KV、Functions）
│  ├─ deploy.sh                  # 一键部署到 Cloudflare Pages
│  ├─ new-post.sh                # 新文章脚手架
│  └─ migrate-content.mjs        # 从 ../content/posts（Hugo）迁移
├─ astro.config.mjs
├─ wrangler.toml                 # Pages Functions + KV 本地绑定
├─ .dev.vars.example             # 本地密钥模板
└─ package.json
```

---

## 写文章

### 创建一篇

```bash
./scripts/new-post.sh "我的新文章" 随笔 zh-cn
#                         ^title    ^category  ^locale
```

会生成：

```markdown
---
title: "我的新文章"
date: 2026-04-22
draft: true            # 发布时改 false
math: false            # 需要 KaTeX 渲染公式时改 true
description: ""        # 可选，用于卡片摘要/OG/SEO
categories:
  - 随笔
tags: []
---

在这里开始写作吧！
```

**哪里决定这篇文章的 URL？** 是 `src/content/posts/<locale>/<category>/<slug>.md` 的路径。例如 `src/content/posts/zh-cn/算法设计与分析/master-method.md` 会渲染到：

```
/posts/算法设计与分析/master-method    ← （中文，locale = zh-cn 作为默认语言，路径不加前缀）
```

英文版则是 `/en/posts/...`。

> Astro 会对路径段做小写化 + 简单 slugify，所以实际 URL 会是全部小写。这点与 Hugo 行为一致。

### 分类、标签、归档怎么工作

- **分类**：`categories: [...]` 数组。每个值会：
  - 聚合到 `/categories` 页面（按文章数排序）
  - 生成独立 `/categories/<name>` 列表页
  - 显示在文章卡片与正文顶部
- **标签**：`tags: [...]` 数组，用法同上，页面是 `/tags`、`/tags/<name>`。
- **归档**：`/archives` 自动按年份倒序列出所有非草稿文章。
- **草稿**：`draft: true` 的文章不会出现在任何列表、RSS 或相关文章里。

> 💡 一篇文章可以有多个分类 / 标签。分类偏宏观（一个系列 / 一个方向），标签偏细节（一个技术点）。

### Markdown 写作规则（继承自旧 Hugo 博客）

为了让旧文章零改动跑起来，迁移脚本 + 自定义 remark 插件已经把下面这些 Hugo-narrow 的习惯搬过来了：


| 你在旧博客的写法                                      | 新博客会如何处理                                                     |
| --------------------------------------------- | ------------------------------------------------------------ |
| `==重点==`                                      | 渲染为带高亮底色的 `<mark>`（见 `memo.txt` 第 5 条）                       |
| `<font style="background: #FEEDD5;">…</font>` | 原样输出；`prose` 样式会把 `font[style*=background]` 统一着色成同一种 mark 风格 |
| `<!--more-->`                                 | 无害（不再用来做摘要分割）                                                |
| `<center>…</center>`                          | 原样输出；在 prose 里 `<center>` / 只含 `<img>` 的段落会自动居中              |
| 内联数学 `\( … \)`                                | 等价于 `$ … $`（自定义 `remark-backslash-math` 插件自动转换）              |
| 块数学 `\[ … \]`                                 | 等价于 `$$ … $$`                                                |
| 行内 `$x^2$` / 块 `$$…$$`                        | 走 remark-math + rehype-katex                                 |
| 表格 / 列表 / 脚注                                  | 原样渲染，样式沿用 hugo-narrow                                        |


### 图片

放在 `public/images/<分组>/<文件名>`，Markdown 里按旧习惯写：

```markdown
<center>
  <img src="/images/Algorithm Design/主方法-01.png" alt="递归树"/>
</center>
```

或者标准 Markdown 也行：

```markdown
![递归树](/images/Algorithm Design/主方法-01.png)
```

只含图片的段落会自动居中，不一定非要 `<center>`。

### 公式

在 frontmatter 里加 `math: true`（可选，目前 KaTeX 始终加载，但加这个 flag 是个好习惯）。然后按旧博客的规则写就行。记一下 `memo.txt` 里的坑：

1. 公式里的 `*` 要么两边加空格，要么用花括号包起来
2. 尽量用空格分隔
3. 内联公式里的 `{}` 记得用 `\\` 转义 → 新博客下也一样，KaTeX 和 Hugo 的 markup 一致
4. 大于 / 小于号优先用命令：`\gt` / `\lt` / `\ge` / `\le`
5. 别用行内 `\(` 时同一行还跟着别的反斜杠命令粘在一起；加空格更稳

### 代码

Shiki 内置多语言高亮（light 主题 `github-light`，dark 主题 `github-dark-dimmed`）。三个反引号正常写就行：

```
```cpp
#include 
int main() { std::cout << "hi\n"; }
```

```

### 多语言

- 默认 `zh-cn`，URL 不加前缀
- 英文版落在 `src/content/posts/en/...`，URL 以 `/en/` 开头
- **跨语言切换靠笔记编号**：每篇笔记的标题都以 `[SERIES 0xNN]`（如 `[SATV 0x02]`、`[KATA 0x01]`）开头，这个编号在中英文版里必须保持完全一致。`src/lib/posts.ts` 的 `getTranslationIndex()` 会扫所有文章，按这个编号把中英文版两两配对。
- **标签/分类的对应关系来自位置对应**：对任意一对中英文文章，`tags` 数组里第 `i` 个标签彼此翻译、`categories` 数组里第 `i` 个分类彼此翻译。因此 Header 右上角的语言切换按钮能从 `/tags/IR` 正确跳到 `/tags/中间表示`，从 `/categories/软件分析测试与验证` 跳到 `/en/categories/Software Analysis, Testing and Verification`。
- **没有对应翻译怎么办**：Header 的语言切换按钮不会给你一个 404，而是跳到 `/unavailable`（或 `/en/unavailable`）—— 这是一张友好的落地页，会告诉访客"这篇文章 / 标签 / 分类还没有 XX 语言版本"，并给出回到原页面 / 回到首页的按钮。`/unavailable` 页面本身带 `<meta name="robots" content="noindex">`，不会被搜索引擎收录。
- **写作时请保证**：给同一篇笔记写双语时，① 标题里的 `[SERIES 0xNN]` 前缀完全一致；② `tags:` 数组长度和顺序一一对应（第 `i` 个 ZH 标签就是第 `i` 个 EN 标签的翻译）；③ `categories:` 同理。位置错乱会让映射表学到错误的配对。

---

## 从旧 Hugo 博客迁移

第一次 clone 或者想把新写的 Hugo 文章拉进来：

```bash
./scripts/build.sh       # build.sh 会先跑 migrate-content.mjs
# 或单独跑：
node scripts/migrate-content.mjs
```

脚本会：

1. 读 `../content/posts/**/*.md`，探测 locale（`.en.md` / `.zh-cn.md` 后缀，默认 `zh-cn`）
2. 清洗 frontmatter（保留 `title/date/categories/tags/draft/math/note/description/series`）
3. 把结果写到 `src/content/posts/<locale>/<原目录>/<slug>.md`
4. 把 `../content/about/_index.*.md` 复制到 `src/content/pages/about.*.md`
5. 把 `../static/` 整个 rsync 到 `./public/`

**注意**：迁移脚本是"覆盖式"的，如果你改过 `src/content/posts/...` 里的文件，下次迁移会被盖掉。建议：

- 批量导入旧文章 / 图片 → 用迁移脚本
- 在新博客上新增或修改 → 直接改 `src/content/posts/...`

---

## 主题与外观

- **设计 tokens** 在 `src/styles/global.css` 里，用 OKLCH 表示。浅色对应 `:root`，深色对应 `.dark`，完全沿用了 hugo-narrow 的默认主题。
- **字体**：默认 `Inter` + 系统中文栈（苹方 / 微软雅黑）+ `JetBrains Mono`。改 `--font-sans` / `--font-mono` 即可替换。
- **宽度**：`--content-width: 56rem`，和旧主题一致。
- **排版（prose）**：h1 有底部渐变短划线，h2 有左边的主色竖条，blockquote 左边 4px 主色 + 柔色背景，代码块有圆角 + 边框 + 等宽字体……全部对齐 hugo-narrow。
- **阅读进度条**：自动插入 body 顶部，无需配置。
- **TOC**：文章右侧栏目大屏显示，小屏自动隐藏。滚动时自动高亮当前标题。

想新增一个配色：在 `global.css` 里仿着 `:root` / `.dark` 的格式加一段 `[data-theme="..."] { ... }`，然后在 Header 加一个切换按钮设置 `document.documentElement.dataset.theme` 即可。

---

## 评论（Giscus）

新博客复用了你原来已经配好的 Giscus。要让它跑起来：

1. 打开 [giscus.app](https://giscus.app) → 填你的仓库（公开仓库）。
2. 在目标仓库里把 **Discussions** 开起来。
3. 安装 [giscus app](https://github.com/apps/giscus)。
4. 填一个 Discussion 分类（建议 **General**），页面会生成你要的 `repoId` 与 `categoryId`。
5. 把这两个值填到：
  ```ts
   // src/config.ts
   export const SITE = {
     giscus: {
       enabled: true,
       repo: "你的名字/你的仓库",
       repoId: "R_xxx...",
       category: "General",
       categoryId: "DIC_xxx...",
       mapping: "pathname",
       // …其他保持默认即可
     },
   }
  ```
6. 重新 build / deploy 即可。

效果：每篇文章底部一个 Giscus 面板，评论就是仓库的 Discussions，点赞（Reactions）也能用。所有人都必须登录 GitHub 才能评论 / 点赞 Reaction，完美满足"追踪用户"的需求。

---

## 点赞 + GitHub 登录（Cloudflare Pages Functions）

Giscus 的 Reactions 已经能当作"点赞"。但用户希望有一个独立的、更显眼的点赞按钮，并且 Cloudflare Pages Functions 本身也是本次重构的重要理由。所以我们**另外做了一套**点赞：

- 按钮样式极简、配主题色，点击后心形填充 + 计数 +1
- 后端存 Cloudflare KV，每个 GitHub 账号每篇文章只能点 1 次（再点取消）
- 未登录用户看到"使用 GitHub 登录"的链接
- 登录成功后自动回到原文章

### 原理概览

```
┌─── 浏览器 ────┐        ┌────── Cloudflare Pages Function ─────┐
│  Likes.astro  │ GET /api/likes/:slug  →  读 KV，返回 {count,liked,user}
│   ↓ 点击      │ POST /api/likes/:slug →  改 KV，返回新 {count,liked}
│ HeartButton   │ GET /api/auth/github/login  →  302 到 GitHub
│               │ GET /api/auth/github/callback → 换 token → 写 HMAC-签名 cookie
└───────────────┘        └──────────────────────────────────────┘
                              读 KV: binding = LIKES_KV
```

Session 不用数据库，直接用 HMAC-SHA256 签名 cookie（见 `functions/_lib/session.ts`），密钥从环境变量 `SESSION_SECRET` 注入，30 天过期。

### 本地调试

1. 在 GitHub 建一个 **OAuth App**：[https://github.com/settings/developers](https://github.com/settings/developers) → New OAuth App。
  - Homepage URL: `http://localhost:8788`
  - Authorization callback URL: `http://localhost:8788/api/auth/github/callback`
2. 拿到 `Client ID` 和生成一个 `Client Secret`。
3. 复制本地变量模板：
  ```bash
   cp .dev.vars.example .dev.vars
   # 填入 GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET
   # SESSION_SECRET 用：openssl rand -hex 48
  ```
4. 启动：
  ```bash
   ./scripts/preview.sh
  ```
   打开 [http://localhost:8788](http://localhost:8788)，进入任意文章底部点"使用 GitHub 登录" → 授权 → 跳回原文章 → 按心形点赞。再次点击取消。
  > Wrangler 在本地用 SQLite 模拟 KV，重启不会丢数据。清零：`rm -rf .wrangler`。

### 部署到 Cloudflare Pages

第一次部署分几步，之后只用 `./scripts/deploy.sh` 或者 push 到 GitHub 就行。

#### 1. 建一个 Pages 项目

- 去 Cloudflare Dashboard → Workers & Pages → Create → Pages → Connect to Git
- 选你的仓库，分支、根目录（`astro-blog/`）
- **Build command**: `bash ./scripts/build.sh`
- **Build output directory**: `dist`
- **Node version**: 20 或以上（Pages → Settings → Environment → `NODE_VERSION=20`）

或者直接用 CLI 推一次，项目会自动创建：

```bash
CF_PAGES_PROJECT=caeastro-blog ./scripts/deploy.sh
```

#### 2. 创建一个生产 KV Namespace

```bash
npx wrangler kv namespace create LIKES_KV
# 输出类似：id = "abc123..."
```

把 `abc123...` 填到 Cloudflare Pages 项目 → Settings → **Functions → KV namespace bindings**：


| Variable name | KV namespace |
| ------------- | ------------ |
| `LIKES_KV`    | 选刚才创建的那个     |


#### 3. 配环境变量 & Secrets

在同一个 Settings 页面的 **Environment variables**：


| 变量名                    | 环境         | 值                                           |
| ---------------------- | ---------- | ------------------------------------------- |
| `GITHUB_CLIENT_ID`     | Production | 新建一个生产 OAuth App 的 client id                |
| `GITHUB_CLIENT_SECRET` | Production | 对应 secret（**勾 Encrypted**）                  |
| `SESSION_SECRET`       | Production | `openssl rand -hex 48` 的输出（**勾 Encrypted**） |
| `COOKIE_DOMAIN`        | Production | 自定义域时填 `.yourdomain.com`，否则留空               |


（Preview 环境也可以同样配一套，指向同一个或单独的 OAuth App。）

**重要**：OAuth App 的回调 URL 必须写生产域名：

```
https://caeastro-blog.pages.dev/api/auth/github/callback
# 或者你的自定义域
```

#### 4. 部署

手动：

```bash
./scripts/deploy.sh                 # 会先 build，再 wrangler pages deploy
```

或者接上 GitHub，Cloudflare Pages 每次 push 自动构建。

---

## 常见问题

**文章里的 ==高亮== 没有颜色？**

确认你的文本没有被代码块或 HTML 包裹。插件只处理 markdown 正文的 text 节点。如果你在旧博客里手写的是 `<font style="background: #FEEDD5;">…</font>`，也会被 CSS 统一着色，颜色受主题色值约束——是按主题的亮底色设计的，深色模式会自动换成深底 + 浅字。

**公式渲染出错 / 不渲染？**

1. 确认 `memo.txt` 里那几条规则：
  - `*` 前后加空格，或者 `{*}`
  - `{}` 在内联公式中用 `\\` 转义（其实 KaTeX 里写 `\{\}` 就行）
  - 大于小于号用 `\gt` / `\lt` / `\ge` / `\le`
2. 启动时看终端，`rehype-katex` 报错会打印行号。
3. 打开浏览器控制台，KaTeX 渲染失败的公式会有红色高亮。

**点赞按钮一直显示"登录以点赞"？**

1. 如果是本地，你必须用 `./scripts/preview.sh`（8788 端口）而不是 `./scripts/dev.sh`（4321 端口）——Pages Functions 只在 Wrangler 里能跑。
2. 检查 `.dev.vars` 是否填了 `GITHUB_CLIENT_ID`、`SESSION_SECRET`。
3. OAuth App 的回调 URL 必须和当前访问的 origin 完全一致。

**怎么一次性"清掉所有点赞"？**

```bash
# 本地
rm -rf .wrangler

# 生产（谨慎）
npx wrangler kv key list --namespace-id <id> | jq -r '.[].name' \
  | xargs -n1 npx wrangler kv key delete --namespace-id <id>
```

**部署后没看见评论？**

检查 `src/config.ts` 里的 `giscus.repoId` / `categoryId` 是否还留着 `REPLACE_ME`。如果是，组件会显示一条"尚未配置"的提示。换成真值并重新部署即可。

**想加一张新图片 / 删一张旧图片？**

直接操作 `public/images/...`。Astro 对 `public/` 的资源不做处理，原路径原样提供——也就是说你 markdown 里写的 `/images/xxx/yyy.png` 就是最终 URL。

**如何改站点标题 / 作者信息 / 社交链接？**

改 `src/config.ts` 的 `SITE.author` / `SITE.title` 等字段即可。不会影响已发布文章的 URL。

**我习惯** `hugo --cleanDestinationDir build`**，现在怎么做？**

`./scripts/build.sh` 会每次都重新 migrate + build 到干净的 `./dist`。`dist/` 在 `.gitignore` 里，和旧博客一样不会被提交。不会出现旧博客 `localhost` 残留的问题。

---

## 许可

- 代码：MIT
- 文章：CC BY-NC-SA 4.0（单篇页底部有提示）

