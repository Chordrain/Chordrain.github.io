export const SITE = {
  title: "Caeastro 小特",
  shortName: 'Caeastro',
  description: "Welcome to Caeastro's white space.",
  defaultLocale: 'zh-cn' as const,
  supportedLocales: ['zh-cn', 'en'] as const,
  author: {
    name: 'Caeastro',
    title: '梦游中……',
    description:
      '这里是我的知识库。部分笔记可能只有英文或中文版本，需切换语言才能看到。',
    avatar: '/avatar.JPG',
    social: [
      { name: 'GitHub', url: 'https://github.com/chordrain', icon: 'github' },
      { name: 'X', url: 'https://twitter.com/elonmusk', icon: 'x' },
      { name: 'Email', url: 'mailto:51285902022@stu.ecnu.edu.cn', icon: 'email' },
    ],
  },
  /** Home page shows this many recent posts. */
  recentPostsCount: 5,
  /** Pagination size for /posts listing. */
  postsPerPage: 8,
  /** Content width (must match CSS variable `--content-width`). */
  contentWidth: '56rem',
  giscus: {
    enabled: true,
    repo: 'Chordrain/Chordrain.github.io',
    repoId: 'R_kgDOKShMng',
    category: 'General',
    categoryId: 'DIC_kwDOKShMns4C7mvS',
    mapping: 'pathname',
    strict: '0',
    reactionsEnabled: '1',
    emitMetadata: '0',
    inputPosition: 'bottom',
    theme: 'preferred_color_scheme',
    lang: 'zh-CN',
  },
  likes: {
    enabled: true,
    /** GitHub OAuth app client id (public). Client secret lives only on CF Pages env. */
    githubClientId: '',
  },
} as const;

export const NAV: {
  id: string;
  label: { 'zh-cn': string; en: string };
  href: string;
}[] = [
  { id: 'posts', label: { 'zh-cn': '文章', en: 'Posts' }, href: '/posts' },
  { id: 'categories', label: { 'zh-cn': '分类', en: 'Categories' }, href: '/categories' },
  { id: 'tags', label: { 'zh-cn': '标签', en: 'Tags' }, href: '/tags' },
  { id: 'archives', label: { 'zh-cn': '归档', en: 'Archives' }, href: '/archives' },
  { id: 'about', label: { 'zh-cn': '关于', en: 'About' }, href: '/about' },
];

export type Locale = (typeof SITE.supportedLocales)[number];
