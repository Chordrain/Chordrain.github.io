import type { Locale } from '../config';
import { SITE } from '../config';

type Dict = Record<string, { 'zh-cn': string; en: string }>;

const dict: Dict = {
  'nav.posts': { 'zh-cn': '文章', en: 'Posts' },
  'nav.categories': { 'zh-cn': '分类', en: 'Categories' },
  'nav.tags': { 'zh-cn': '标签', en: 'Tags' },
  'nav.archives': { 'zh-cn': '归档', en: 'Archives' },
  'nav.about': { 'zh-cn': '关于', en: 'About' },

  'home.recent': { 'zh-cn': '最新文章', en: 'Recent posts' },
  'home.viewAll': { 'zh-cn': '查看全部 →', en: 'View all →' },

  'post.publishedAt': { 'zh-cn': '发布于', en: 'Published' },
  'post.updatedAt': { 'zh-cn': '更新于', en: 'Updated' },
  'post.readingTime': { 'zh-cn': '分钟阅读', en: 'min read' },
  'post.wordCount': { 'zh-cn': '字', en: 'words' },
  'post.category': { 'zh-cn': '分类', en: 'Category' },
  'post.tags': { 'zh-cn': '标签', en: 'Tags' },
  'post.toc': { 'zh-cn': '目录', en: 'On this page' },
  'post.related': { 'zh-cn': '相关文章', en: 'Related posts' },
  'post.prev': { 'zh-cn': '上一篇', en: 'Previous' },
  'post.next': { 'zh-cn': '下一篇', en: 'Next' },
  'post.license': {
    'zh-cn': '本文采用 CC BY-NC-SA 4.0 协议，转载请注明出处。',
    en: 'Licensed under CC BY-NC-SA 4.0. Please attribute when reusing.',
  },

  'likes.label': { 'zh-cn': '点赞', en: 'Like' },
  'likes.liked': { 'zh-cn': '已点赞', en: 'Liked' },
  'likes.loginToLike': { 'zh-cn': '登录以点赞', en: 'Sign in to like' },
  'likes.loginWithGitHub': { 'zh-cn': '使用 GitHub 登录', en: 'Sign in with GitHub' },
  'likes.logout': { 'zh-cn': '登出', en: 'Sign out' },

  'comments.title': { 'zh-cn': '评论', en: 'Comments' },

  'archive.title': { 'zh-cn': '归档', en: 'Archives' },
  'categories.title': { 'zh-cn': '分类', en: 'Categories' },
  'tags.title': { 'zh-cn': '标签', en: 'Tags' },
  'posts.title': { 'zh-cn': '所有文章', en: 'All posts' },
  'pagination.prev': { 'zh-cn': '上一页', en: 'Prev' },
  'pagination.next': { 'zh-cn': '下一页', en: 'Next' },

  'footer.poweredBy': { 'zh-cn': '由 Astro 驱动', en: 'Powered by Astro' },

  'search.placeholder': { 'zh-cn': '搜索文章…', en: 'Search posts…' },
  'search.noResults': { 'zh-cn': '没有找到文章。', en: 'No posts found.' },

  'theme.light': { 'zh-cn': '浅色', en: 'Light' },
  'theme.dark': { 'zh-cn': '深色', en: 'Dark' },

  'social.hidden': { 'zh-cn': '我才不告诉你 :)', en: "won't tell ya :)" },

  'unavailable.title': {
    'zh-cn': '没有对应的中文版本',
    en: 'No Chinese counterpart',
  },
  'unavailable.titleEn': {
    'zh-cn': '没有对应的英文版本',
    en: 'No English version yet',
  },
  'unavailable.leadPost': {
    'zh-cn': '你正在查看的文章还没有对应的中文版。',
    en: 'The post you were reading does not have an English translation yet.',
  },
  'unavailable.leadPostEn': {
    'zh-cn': '你正在查看的文章还没有对应的英文版。',
    en: 'The post you were reading does not have a Chinese translation yet.',
  },
  'unavailable.leadTag': {
    'zh-cn': '该标签目前没有对应的中文版本。',
    en: 'This tag has no English counterpart yet.',
  },
  'unavailable.leadCategory': {
    'zh-cn': '该分类目前没有对应的中文版本。',
    en: 'This category has no English counterpart yet.',
  },
  'unavailable.altCta': {
    'zh-cn': '回到原页面',
    en: 'Back to the original page',
  },
  'unavailable.homeCta': {
    'zh-cn': '回到首页',
    en: 'Go to home',
  },
  'unavailable.hint': {
    'zh-cn': '如果你希望看到中文版本，欢迎在 GitHub 仓库提 issue 或 PR。',
    en: 'Want to see this in English? Feel free to open an issue or PR.',
  },
};

export function t(key: keyof typeof dict | string, locale: Locale): string {
  const entry = dict[key as keyof typeof dict];
  if (!entry) return key;
  return entry[locale] ?? entry['zh-cn'];
}

export function localePath(path: string, locale: Locale): string {
  const clean = path.startsWith('/') ? path : `/${path}`;
  if (locale === SITE.defaultLocale) return clean.replace(/\/+$/, '') || '/';
  return (`/${locale}${clean}`).replace(/\/+$/, '') || `/${locale}`;
}

export function detectLocale(pathname: string): Locale {
  if (pathname.startsWith('/en')) return 'en';
  return 'zh-cn';
}

export function stripLocale(pathname: string): string {
  if (pathname.startsWith('/en/')) return pathname.slice(3);
  if (pathname === '/en') return '/';
  return pathname;
}

export function otherLocale(locale: Locale): Locale {
  return locale === 'zh-cn' ? 'en' : 'zh-cn';
}

export function formatDate(date: Date | string, locale: Locale): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const fmt = new Intl.DateTimeFormat(locale === 'zh-cn' ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  return fmt.format(d);
}
