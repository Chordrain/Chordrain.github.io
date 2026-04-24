import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';

import { remarkMarkHighlight } from './src/lib/remark-mark.mjs';
import { remarkReadingTime } from './src/lib/remark-reading-time.mjs';
import { remarkBackslashMath } from './src/lib/remark-backslash-math.mjs';
import { rehypeExternalLinks } from './src/lib/rehype-external-links.mjs';

export default defineConfig({
  site: 'https://caeastro.pages.dev',
  trailingSlash: 'never',
  i18n: {
    defaultLocale: 'zh-cn',
    locales: ['zh-cn', 'en'],
    routing: { prefixDefaultLocale: false, redirectToDefaultLocale: false },
  },
  integrations: [
    mdx(),
    sitemap({
      i18n: {
        defaultLocale: 'zh-cn',
        locales: { 'zh-cn': 'zh-CN', en: 'en-US' },
      },
    }),
  ],
  markdown: {
    remarkPlugins: [remarkReadingTime, remarkBackslashMath, remarkMath, remarkMarkHighlight],
    rehypePlugins: [
        rehypeKatex,
        rehypeSlug,
        // Headings still carry an id (via rehype-slug) so the TOC can
        // jump to them, but we no longer append a visible `#` link.
        rehypeExternalLinks,
    ],
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      wrap: true,
    },
    remarkRehype: { footnoteLabel: 'Footnotes', footnoteBackLabel: '↩' },
  },
  vite: {
    plugins: [tailwindcss()],
    ssr: {
      // KaTeX stylesheet ships as a static asset.
    },
  },
});
