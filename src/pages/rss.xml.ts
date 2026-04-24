import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { SITE } from '~/config';
import { entryPath, getPublishedPosts } from '~/lib/posts';

export async function GET(context: APIContext) {
  const posts = await getPublishedPosts('zh-cn');
  return rss({
    title: SITE.title['zh-cn'],
    description: SITE.description,
    site: context.site ?? 'https://caeastro.pages.dev',
    items: posts.map((p) => ({
      title: p.data.title,
      description: p.data.description ?? '',
      pubDate: p.data.date,
      link: entryPath(p),
      categories: (p.data.categories as string[]) ?? [],
    })),
  });
}
