import { getCollection, type CollectionEntry } from 'astro:content';
import type { Locale } from '../config';
import { SITE } from '../config';

export type Post = CollectionEntry<'posts'>;

/**
 * An entry's ID looks like `zh-cn/Algorithm Design/master-method` or
 * `en/Software.../01 Intro`.
 * We derive the locale from the first path segment.
 */
export function entryLocale(entry: Post): Locale {
  const seg = entry.id.split('/')[0];
  if (seg === 'en' || seg === 'zh-cn') return seg;
  return SITE.defaultLocale;
}

export function entrySlug(entry: Post): string {
  const parts = entry.id.split('/');
  const rest = parts.slice(1);
  return rest.map(slugifySegment).join('/');
}

export function entryPath(entry: Post): string {
  const slug = entrySlug(entry);
  const locale = entryLocale(entry);
  if (locale === SITE.defaultLocale) return `/posts/${slug}`;
  return `/${locale}/posts/${slug}`;
}

const NON_URL_SAFE = /[\s]+/g;
function slugifySegment(s: string): string {
  return s
    .trim()
    .replace(/\[([^\]]+)\]/g, '$1')
    .replace(/[()]/g, '')
    .replace(/,/g, '')
    .replace(NON_URL_SAFE, '-')
    .replace(/--+/g, '-');
}

export async function getPublishedPosts(locale?: Locale): Promise<Post[]> {
  const all = await getCollection('posts', ({ data }) => !data.draft);
  const filtered = locale ? all.filter((p) => entryLocale(p) === locale) : all;
  return filtered.sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

export function groupByYear(posts: Post[]): { year: number; posts: Post[] }[] {
  const map = new Map<number, Post[]>();
  for (const p of posts) {
    const y = p.data.date.getFullYear();
    if (!map.has(y)) map.set(y, []);
    map.get(y)!.push(p);
  }
  return [...map.entries()]
    .map(([year, posts]) => ({ year, posts }))
    .sort((a, b) => b.year - a.year);
}

export function collectTaxonomy(
  posts: Post[],
  field: 'categories' | 'tags'
): { name: string; slug: string; count: number; posts: Post[] }[] {
  const map = new Map<string, Post[]>();
  for (const p of posts) {
    const values = (p.data[field] as string[]) ?? [];
    for (const v of values) {
      const name = v.trim();
      if (!name) continue;
      if (!map.has(name)) map.set(name, []);
      map.get(name)!.push(p);
    }
  }
  return [...map.entries()]
    .map(([name, ps]) => ({
      name,
      slug: slugifySegment(name).toLowerCase(),
      count: ps.length,
      posts: ps,
    }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

export function relatedPosts(entry: Post, all: Post[], limit = 3): Post[] {
  const mine = new Set([
    ...(entry.data.categories as string[]),
    ...(entry.data.tags as string[]),
  ]);
  const locale = entryLocale(entry);
  const scored = all
    .filter((p) => p.id !== entry.id && entryLocale(p) === locale)
    .map((p) => {
      const theirs = new Set([
        ...(p.data.categories as string[]),
        ...(p.data.tags as string[]),
      ]);
      let score = 0;
      for (const v of theirs) if (mine.has(v)) score++;
      return { p, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || b.p.data.date.getTime() - a.p.data.date.getTime());

  if (scored.length >= limit) return scored.slice(0, limit).map((x) => x.p);
  // Fallback to latest of the same locale.
  const fillers = all
    .filter((p) => p.id !== entry.id && entryLocale(p) === locale)
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
  const seen = new Set(scored.map((x) => x.p.id));
  const out = scored.map((x) => x.p);
  for (const f of fillers) {
    if (out.length >= limit) break;
    if (!seen.has(f.id)) {
      out.push(f);
      seen.add(f.id);
    }
  }
  return out.slice(0, limit);
}

export function adjacentPosts(entry: Post, all: Post[]): { prev?: Post; next?: Post } {
  const locale = entryLocale(entry);
  const sameLocale = all
    .filter((p) => entryLocale(p) === locale)
    .sort((a, b) => a.data.date.getTime() - b.data.date.getTime());
  const idx = sameLocale.findIndex((p) => p.id === entry.id);
  return {
    prev: idx > 0 ? sameLocale[idx - 1] : undefined,
    next: idx >= 0 && idx < sameLocale.length - 1 ? sameLocale[idx + 1] : undefined,
  };
}

export { slugifySegment };

// ---------------------------------------------------------------------------
// Cross-locale pairing
// ---------------------------------------------------------------------------
//
// Every note title is prefixed with a bracketed series + hex ordinal, e.g.
// `[SATV 0x02]`, `[ALDE 0x06]`, `[KATA 0x01]`. The prefix is deliberately
// stable across translations, so two posts sharing the same prefix are the
// same note in different languages.
//
// From each pair we derive two auxiliary mappings:
//   * tag position i in zh ↔ tag position i in en  → tag translation map
//   * the single category on each side              → category map
//
// These maps drive the language-switch button so that `/tags/IR` (en) can
// flip to `/tags/中间表示` (zh-cn) and vice versa. When no mapping exists
// we redirect to an /unavailable landing page instead of a raw 404.

const ENTRY_KEY_RE = /\[\s*([A-Za-z0-9]+)\s+0x([0-9a-fA-F]+)\s*\]/;

export function extractEntryKey(post: Post): string | null {
  const title = post.data.title ?? '';
  const m = title.match(ENTRY_KEY_RE);
  if (!m) return null;
  return `${m[1].toUpperCase()}-0x${m[2].toLowerCase()}`;
}

export interface PostPair {
  key: string;
  'zh-cn'?: Post;
  en?: Post;
}

export interface TranslationIndex {
  /** key → {zh-cn, en} post pair (partial when only one side exists). */
  pairs: Map<string, PostPair>;
  /** post.id → counterpart post in the other locale (if any). */
  postCounterpart: Map<string, Post>;
  /** tagName in sourceLocale → Record<targetLocale, translated tag>. */
  tags: Record<Locale, Map<string, Partial<Record<Locale, string>>>>;
  /** categoryName in sourceLocale → Record<targetLocale, translated category>. */
  categories: Record<Locale, Map<string, Partial<Record<Locale, string>>>>;
}

let _translationIndex: TranslationIndex | null = null;

export async function getTranslationIndex(): Promise<TranslationIndex> {
  if (_translationIndex) return _translationIndex;

  const all = await getCollection('posts');
  const pairs = new Map<string, PostPair>();
  for (const post of all) {
    const key = extractEntryKey(post);
    if (!key) continue;
    const cur = pairs.get(key) ?? ({ key } as PostPair);
    cur[entryLocale(post)] = post;
    pairs.set(key, cur);
  }

  const postCounterpart = new Map<string, Post>();
  for (const pair of pairs.values()) {
    if (pair['zh-cn'] && pair.en) {
      postCounterpart.set(pair['zh-cn'].id, pair.en);
      postCounterpart.set(pair.en.id, pair['zh-cn']);
    }
  }

  const tags: TranslationIndex['tags'] = {
    'zh-cn': new Map(),
    en: new Map(),
  };
  const categories: TranslationIndex['categories'] = {
    'zh-cn': new Map(),
    en: new Map(),
  };

  const register = (
    bucket: Map<string, Partial<Record<Locale, string>>>,
    srcLocale: Locale,
    srcName: string,
    dstLocale: Locale,
    dstName: string,
  ) => {
    if (!srcName || !dstName) return;
    const entry = bucket.get(srcName) ?? {};
    // First occurrence wins; don't silently overwrite a prior mapping.
    entry[srcLocale] = srcName;
    if (entry[dstLocale] === undefined) entry[dstLocale] = dstName;
    bucket.set(srcName, entry);
  };

  for (const pair of pairs.values()) {
    const zh = pair['zh-cn'];
    const en = pair.en;
    if (!zh || !en) continue;

    const zhTags = (zh.data.tags as string[]) ?? [];
    const enTags = (en.data.tags as string[]) ?? [];
    const tagN = Math.min(zhTags.length, enTags.length);
    for (let i = 0; i < tagN; i++) {
      register(tags['zh-cn'], 'zh-cn', zhTags[i], 'en', enTags[i]);
      register(tags.en, 'en', enTags[i], 'zh-cn', zhTags[i]);
    }

    const zhCats = (zh.data.categories as string[]) ?? [];
    const enCats = (en.data.categories as string[]) ?? [];
    const catN = Math.min(zhCats.length, enCats.length);
    for (let i = 0; i < catN; i++) {
      register(categories['zh-cn'], 'zh-cn', zhCats[i], 'en', enCats[i]);
      register(categories.en, 'en', enCats[i], 'zh-cn', zhCats[i]);
    }
  }

  _translationIndex = { pairs, postCounterpart, tags, categories };
  return _translationIndex;
}

/** Look up the counterpart post of `post` in `target` locale (undefined if none). */
export async function postCounterpart(
  post: Post,
  target: Locale,
): Promise<Post | undefined> {
  const idx = await getTranslationIndex();
  const cp = idx.postCounterpart.get(post.id);
  if (!cp) return undefined;
  return entryLocale(cp) === target ? cp : undefined;
}

/** Translate a tag name from `from` locale to `to` locale. */
export async function translateTag(
  name: string,
  from: Locale,
  to: Locale,
): Promise<string | undefined> {
  if (from === to) return name;
  const idx = await getTranslationIndex();
  return idx.tags[from].get(name)?.[to];
}

/** Translate a category name from `from` locale to `to` locale. */
export async function translateCategory(
  name: string,
  from: Locale,
  to: Locale,
): Promise<string | undefined> {
  if (from === to) return name;
  const idx = await getTranslationIndex();
  return idx.categories[from].get(name)?.[to];
}
