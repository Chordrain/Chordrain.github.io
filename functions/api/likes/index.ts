import { type Env, json } from '../../_lib/session';

// GET /api/likes?slugs=a,b,c  →  { counts: { a: 12, b: 0, c: 5 } }
// Used by post cards / lists to fetch several counts in one round-trip.

const MAX_SLUGS = 60;

function sanitizeSlug(slug: string): string | null {
  if (!slug) return null;
  if (slug.length > 256) return null;
  if (!/^[\w\u4e00-\u9fff\u3400-\u4dbf./\-\s]+$/u.test(slug)) return null;
  return slug;
}

export const onRequestGet: PagesFunction<Env> = async ({ request, env }) => {
  const url = new URL(request.url);
  const raw = url.searchParams.get('slugs') ?? '';
  const slugs = raw
    .split(',')
    .map((s) => decodeURIComponent(s.trim()))
    .filter(Boolean)
    .slice(0, MAX_SLUGS);

  const counts: Record<string, number> = {};
  await Promise.all(
    slugs.map(async (raw) => {
      const slug = sanitizeSlug(raw);
      if (!slug) return;
      const v = await env.LIKES_KV.get(`likes:count:${slug}`);
      const n = v ? parseInt(v, 10) : 0;
      counts[slug] = Number.isFinite(n) && n >= 0 ? n : 0;
    }),
  );

  return json({ counts });
};
