import { type Env, json, readSession } from '../../_lib/session';

// Keys:
//   likes:count:<slug>  -> integer string
//   likes:user:<slug>:<userId> -> "1" (presence = liked)

function countKey(slug: string) {
  return `likes:count:${slug}`;
}
function userKey(slug: string, userId: number | string) {
  return `likes:user:${slug}:${userId}`;
}

async function readCount(env: Env, slug: string): Promise<number> {
  const raw = await env.LIKES_KV.get(countKey(slug));
  const n = raw ? parseInt(raw, 10) : 0;
  return Number.isFinite(n) && n >= 0 ? n : 0;
}

function sanitizeSlug(slug: string): string | null {
  if (!slug) return null;
  const s = decodeURIComponent(slug);
  if (s.length > 256) return null;
  if (!/^[\w\u4e00-\u9fff\u3400-\u4dbf./\-\s]+$/u.test(s)) return null;
  return s;
}

export const onRequestGet: PagesFunction<Env> = async ({ request, env, params }) => {
  const slug = sanitizeSlug(params.slug as string);
  if (!slug) return json({ error: 'bad slug' }, { status: 400 });

  const user = await readSession(request, env);
  const count = await readCount(env, slug);
  let liked = false;
  if (user) {
    liked = (await env.LIKES_KV.get(userKey(slug, user.id))) === '1';
  }
  return json({ count, liked, user });
};

export const onRequestPost: PagesFunction<Env> = async ({ request, env, params }) => {
  const slug = sanitizeSlug(params.slug as string);
  if (!slug) return json({ error: 'bad slug' }, { status: 400 });

  const user = await readSession(request, env);
  if (!user) return json({ error: 'unauthorized' }, { status: 401 });

  let action: 'like' | 'unlike' | 'toggle' = 'toggle';
  try {
    const body = (await request.json()) as { action?: string };
    if (body?.action === 'like' || body?.action === 'unlike') action = body.action;
  } catch {
    // empty body = toggle
  }

  const uKey = userKey(slug, user.id);
  const already = (await env.LIKES_KV.get(uKey)) === '1';
  const shouldLike = action === 'toggle' ? !already : action === 'like';

  if (shouldLike === already) {
    // Already in desired state; just return count.
    const count = await readCount(env, slug);
    return json({ count, liked: shouldLike, user });
  }

  const current = await readCount(env, slug);
  const next = Math.max(0, current + (shouldLike ? 1 : -1));
  await env.LIKES_KV.put(countKey(slug), String(next));
  if (shouldLike) {
    await env.LIKES_KV.put(uKey, '1');
  } else {
    await env.LIKES_KV.delete(uKey);
  }
  return json({ count: next, liked: shouldLike, user });
};
