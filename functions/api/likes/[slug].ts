import { type Env, json } from '../../_lib/session';

// KV layout
//   likes:count:<slug>           → integer string
//   likes:fp:<slug>:<fpHash>     → "1" (presence ⇒ this fingerprint liked)
//                                  Stored with TTL so old keys age out.

const FINGERPRINT_TTL_SEC = 60 * 60 * 24 * 30; // 30 days

function countKey(slug: string) {
  return `likes:count:${slug}`;
}
function fpKey(slug: string, fp: string) {
  return `likes:fp:${slug}:${fp}`;
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

/**
 * Compute a stable visitor fingerprint: SHA-256 of
 *   <ip> || "\n" || <user-agent> || "\n" || <secret>
 *
 * This is only a soft de-dupe — a visitor can still bypass it by switching
 * networks / browsers / clearing local storage. That is fine: we trade
 * absolute correctness for zero friction (no login).
 */
async function visitorFingerprint(request: Request, env: Env): Promise<string> {
  const ip =
    request.headers.get('CF-Connecting-IP') ||
    request.headers.get('X-Forwarded-For')?.split(',')[0]?.trim() ||
    'unknown';
  const ua = request.headers.get('User-Agent') || 'unknown';
  const secret = env.SESSION_SECRET || 'dev-fingerprint-salt';
  const input = `${ip}\n${ua}\n${secret}`;
  const data = new TextEncoder().encode(input);
  const hashBuf = await crypto.subtle.digest('SHA-256', data);
  const bytes = new Uint8Array(hashBuf);
  let hex = '';
  for (let i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, '0');
  // Truncate: 128 bits of entropy is more than enough for de-dupe.
  return hex.slice(0, 32);
}

export const onRequestGet: PagesFunction<Env> = async ({ request, env, params }) => {
  const slug = sanitizeSlug(params.slug as string);
  if (!slug) return json({ error: 'bad slug' }, { status: 400 });

  const [count, fp] = await Promise.all([
    readCount(env, slug),
    visitorFingerprint(request, env),
  ]);
  const liked = (await env.LIKES_KV.get(fpKey(slug, fp))) === '1';
  return json({ count, liked });
};

export const onRequestPost: PagesFunction<Env> = async ({ request, env, params }) => {
  const slug = sanitizeSlug(params.slug as string);
  if (!slug) return json({ error: 'bad slug' }, { status: 400 });

  let action: 'like' | 'unlike' | 'toggle' = 'toggle';
  try {
    const body = (await request.json()) as { action?: string };
    if (body?.action === 'like' || body?.action === 'unlike') action = body.action;
  } catch {
    // no body ⇒ toggle
  }

  const fp = await visitorFingerprint(request, env);
  const key = fpKey(slug, fp);
  const already = (await env.LIKES_KV.get(key)) === '1';
  const shouldLike = action === 'toggle' ? !already : action === 'like';

  if (shouldLike === already) {
    const count = await readCount(env, slug);
    return json({ count, liked: shouldLike });
  }

  const current = await readCount(env, slug);
  const next = Math.max(0, current + (shouldLike ? 1 : -1));
  await env.LIKES_KV.put(countKey(slug), String(next));
  if (shouldLike) {
    await env.LIKES_KV.put(key, '1', { expirationTtl: FINGERPRINT_TTL_SEC });
  } else {
    await env.LIKES_KV.delete(key);
  }
  return json({ count: next, liked: shouldLike });
};
