// Shared helpers for the Cloudflare Pages Functions layer.
// Runs on the Workers runtime (Web Crypto / no Node APIs).

export interface SessionUser {
  id: number;
  login: string;
  avatar_url?: string;
  name?: string | null;
}

export interface Env {
  LIKES_KV: KVNamespace;
  /** GitHub OAuth app – https://github.com/settings/developers */
  GITHUB_CLIENT_ID: string;
  GITHUB_CLIENT_SECRET: string;
  /** Random long string, used to sign session cookies. */
  SESSION_SECRET: string;
  /** Optional override of the cookie domain (e.g. `.example.com`). */
  COOKIE_DOMAIN?: string;
}

const COOKIE_NAME = 'caeastro_session';
const STATE_COOKIE = 'caeastro_oauth_state';
const SESSION_TTL_SEC = 60 * 60 * 24 * 30; // 30 days

/* --------------------------- crypto utilities ---------------------------- */
const enc = new TextEncoder();
const dec = new TextDecoder();

async function hmacKey(secret: string) {
  return crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign', 'verify']
  );
}

function b64urlEncode(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let str = '';
  for (let i = 0; i < bytes.length; i++) str += String.fromCharCode(bytes[i]);
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function b64urlDecode(str: string): Uint8Array {
  const pad = str.length % 4 === 0 ? '' : '='.repeat(4 - (str.length % 4));
  const b64 = (str + pad).replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function sign(payload: string, secret: string): Promise<string> {
  const key = await hmacKey(secret);
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(payload));
  return b64urlEncode(sig);
}

async function verify(payload: string, sig: string, secret: string): Promise<boolean> {
  try {
    const key = await hmacKey(secret);
    return await crypto.subtle.verify('HMAC', key, b64urlDecode(sig), enc.encode(payload));
  } catch {
    return false;
  }
}

/* ---------------------------- session cookies ---------------------------- */
interface SessionPayload {
  user: SessionUser;
  exp: number; // seconds
}

export async function createSession(user: SessionUser, env: Env): Promise<string> {
  const payload: SessionPayload = {
    user,
    exp: Math.floor(Date.now() / 1000) + SESSION_TTL_SEC,
  };
  const data = b64urlEncode(enc.encode(JSON.stringify(payload)));
  const sig = await sign(data, env.SESSION_SECRET);
  return `${data}.${sig}`;
}

export async function readSession(request: Request, env: Env): Promise<SessionUser | null> {
  const cookie = parseCookie(request.headers.get('Cookie'))[COOKIE_NAME];
  if (!cookie) return null;
  const [data, sig] = cookie.split('.');
  if (!data || !sig) return null;
  if (!(await verify(data, sig, env.SESSION_SECRET))) return null;
  try {
    const obj = JSON.parse(dec.decode(b64urlDecode(data))) as SessionPayload;
    if (obj.exp * 1000 < Date.now()) return null;
    return obj.user;
  } catch {
    return null;
  }
}

export function sessionCookie(token: string, env: Env): string {
  const domain = env.COOKIE_DOMAIN ? `; Domain=${env.COOKIE_DOMAIN}` : '';
  return `${COOKIE_NAME}=${token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${SESSION_TTL_SEC}${domain}`;
}

export function clearSessionCookie(env: Env): string {
  const domain = env.COOKIE_DOMAIN ? `; Domain=${env.COOKIE_DOMAIN}` : '';
  return `${COOKIE_NAME}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0${domain}`;
}

export function stateCookie(state: string): string {
  return `${STATE_COOKIE}=${state}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=600`;
}

export function readStateCookie(request: Request): string | null {
  return parseCookie(request.headers.get('Cookie'))[STATE_COOKIE] ?? null;
}

export function clearStateCookie(): string {
  return `${STATE_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`;
}

/* -------------------------- cookie parsing ------------------------------- */
function parseCookie(header: string | null): Record<string, string> {
  const out: Record<string, string> = {};
  if (!header) return out;
  for (const part of header.split(/;\s*/)) {
    const idx = part.indexOf('=');
    if (idx < 0) continue;
    const k = part.slice(0, idx).trim();
    const v = part.slice(idx + 1).trim();
    if (k) out[k] = decodeURIComponent(v);
  }
  return out;
}

/* --------------------------------- misc ---------------------------------- */
export function json(data: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(data), {
    status: init.status ?? 200,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
      ...(init.headers as Record<string, string> | undefined),
    },
  });
}

export function randomState(): string {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return b64urlEncode(bytes);
}
