import { type Env, clearSessionCookie, json } from '../../_lib/session';

export const onRequestPost: PagesFunction<Env> = async ({ env }) => {
  return json({ ok: true }, { headers: { 'Set-Cookie': clearSessionCookie(env) } });
};

export const onRequestGet: PagesFunction<Env> = async ({ env }) => {
  return json({ ok: true }, { headers: { 'Set-Cookie': clearSessionCookie(env) } });
};
