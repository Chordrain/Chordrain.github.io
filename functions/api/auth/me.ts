import { type Env, json, readSession } from '../../_lib/session';

export const onRequestGet: PagesFunction<Env> = async ({ request, env }) => {
  const user = await readSession(request, env);
  return json({ user });
};
