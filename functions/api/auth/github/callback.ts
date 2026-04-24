import {
  type Env,
  type SessionUser,
  clearStateCookie,
  createSession,
  readStateCookie,
  sessionCookie,
} from '../../../_lib/session';

export const onRequestGet: PagesFunction<Env> = async ({ request, env }) => {
  const url = new URL(request.url);
  const code = url.searchParams.get('code');
  const state = url.searchParams.get('state');
  const cookieState = readStateCookie(request);

  if (!code || !state || !cookieState || state !== cookieState) {
    return new Response('Invalid OAuth state.', { status: 400 });
  }

  // Decode return_to from state (`<rand>|<url>`).
  const [, returnToEnc = '/'] = state.split('|');
  let returnTo = '/';
  try {
    returnTo = decodeURIComponent(returnToEnc);
    if (!returnTo.startsWith('/')) returnTo = '/';
  } catch {}

  // 1. Exchange code → access_token.
  const tokenRes = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'User-Agent': 'caeastro-blog',
    },
    body: JSON.stringify({
      client_id: env.GITHUB_CLIENT_ID,
      client_secret: env.GITHUB_CLIENT_SECRET,
      code,
      redirect_uri: `${url.origin}/api/auth/github/callback`,
    }),
  });
  if (!tokenRes.ok) {
    return new Response('Failed to exchange OAuth code.', { status: 502 });
  }
  const tokenData = (await tokenRes.json()) as { access_token?: string; error?: string };
  if (!tokenData.access_token) {
    return new Response(`GitHub OAuth error: ${tokenData.error ?? 'unknown'}`, { status: 502 });
  }

  // 2. Fetch user profile.
  const userRes = await fetch('https://api.github.com/user', {
    headers: {
      Authorization: `Bearer ${tokenData.access_token}`,
      Accept: 'application/vnd.github+json',
      'User-Agent': 'caeastro-blog',
    },
  });
  if (!userRes.ok) {
    return new Response('Failed to fetch GitHub user.', { status: 502 });
  }
  const user = (await userRes.json()) as SessionUser;

  const token = await createSession(
    {
      id: user.id,
      login: user.login,
      avatar_url: user.avatar_url,
      name: user.name ?? null,
    },
    env
  );

  const headers = new Headers();
  headers.append('Set-Cookie', sessionCookie(token, env));
  headers.append('Set-Cookie', clearStateCookie());
  headers.set('Location', returnTo);
  return new Response(null, { status: 302, headers });
};
