import {
  type Env,
  randomState,
  stateCookie,
} from '../../../_lib/session';

export const onRequestGet: PagesFunction<Env> = async ({ request, env }) => {
  if (!env.GITHUB_CLIENT_ID) {
    return new Response('GITHUB_CLIENT_ID not configured.', { status: 500 });
  }

  const url = new URL(request.url);
  const returnTo = url.searchParams.get('return_to') ?? '/';
  const state = `${randomState()}|${encodeURIComponent(returnTo)}`;

  const redirectUri = `${url.origin}/api/auth/github/callback`;

  const authorize = new URL('https://github.com/login/oauth/authorize');
  authorize.searchParams.set('client_id', env.GITHUB_CLIENT_ID);
  authorize.searchParams.set('redirect_uri', redirectUri);
  authorize.searchParams.set('scope', 'read:user');
  authorize.searchParams.set('state', state);
  authorize.searchParams.set('allow_signup', 'true');

  return new Response(null, {
    status: 302,
    headers: {
      Location: authorize.toString(),
      'Set-Cookie': stateCookie(state),
    },
  });
};
