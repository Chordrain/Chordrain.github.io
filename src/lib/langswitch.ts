import type { Locale } from '../config';
import { localePath } from './i18n';
import {
  entryPath,
  type Post,
  postCounterpart,
  translateCategory,
  translateTag,
} from './posts';

export type SwitchDescriptor =
  | { kind: 'path'; path: string }
  | { kind: 'post'; post: Post }
  | { kind: 'tag'; name: string }
  | { kind: 'category'; name: string };

/**
 * Build the "switch to the other language" link for the current page.
 *
 * Unlike the naive prefix strip/add, this resolves:
 *   * post slug collisions (in practice posts share slugs; the counterpart is
 *     still looked up via the shared [PREFIX 0x??] key);
 *   * tag / category translations using the position-based mapping from
 *     `getTranslationIndex()`;
 *   * the "no counterpart" case, which routes to a friendly `/unavailable`
 *     landing page rather than producing a hard 404.
 */
export async function buildSwitchHref(
  desc: SwitchDescriptor,
  from: Locale,
  to: Locale,
): Promise<string> {
  if (from === to) {
    return desc.kind === 'path' ? localePath(desc.path, to) : '#';
  }

  switch (desc.kind) {
    case 'path': {
      return localePath(desc.path, to);
    }
    case 'post': {
      const cp = await postCounterpart(desc.post, to);
      if (cp) return entryPath(cp);
      return unavailableHref(to, {
        from,
        what: 'post',
        name: desc.post.data.title,
        altUrl: entryPath(desc.post),
      });
    }
    case 'tag': {
      const translated = await translateTag(desc.name, from, to);
      if (translated) return localePath(`/tags/${translated}`, to);
      return unavailableHref(to, {
        from,
        what: 'tag',
        name: desc.name,
        altUrl: localePath(`/tags/${desc.name}`, from),
      });
    }
    case 'category': {
      const translated = await translateCategory(desc.name, from, to);
      if (translated) return localePath(`/categories/${translated}`, to);
      return unavailableHref(to, {
        from,
        what: 'category',
        name: desc.name,
        altUrl: localePath(`/categories/${desc.name}`, from),
      });
    }
  }
}

interface UnavailablePayload {
  from: Locale;
  what: 'post' | 'tag' | 'category';
  name: string;
  altUrl: string;
}

function unavailableHref(to: Locale, payload: UnavailablePayload): string {
  const qs = new URLSearchParams({
    from: payload.from,
    what: payload.what,
    name: payload.name,
    alt: payload.altUrl,
  });
  return `${localePath('/unavailable', to)}?${qs.toString()}`;
}
