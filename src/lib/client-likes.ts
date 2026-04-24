// Fills every `[data-like-count-display][data-slug=…]` badge on the page
// by making one batch request to `/api/likes?slugs=…`. Runs on first paint
// and after every Astro view transition so SPA navigation stays in sync.
//
// The individual `Likes.astro` widget dispatches a `like-updated` event
// when the visitor (un)likes from the big button — we listen to that and
// refresh any matching badges so counts stay consistent without a reload.

const BATCH_ENDPOINT = '/api/likes';

async function fillAll(root: ParentNode = document) {
  const nodes = Array.from(
    root.querySelectorAll<HTMLElement>('[data-like-count-display][data-slug]'),
  );
  if (!nodes.length) return;

  const slugs = Array.from(
    new Set(
      nodes
        .map((n) => n.dataset.slug?.trim())
        .filter((s): s is string => Boolean(s)),
    ),
  );
  if (!slugs.length) return;

  const qs = slugs.map((s) => encodeURIComponent(s)).join(',');
  let counts: Record<string, number> = {};
  try {
    const r = await fetch(`${BATCH_ENDPOINT}?slugs=${qs}`, { credentials: 'include' });
    if (r.ok) {
      const j = (await r.json()) as { counts?: Record<string, number> };
      counts = j.counts ?? {};
    }
  } catch {
    // Offline / function not deployed — leave badges as "—".
    return;
  }

  nodes.forEach((n) => {
    const slug = n.dataset.slug;
    if (!slug) return;
    const num = slug in counts ? counts[slug] : 0;
    const span = n.querySelector('span.tabular-nums') ?? n.querySelector('span');
    if (span) span.textContent = String(num);
  });
}

function updateOne(slug: string, count: number) {
  const selector = `[data-like-count-display][data-slug="${CSS.escape(slug)}"]`;
  document.querySelectorAll<HTMLElement>(selector).forEach((n) => {
    const span = n.querySelector('span.tabular-nums') ?? n.querySelector('span');
    if (span) span.textContent = String(count);
  });
}

document.addEventListener('like-updated', (e) => {
  const detail = (e as CustomEvent<{ slug?: string; count?: number }>).detail;
  if (detail?.slug && typeof detail.count === 'number') {
    updateOne(detail.slug, detail.count);
  }
});

fillAll();
document.addEventListener('astro:after-swap', () => fillAll());
