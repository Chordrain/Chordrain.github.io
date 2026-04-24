#!/usr/bin/env bash
# Build the static site into ./dist and run type checking first.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d node_modules ]; then
  echo "▸ Installing dependencies…"
  npm install
fi

# Only run the Hugo→Astro migration when the sibling Hugo content directory
# actually exists (i.e. local development). On Cloudflare Pages the repo is
# cloned standalone and src/content/posts/ is already committed, so we skip
# the migration entirely to avoid ENOENT errors.
HUGO_CONTENT="$(dirname "$0")/../../content/posts"
if [ -d "$HUGO_CONTENT" ]; then
  echo "▸ Migrating content from ../content (Hugo) into src/content (Astro)…"
  node scripts/migrate-content.mjs
else
  echo "▸ Skipping migration (Hugo sibling directory not found — CI/CD mode)."
fi

echo "▸ Building Astro site…"
npm run build

echo "✓ Build finished. Output: ./dist"
