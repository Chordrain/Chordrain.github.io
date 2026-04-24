#!/usr/bin/env bash
# Build the static site into ./dist and run type checking first.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d node_modules ]; then
  echo "▸ Installing dependencies…"
  npm install
fi

echo "▸ Migrating content from ../content (Hugo) into src/content (Astro)…"
node scripts/migrate-content.mjs

echo "▸ Building Astro site…"
npm run build

echo "✓ Build finished. Output: ./dist"
