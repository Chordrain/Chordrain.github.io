#!/usr/bin/env bash
# Create a new post skeleton.
#
# Usage:
#   ./scripts/new-post.sh "我的新文章" [category] [locale]
#     locale = zh-cn | en   (default: zh-cn)
set -euo pipefail
cd "$(dirname "$0")/.."

TITLE="${1:-}"
CATEGORY="${2:-随笔}"
LOCALE="${3:-zh-cn}"

if [ -z "$TITLE" ]; then
  echo "Usage: $0 <title> [category] [zh-cn|en]" >&2
  exit 1
fi

SLUG=$(node -e "
  const t = process.argv[1];
  const out = t
    .replace(/\\[([^\\]]+)\\]/g, '\$1')
    .replace(/[()]/g, '')
    .replace(/,/g, '')
    .replace(/\\s+/g, '-')
    .replace(/-+/g, '-');
  console.log(out);
" "$TITLE")

DIR="src/content/posts/${LOCALE}/${CATEGORY}"
FILE="${DIR}/${SLUG}.md"
mkdir -p "$DIR"

DATE=$(date +%Y-%m-%d)
cat > "$FILE" <<EOF
---
title: "${TITLE}"
date: ${DATE}
draft: true
math: false
description: ""
categories:
  - ${CATEGORY}
tags: []
---

在这里开始写作吧！
EOF

echo "✓ Created: $FILE"
