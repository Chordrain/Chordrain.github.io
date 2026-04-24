#!/usr/bin/env node
/**
 * Migrate Hugo content (content/posts/**) into Astro content collections
 * (src/content/posts/<locale>/<category>/<slug>.md).
 *
 * What it does:
 *   - Reads front matter, detects locale from filename suffix (.en.md /
 *     .zh-cn.md) or Hugo `languageCode`. Defaults to zh-cn.
 *   - Strips hugo-only front matter keys (draft drafts stay).
 *   - Rewrites <!--more--> comments (Astro ignores them, but we drop noise).
 *   - Normalises image paths: leaves /images/... as-is (public/images).
 *   - Copies /static/** to /astro-blog/public/**.
 *
 * Usage:
 *   node scripts/migrate-content.mjs
 *   (run from astro-blog/)
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HUGO_ROOT = path.resolve(__dirname, '../..');
const SRC = path.join(HUGO_ROOT, 'content', 'posts');
const DST = path.resolve(__dirname, '../src/content/posts');
const PAGES_DST = path.resolve(__dirname, '../src/content/pages');
const STATIC_SRC = path.join(HUGO_ROOT, 'static');
const PUBLIC_DST = path.resolve(__dirname, '../public');

/* ------------------------------- helpers ------------------------------- */
function slugifyFileName(stem) {
  // Remove leading numeric prefix like "01 " and brackets.
  return stem
    .replace(/\[([^\]]+)\]/g, '$1')
    .replace(/\(/g, '')
    .replace(/\)/g, '')
    .replace(/,/g, '')
    .replace(/\s+/g, '-')
    .replace(/--+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function parseFrontMatter(src) {
  if (!src.startsWith('---')) return { meta: {}, body: src };
  const end = src.indexOf('\n---', 3);
  if (end < 0) return { meta: {}, body: src };
  const raw = src.slice(3, end).trim();
  const body = src.slice(end + 4).replace(/^\r?\n/, '');

  const meta = {};
  const lines = raw.split(/\r?\n/);
  let currentKey = null;
  let arrayBuffer = null;
  for (const line of lines) {
    if (!line.trim()) continue;
    if (/^[A-Za-z_][\w-]*:/.test(line)) {
      if (currentKey && arrayBuffer) {
        meta[currentKey] = arrayBuffer;
        arrayBuffer = null;
      }
      const idx = line.indexOf(':');
      const key = line.slice(0, idx).trim();
      const val = line.slice(idx + 1).trim();
      if (val === '') {
        currentKey = key;
        meta[key] = '';
      } else {
        meta[key] = parseScalar(val);
        currentKey = null;
      }
    } else if (/^\s+-\s+/.test(line)) {
      if (!arrayBuffer) arrayBuffer = [];
      const item = line.replace(/^\s+-\s+/, '').trim();
      arrayBuffer.push(parseScalar(item));
    } else if (currentKey && !arrayBuffer) {
      meta[currentKey] = (meta[currentKey] ? meta[currentKey] + ' ' : '') + line.trim();
    }
  }
  if (currentKey && arrayBuffer) meta[currentKey] = arrayBuffer;
  return { meta, body };
}

function parseScalar(v) {
  const s = v.trim();
  if (s === 'true') return true;
  if (s === 'false') return false;
  if (s === 'null' || s === '') return null;
  if (/^\[.*\]$/.test(s)) {
    // JSON-ish inline array
    try {
      return JSON.parse(s.replace(/'/g, '"'));
    } catch {
      return s
        .slice(1, -1)
        .split(',')
        .map((x) => x.trim().replace(/^"|"$/g, '').replace(/^'|'$/g, ''));
    }
  }
  if (/^".*"$/.test(s)) return s.slice(1, -1);
  if (/^'.*'$/.test(s)) return s.slice(1, -1);
  return s;
}

function stringifyYaml(obj) {
  const lines = ['---'];
  const keys = ['title', 'date', 'updated', 'draft', 'math', 'description', 'categories', 'tags', 'series', 'note'];
  const remaining = Object.keys(obj).filter((k) => !keys.includes(k));
  for (const k of [...keys, ...remaining]) {
    if (!(k in obj)) continue;
    const v = obj[k];
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) {
      if (v.length === 0) {
        lines.push(`${k}: []`);
      } else {
        lines.push(`${k}:`);
        for (const item of v) lines.push(`  - ${yamlScalar(item)}`);
      }
    } else if (typeof v === 'boolean' || typeof v === 'number') {
      lines.push(`${k}: ${v}`);
    } else if (v instanceof Date) {
      lines.push(`${k}: ${v.toISOString().slice(0, 10)}`);
    } else {
      lines.push(`${k}: ${yamlScalar(v)}`);
    }
  }
  lines.push('---', '');
  return lines.join('\n');
}

function yamlScalar(v) {
  if (v === null) return 'null';
  if (typeof v === 'boolean' || typeof v === 'number') return String(v);
  const s = String(v);
  if (/[:#\[\]{},&*!|>%@`"'\\]/.test(s) || /^\s|\s$/.test(s) || /^[-?]/.test(s)) {
    return '"' + s.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
  }
  return s;
}

async function* walk(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else yield full;
  }
}

async function copyDir(src, dst) {
  await fs.mkdir(dst, { recursive: true });
  for (const entry of await fs.readdir(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dst, entry.name);
    if (entry.isDirectory()) await copyDir(s, d);
    else await fs.copyFile(s, d);
  }
}

function detectLocale(fileName, meta) {
  const stem = fileName.replace(/\.mdx?$/, '');
  if (stem.endsWith('.en')) return { locale: 'en', baseStem: stem.slice(0, -3) };
  if (stem.endsWith('.zh-cn')) return { locale: 'zh-cn', baseStem: stem.slice(0, -6) };
  if (typeof meta.languageCode === 'string') {
    if (meta.languageCode.toLowerCase().startsWith('en')) return { locale: 'en', baseStem: stem };
  }
  return { locale: 'zh-cn', baseStem: stem };
}

/* --------------------------------- run -------------------------------- */
const stats = { posts: 0, pages: 0, copied: 0 };

await fs.mkdir(DST, { recursive: true });
await fs.mkdir(PAGES_DST, { recursive: true });

for await (const file of walk(SRC)) {
  if (!file.match(/\.mdx?$/)) continue;
  const rel = path.relative(SRC, file); // e.g. Algorithm Design/master-method.md
  const dir = path.dirname(rel);
  const fileName = path.basename(rel);

  const raw = await fs.readFile(file, 'utf8');
  const { meta, body } = parseFrontMatter(raw);
  const { locale, baseStem } = detectLocale(fileName, meta);

  // Transform frontmatter
  const fm = { ...meta };
  // Drop hugo-only keys we don't use.
  delete fm.layout;
  delete fm.summary;
  // Normalise.
  if (fm.tags && !Array.isArray(fm.tags)) fm.tags = [fm.tags];
  if (fm.categories && !Array.isArray(fm.categories)) fm.categories = [fm.categories];
  if (Array.isArray(fm.tags)) fm.tags = fm.tags.filter((t) => t && String(t).trim());
  fm.title = fm.title ?? baseStem;

  // Output path
  const categoryDir = dir;
  const slug = slugifyFileName(baseStem);
  const outDir = path.join(DST, locale, categoryDir);
  const outPath = path.join(outDir, `${slug}.md`);
  await fs.mkdir(outDir, { recursive: true });

  const outContent = stringifyYaml(fm) + body;
  await fs.writeFile(outPath, outContent, 'utf8');
  stats.posts++;
}

/* pages: about */
async function maybeCopy(src, dst) {
  try {
    await fs.copyFile(src, dst);
    stats.pages++;
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
}

const aboutZh = path.join(HUGO_ROOT, 'content', 'about', '_index.zh-cn.md');
const aboutEn = path.join(HUGO_ROOT, 'content', 'about', '_index.en.md');
try {
  const srcZh = await fs.readFile(aboutZh, 'utf8');
  const { meta, body } = parseFrontMatter(srcZh);
  meta.title = meta.title ?? '关于';
  await fs.writeFile(
    path.join(PAGES_DST, 'about.zh-cn.md'),
    stringifyYaml(meta) + body,
    'utf8'
  );
  stats.pages++;
} catch {}
try {
  const srcEn = await fs.readFile(aboutEn, 'utf8');
  const { meta, body } = parseFrontMatter(srcEn);
  meta.title = meta.title ?? 'About';
  await fs.writeFile(
    path.join(PAGES_DST, 'about.en.md'),
    stringifyYaml(meta) + body,
    'utf8'
  );
  stats.pages++;
} catch {}

/* static assets */
try {
  await copyDir(STATIC_SRC, PUBLIC_DST);
  stats.copied++;
} catch (e) {
  console.warn('[warn] copy static failed:', e.message);
}

console.log(`[migrate] wrote ${stats.posts} posts, ${stats.pages} pages; static copied: ${stats.copied ? 'yes' : 'no'}.`);
