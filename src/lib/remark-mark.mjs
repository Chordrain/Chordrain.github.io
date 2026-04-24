// Converts ==text== inside paragraphs to <mark>text</mark>.
// Mirrors the Hugo `mark` extra-extension, so authors can keep using ==...==.
import { visit } from 'unist-util-visit';

const MARK_RE = /==([^=]+)==/g;

export function remarkMarkHighlight() {
  return (tree) => {
    visit(tree, 'text', (node, index, parent) => {
      if (!parent || typeof index !== 'number') return;
      const value = node.value;
      if (!value || !value.includes('==')) return;

      const nodes = [];
      let lastIndex = 0;
      let match;
      MARK_RE.lastIndex = 0;
      while ((match = MARK_RE.exec(value)) !== null) {
        if (match.index > lastIndex) {
          nodes.push({ type: 'text', value: value.slice(lastIndex, match.index) });
        }
        nodes.push({
          type: 'html',
          value: `<mark class="prose-mark">${escapeHtml(match[1])}</mark>`,
        });
        lastIndex = match.index + match[0].length;
      }
      if (nodes.length === 0) return;
      if (lastIndex < value.length) {
        nodes.push({ type: 'text', value: value.slice(lastIndex) });
      }
      parent.children.splice(index, 1, ...nodes);
      return index + nodes.length;
    });
  };
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
