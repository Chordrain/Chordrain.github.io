// Convert Hugo-style math delimiters \( ... \) and \[ ... \] into $ ... $
// and $$ ... $$ respectively, so remark-math can parse them.
// Only runs on text nodes (so fenced code / inline code stays intact).
import { visit, SKIP } from 'unist-util-visit';

export function remarkBackslashMath() {
  return (tree) => {
    visit(tree, 'text', (node, index, parent) => {
      if (!parent || typeof index !== 'number') return;
      if (!node.value || !/\\[(\[]/.test(node.value)) return;

      const text = node.value;
      const nodes = [];
      let i = 0;

      while (i < text.length) {
        const openBlock = text.indexOf('\\[', i);
        const openInline = text.indexOf('\\(', i);

        // Pick the earliest opener.
        let openPos = -1;
        let block = false;
        if (openBlock >= 0 && (openInline < 0 || openBlock < openInline)) {
          openPos = openBlock;
          block = true;
        } else if (openInline >= 0) {
          openPos = openInline;
          block = false;
        }

        if (openPos < 0) {
          nodes.push({ type: 'text', value: text.slice(i) });
          break;
        }

        if (openPos > i) {
          nodes.push({ type: 'text', value: text.slice(i, openPos) });
        }

        const closer = block ? '\\]' : '\\)';
        const closePos = text.indexOf(closer, openPos + 2);
        if (closePos < 0) {
          // Unbalanced, emit rest as-is.
          nodes.push({ type: 'text', value: text.slice(openPos) });
          break;
        }

        const math = text.slice(openPos + 2, closePos);
        nodes.push(
          block
            ? { type: 'math', value: math }
            : { type: 'inlineMath', value: math }
        );
        i = closePos + 2;
      }

      if (nodes.length === 0) return;
      parent.children.splice(index, 1, ...nodes);
      return [SKIP, index + nodes.length];
    });
  };
}
