import getReadingTime from 'reading-time';
import { toString } from 'mdast-util-to-string';

export function remarkReadingTime() {
  return function (tree, { data }) {
    const textOnPage = toString(tree);
    const readingTime = getReadingTime(textOnPage);
    data.astro.frontmatter.readingTime = Math.max(1, Math.round(readingTime.minutes));
    data.astro.frontmatter.wordCount = textOnPage.replace(/\s+/g, '').length;
  };
}
