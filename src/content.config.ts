import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const postSchema = z
  .object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    draft: z.boolean().optional().default(false),
    math: z.boolean().optional().default(false),
    categories: z
      .union([z.string(), z.array(z.string())])
      .optional()
      .transform((v) =>
        v === undefined ? [] : Array.isArray(v) ? v : [v]
      ),
    tags: z
      .union([z.string(), z.array(z.string())])
      .optional()
      .transform((v) =>
        v === undefined ? [] : Array.isArray(v) ? v : [v]
      )
      .transform((list) => list.filter((t) => t && t.trim().length > 0)),
    series: z.string().optional(),
    note: z.string().optional(),
    cover: z.string().optional(),
    author: z.string().optional(),
    license: z
      .object({
        name: z.string().optional(),
        url: z.string().optional(),
      })
      .optional(),
  })
  .passthrough();

const posts = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
  schema: postSchema,
});

const pages = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/pages' }),
  schema: z
    .object({
      title: z.string(),
      description: z.string().optional(),
      date: z.coerce.date().optional(),
      layout: z.string().optional(),
    })
    .passthrough(),
});

export const collections = { posts, pages };
export type PostSchema = z.infer<typeof postSchema>;
