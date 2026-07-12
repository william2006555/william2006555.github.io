import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    course: z.string().optional(),
    courseCode: z.string().optional(),
    tags: z.array(z.string()).default([]),
    kind: z.enum(['article', 'note']).default('article'),
    draft: z.boolean().default(false),
  }),
});

export const collections = { posts };
