import { defineCollection, z } from 'astro:content';

const pages = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    mode: z.enum(['laboratory', 'illuminated', 'blended']).default('blended'),
    // page-specific fields (optional)
    collaboration: z.string().optional(),
    order: z.number().optional(),
    demoIntro: z.string().optional(),
  }),
});

export const collections = { pages };
