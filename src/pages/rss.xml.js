import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { site } from '../site.config';

export async function GET(context) {
  const posts = (await getCollection('posts', ({ data }) => !data.draft)).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: site.title,
    description: site.description,
    site: context.site,
    items: posts.map((post) => ({ title: post.data.title, description: post.data.description, pubDate: post.data.date, link: `/blog/${post.id}` })),
  });
}
