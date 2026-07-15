export const primaryCategories = [
  {
    name: '日常记录',
    code: 'DAILY',
    description: '日记、周记、游记、阶段回顾，以及生活里值得留下来的片段。',
  },
  {
    name: '学习笔记',
    code: 'STUDY',
    description: '课程、教材、公开课和系统学习过程中的整理。',
  },
  {
    name: '研究札记',
    code: 'RESEARCH',
    description: '论文阅读、科研想法、方向观察和技术脉络梳理。',
  },
  {
    name: '项目实践',
    code: 'BUILD',
    description: '项目复盘、工程实践、工具搭建和踩坑记录。',
  },
] as const;

export const fallbackCategory = '日常记录';

export function getPostCategory(data: { category?: string; course?: string }) {
  return data.category ?? data.course ?? fallbackCategory;
}

export function getCategoryCode(data: { categoryCode?: string; courseCode?: string; category?: string; course?: string }) {
  const category = getPostCategory(data);
  return data.categoryCode ?? data.courseCode ?? primaryCategories.find((item) => item.name === category)?.code ?? 'NOTE';
}
