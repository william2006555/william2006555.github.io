# William's Fieldnotes

一个以课程和长期学习为中心的 Astro 博客。

## 日常写作

在 `src/content/posts/` 新建 Markdown 文件：

```md
---
title: 文章标题
description: 一句话摘要
date: 2026-07-12
course: 课程名称
courseCode: CS-01
tags: [标签一, 标签二]
kind: article
draft: false
---

从这里开始写正文。
```

`kind` 可以是 `article` 或 `note`。设置 `draft: true` 后不会公开生成。

## 修改个人信息

编辑 `src/site.config.ts`。站点名称、简介、邮箱和 GitHub 地址都集中在那里。

## 本地运行

```bash
pnpm install
pnpm dev
```

构建静态网站：

```bash
pnpm build
```

项目使用 `.node-version` 固定 Node.js 24.18.0。进入目录时，fnm 会自动切换 Node 版本。

推送到 GitHub 的 `main` 分支后，`.github/workflows/deploy.yml` 会自动构建并部署到 GitHub Pages。
