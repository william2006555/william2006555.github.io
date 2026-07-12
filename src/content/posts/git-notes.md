---
title: Git 中我总是记混的三个概念
description: 工作区、暂存区和提交历史之间到底是什么关系。
date: 2026-06-25
course: 开发工具
courseCode: TOOL
tags: [Git, 速查]
kind: note
---

## 三个位置

工作区是正在编辑的文件，暂存区是下一次提交准备包含的内容，提交历史则是已经保存的版本。

```bash
git status
git add path/to/file
git commit -m "Explain the change"
```

关键不是背命令，而是每次先问：这个改动现在在哪一层？
