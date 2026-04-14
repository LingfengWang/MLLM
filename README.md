# MLLM 研究目录

多模态大模型 (Multimodal Large Language Models) 调研与追踪

## 目录结构

```
MLLM/
├── 01-papers/          # 学术论文
│   ├── architecture/   # 模型架构 (ViT, CLIP, Flamingo, LLaVA 等)
│   ├── training/       # 训练方法 (对比学习，指令微调，RLHF 等)
│   ├── applications/   # 应用场景 (VQA, 图像描述，视觉推理等)
│   └── surveys/        # 综述论文
│
├── 02-models/          # 模型信息
│   ├── open-source/    # 开源模型 (LLaVA, BLIP, InstructBLIP 等)
│   └── proprietary/    # 闭源模型 (GPT-4V, Gemini, Claude 等)
│
├── 03-datasets/        # 数据集
│
├── 04-code/            # 代码实现与实验
│
├── 05-notes/           # 笔记
│   ├── reading-notes/  # 论文阅读笔记
│   ├── meeting-notes/  # 讨论记录
│   └── ideas/          # 研究想法
│
└── 06-resources/       # 资源链接、博客、教程等
```

## 命名规范

- 论文文件：`YYYY-MM-DD_标题关键词.pdf`
- 笔记文件：`YYYY-MM-DD_主题.md`
- 模型信息：`模型名称.md`

## 快速开始

1. 阅读 `01-papers/surveys/` 中的综述建立整体认知
2. 在 `05-notes/reading-notes/` 记录论文笔记
3. 在 `05-notes/ideas/` 记录研究灵感

---
创建时间：2026-04-07
