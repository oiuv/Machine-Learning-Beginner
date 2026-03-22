# 🚀 进阶篇：从零实现大模型

本章节不依赖 PyTorch/TensorFlow 的高级封装，逐行手写实现大语言模型，深入理解 LLM 的内部机制。

> 📌 **致谢**：本章节 Llama3 教程基于 [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) 改编，感谢原作者的精彩实现！

## 📚 章节导航

### [fundamentals](fundamentals/) - LLM 基础教程 ⭐
**核心内容：**
- 完整的 LLM 学习路径（9个章节）
- 从文本处理到 RLHF 的完整流程
- 包含实践项目：垃圾短信分类、中文小说 GPT

**适合人群：**
- 想系统学习 LLM 的初学者
- 希望有完整代码和笔记的学习者

**章节内容：**
1. **概述** - LLM 基本概念
2. **文本数据** - Tokenization、Embedding
3. **注意力机制** - Self-Attention、Multi-Head Attention
4. **GPT 模型** - 完整架构实现
5. **预训练** - 训练流程和技巧
6. **微调** - 分类任务微调
7. **指令微调** - 对话能力训练
8. **RLHF** - 人类反馈强化学习
9. **MoE** - 混合专家模型

---

### [llama3-step-by-step](llama3-step-by-step/) - Llama3 分解式教学 ⭐
**核心内容：**
- 10 个渐进式课程，从零实现 Llama3
- 每个组件独立讲解，配有详细图解
- 适合零基础逐步深入理解

**课程安排：**
1. **Lesson 1** - Tokenizer（文本分词）
2. **Lesson 2** - Embeddings（词嵌入）
3. **Lesson 3** - Attention Basics（注意力基础）
4. **Lesson 4** - RoPE（旋转位置编码）
5. **Lesson 5** - Multi-Head Attention（多头注意力）
6. **Lesson 6** - SwiGLU（激活函数与前馈网络）
7. **Lesson 7** - Norm & Residual（归一化与残差连接）
8. **Lesson 8** - Transformer Layer（完整 Transformer 层）
9. **Lesson 9** - Multi-Layer Output（多层输出）
10. **Lesson 10** - Summary（整合与总结）

**特点：**
- 每个 lesson 可独立运行
- 大量可视化图解
- 详细中文注释和比喻教学

---

### [llama3](llama3/) - 从零手写 Llama3-8B（完整版）
**核心内容：**
- 单个 notebook 完整实现 Llama3
- 加载 Meta 官方 Llama3 权重
- 实现完整的推理流程

**你将实现：**
1. **Tokenizer** - BPE 分词器
2. **Embedding** - 词嵌入层
3. **RMSNorm** - 均方根层归一化
4. **RoPE** - 旋转位置编码
5. **GQA** - 分组查询注意力
6. **SwiGLU** - SwiGLU 前馈网络
7. **Transformer Block** - 完整的 Transformer 层
8. **文本生成** - 自回归生成

**与 step-by-step 的区别：**
- `llama3/`：完整实现，适合快速概览整体架构
- `llama3-step-by-step/`：分解教学，适合逐步深入每个组件

## 🎯 学习目标

完成本章节后，你将：
1. ✅ 深入理解大语言模型的每个组件
2. ✅ 掌握位置编码、注意力机制等核心技术
3. ✅ 能够阅读和修改开源 LLM 代码
4. ✅ 为模型微调、优化打下坚实基础

## 📋 前置知识

- Python 基础
- PyTorch 基础张量操作
- 完成 [01-basics](../01-basics/) 或具备同等知识

## 🔧 环境准备

```bash
# 需要下载 Llama3 官方权重
# 访问 https://llama.meta.com/llama-downloads/ 申请

# 安装依赖
pip install torch tiktoken matplotlib
```

## 💡 学习建议

- 不要急于运行，先理解每一行代码
- 配合图解（images/ 目录）理解数据流
- 尝试修改参数，观察输出变化
- 对比官方实现，理解设计选择

## 📝 代码特点

- **纯 PyTorch**：不使用 transformers 库的高级封装
- **逐行注释**：每个操作都有详细说明
- **可视化**：大量图解帮助理解
- **可运行**：可以直接加载官方权重进行推理

---

**Next → [llama3](llama3/)**
