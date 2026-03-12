# 🚀 进阶篇：从零实现大模型

本章节不依赖 PyTorch/TensorFlow 的高级封装，逐行手写实现大语言模型，深入理解 LLM 的内部机制。

> 📌 **致谢**：本章节 Llama3 教程基于 [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) 改编，感谢原作者的精彩实现！

## 📚 章节导航

### [llama3](llama3/) - 从零手写 Llama3-8B
**核心内容：**
- 加载 Meta 官方 Llama3 权重
- 实现完整的推理流程
- 逐行理解每个组件

**你将实现：**
1. **Tokenizer** - BPE 分词器
2. **Embedding** - 词嵌入层
3. **RMSNorm** - 均方根层归一化
4. **RoPE** - 旋转位置编码
5. **GQA** - 分组查询注意力
6. **SwiGLU** - SwiGLU 前馈网络
7. **Transformer Block** - 完整的 Transformer 层
8. **文本生成** - 自回归生成

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
