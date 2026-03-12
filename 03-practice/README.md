# 🛠️ 实践篇：训练自己的模型

本章节使用成熟的深度学习框架（PyTorch + Transformers），训练实用的中文 GPT 模型。

## 📚 章节导航

### [chinese-gpt](chinese-gpt/) - 中文小说 GPT
**核心内容：**
- 训练自定义 BPE 分词器
- 构建中文 GPT 模型
- 完整训练流程

**你将实现：**
1. **数据预处理** - 清洗和分割中文文本
2. **分词器训练** - 使用 tokenizers 训练 BPE
3. **模型构建** - 使用 transformers 构建 GPT2
4. **训练流程** - 训练、验证、保存
5. **文本生成** - 生成小说风格文本

## 🎯 学习目标

完成本章节后，你将：
1. ✅ 掌握使用 Transformers 库构建模型
2. ✅ 能够训练自定义分词器
3. ✅ 理解完整的模型训练流程
4. ✅ 能够训练自己的生成式模型

## 📋 前置知识

- Python 基础
- PyTorch 基础
- 完成 [01-basics](../01-basics/) 或具备神经网络基础

## 🔧 环境准备

```bash
pip install torch transformers tokenizers tqdm
```

## 💡 与进阶篇的区别

| 对比项 | 02-llm-from-scratch | 03-practice |
|--------|---------------------|-------------|
| 目标 | 理解原理 | 工程实践 |
| 框架 | 纯 PyTorch | Transformers |
| 代码量 | 多（手写） | 少（调用 API）|
| 可复用性 | 低 | 高 |
| 适合 | 学习原理 | 实际项目 |

## 📝 学习建议

- 先跑通代码，再理解细节
- 尝试用自己的数据训练
- 调整超参数，观察效果
- 记录实验结果，形成经验

---

**Next → [chinese-gpt](chinese-gpt/)**
