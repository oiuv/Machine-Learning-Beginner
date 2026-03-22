# 中文小说 GPT - 基础教程版本

本项目位于 `02-llm-from-scratch/fundamentals/projects/guimi-gpt/`

## 快速链接

- 📚 [基础教程目录](../../02-llm-from-scratch/fundamentals/)
- 🎯 [项目代码](../../02-llm-from-scratch/fundamentals/projects/guimi-gpt/)
- 📖 [学习笔记](../../02-llm-from-scratch/fundamentals/README.md)

## 项目概述

这是《从零训练大模型》教程的实践项目，使用《诡秘之主》和《宿命之环》训练中文 GPT 模型。

### 配置参数
- 模型参数量：123M
- Transformer层数：12
- 嵌入维度：768
- 词表大小：50000
- 上下文长度：512

### 训练结果
- 最佳验证损失：8.6741
- 训练数据：890万字符

## 使用方法

```bash
# 进入项目目录
cd ../../02-llm-from-scratch/fundamentals/projects/guimi-gpt/

# 训练
python train.py -d ../../data/ -lr 3e-4

# 生成
python generate.py
```

---

**注意**：此项目为教程学习版本，完整功能版本请参见 [chinese-gpt/](./chinese-gpt/) 目录。
