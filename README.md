# Machine-Learning-Beginner

一个从零开始手写代码学习机器学习、神经网络和大语言模型的项目。

## 🎯 项目简介

做为一个传统的程序员，在使用AI并开始学习机器学习和神经网络相关的内容后，一直存在一些疑惑，比如：

1. 机器学习编程和传统编程的核心区别是什么？有没有最简单直观的代码能让我感受感受？
2. 机器到底是怎么学习的？有没有最简单直观的代码能让我感受感受？
3. 为什么说大模型是一个函数？这个函数长什么样的？
4. 为什么要学习率，它到底起个什么作用？为什么学习率小了收敛速度慢，大了又可能震荡或发散？
5. 反向传播是怎么传播的？梯度下降又是什么鬼？
6. 为什么没有激活函数的神经网络只能表达线性关系？激活函数是怎么做非线性变换的？
7. 神经元和隐藏层到底都起的什么作用？
8. ……

是的，我就想看有没有代码让我能直观的理解这些概念。本项目完全从零开始撸代码，从 `y=wx` 开始入门，到手撸 Llama3 大模型，再到训练自己的中文 GPT，循序渐进地理解机器学习的本质。

---

## 📚 学习路径

本项目按照 **基础 → 进阶 → 实践** 的路径组织：

### 🌱 01-basics - 基础篇：机器学习入门
完全从零手写代码，不使用机器学习框架，直观理解核心概念。

| 章节 | 内容 | 文件 |
|------|------|------|
| [01-linear-model](01-basics/01-linear-model/) | 从 `y=wx` 开始了解机器是怎么学习的 | `tutorial.ipynb` |
| [02-gradient-descent](01-basics/02-gradient-descent/) | 从均方误差感受梯度下降的具体实现 | `tutorial.ipynb`, `tutorial.py` |
| [03-activation](01-basics/03-activation/) | 用激活函数感受非线性变换的效果 | `tutorial.ipynb`, `tutorial.py` |
| [04-neural-network](01-basics/04-neural-network/) | 手撸神经网络感受深度学习 | `tutorial.ipynb` |

**你将学到：**
- 机器学习与传统编程的核心区别
- 参数、特征、标签、超参数等概念
- 学习率、梯度下降、反向传播的原理
- 激活函数的非线性变换
- 神经网络的前向传播和反向传播

---

### 🚀 02-llm-from-scratch - 进阶篇：从零实现大模型
不依赖 PyTorch/TensorFlow 的高级封装，逐行手写实现大语言模型。

> 📌 **致谢**：本章节 Llama3 教程基于 [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) 改编，感谢原作者的精彩实现！

| 章节 | 内容 | 文件 |
|------|------|------|
| [fundamentals](02-llm-from-scratch/fundamentals/) | **LLM 基础教程** - 从文本处理到 RLHF 的完整学习路径（9章） | 多章节 |
| [llama3](02-llm-from-scratch/llama3/) | **Llama3 实现** - 从零手写 Llama3-8B 大语言模型 | `tutorial.ipynb` |

**你将学到：**
- 大语言模型的内部架构
- Tokenizer（BPE 分词器）
- Embedding 层
- RMSNorm 归一化
- RoPE 旋转位置编码
- Grouped Query Attention
- SwiGLU 前馈网络
- KV-Cache 推理优化

---

### 🛠️ 03-practice - 实践篇：训练自己的模型
使用成熟框架（PyTorch + Transformers）训练实用的中文 GPT。

| 章节 | 内容 | 文件 |
|------|------|------|
| [chinese-gpt](03-practice/chinese-gpt/) | 训练中文小说 GPT | `train.py` |

**你将学到：**
- 使用 Transformers 库构建 GPT
- 训练自定义 BPE 分词器
- 数据预处理和 Dataset 构建
- 模型训练和验证
- 断点续训和早停机制
- 文本生成

---

### 🧪 tests - 实验与测试
自由实验和测试代码的 playground。

---

### 🔮 04-future - 未来扩展
预留目录，计划补充：
- 模型微调（Fine-tuning）
- 模型量化（Quantization）
- 模型部署（Deployment）
- 更多实战项目

---

## 🚀 快速开始

### 基础篇
```bash
# 进入第一章
jupyter notebook 01-basics/01-linear-model/tutorial.ipynb
```

### 进阶篇
```bash
# 需要先下载 Llama3 权重
jupyter notebook 02-llm-from-scratch/llama3/tutorial.ipynb
```

### 实践篇
```bash
# 训练中文 GPT
python 03-practice/chinese-gpt/train.py -d ../../data/小说.txt
```

---

## 📖 推荐学习顺序

1. **完全零基础**：从 `01-basics/01-linear-model/` 开始，按顺序学习
2. **有基础想深入 LLM**：直接跳到 `02-llm-from-scratch/llama3/`
3. **想快速上手项目**：从 `03-practice/chinese-gpt/` 开始

---

## 💡 学习建议

- 本教程过于基础，相当于在教你数学的四则运算
- 建议配合大模型（如 ChatGPT）学习并扩展知识
- 每个章节都包含详细的代码和图解，建议边运行边理解
- 不要只看，一定要动手跑代码！

---

## 📂 项目结构

```
Machine-Learning-Beginner/
├── README.md                    # 本文件
├── LICENSE
├── .gitignore
│
├── 01-basics/                   # 🌱 基础篇
│   ├── 01-linear-model/
│   ├── 02-gradient-descent/
│   ├── 03-activation/
│   └── 04-neural-network/
│
├── 02-llm-from-scratch/         # 🚀 进阶篇
│   ├── fundamentals/            # 通用 LLM 基础教程（9章）
│   ├── llama3-step-by-step/     # Llama3 分解式教学（10课）⭐
│   └── llama3/                  # Llama3 完整实现（notebook）
│
├── 03-practice/                 # 🛠️ 实践篇
│   ├── chinese-gpt/             # 中文小说 GPT
│   └── chinese-gpt-fundamentals.md  # 基础教程项目链接
│
├── 04-future/                   # 🔮 未来扩展
│
└── tests/                       # 🧪 测试
```

---

## 🙏 致谢

- [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) - 本项目的 Llama3 教程改编自该仓库，感谢原作者 Naklecha 的精彩实现和分享！

---

## 📝 License

MIT License

---

**Happy Learning! 🎉**
