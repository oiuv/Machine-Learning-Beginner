# 从零训练大模型 - 学习笔记

## 环境信息
- Python: 3.14.3
- PyTorch: 2.10.0+cu130
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- CUDA: 13.0

## 学习目录结构
```
learning/
├── ch01-overview/              # 教程概述 🆕
├── ch02-text-data/              # 第2章：文本数据处理
├── ch03-attention/              # 第3章：注意力机制
├── ch04-gpt-model/              # 第4章：GPT模型架构
├── ch05-pretraining/            # 第5章：预训练
├── ch06-finetuning/             # 第6章：微调
├── ch07-instruction-finetuning/ # 第7章：指令微调
├── ch08-rlhf/                   # 第8章：RLHF（从人类反馈中学习）
├── ch09-moe/                    # 第9章：MoE架构 🆕
├── projects/                    # 实践项目
│   ├── spam-classifier/         # 垃圾短信分类器
│   └── guimi-gpt/               # 诡秘之主中文GPT模型 ⭐
├── data/                        # 训练数据
│   ├── 诡秘之主.txt
│   └── 宿命之环.txt
└── notes/                       # 学习笔记
```

## 学习进度

| 章节 | 主题 | 状态 | 关键文件 |
|------|------|------|----------|
| Ch01 | 教程概述 | ✅ 已完成 | ch01-overview/ |
| Ch02 | 文本数据处理 | ✅ 已完成 | ch02-text-data/ |
| Ch03 | 注意力机制 | ✅ 已完成 | ch03-attention/ |
| Ch04 | GPT模型架构 | ✅ 已完成 | ch04-gpt-model/ |
| Ch05 | 预训练 | ✅ 已完成 | ch05-pretraining/ |
| Ch06 | 微调 | 📖 学习中 | ch06-finetuning/ |
| Ch07 | 指令微调 | 📖 学习中 | ch07-instruction-finetuning/ |
| Ch08 | RLHF | 📖 学习中 | ch08-rlhf/ |
| Ch09 | **MoE架构** | 🆕 新增 | **ch09-moe/** |
| **项目** | **诡秘之主GPT** | **✅ 已完成** | **projects/guimi-gpt/** |

## LLM 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM 训练完整流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 文本数据 (Ch02)                                          │
│     原始文本 → Tokenization → Token IDs → Embedding          │
│                                                             │
│  2. 注意力机制 (Ch03)                                        │
│     Query, Key, Value → Attention Scores → Context Vector   │
│                                                             │
│  3. GPT架构 (Ch04)                                          │
│     Embedding → Transformer Blocks → Output Layer           │
│                                                             │
│  4. 预训练 (Ch05)                                           │
│     Forward → Loss → Backward → Optimizer Update            │
│                                                             │
│  5. 微调 (Ch06)                                             │
│     加载预训练权重 → 修改输出层 → 分类训练                    │
│                                                             │
│  6. 指令微调 (Ch07)                                          │
│     指令数据 → Prompt Template → 多任务训练                  │
│                                                             │
│  7. RLHF (Ch08)                                              │
│     人类比较数据 → 奖励模型 → PPO训练 → 对齐模型              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心概念速查

### Tokenization (分词)
将文本转换为数字序列，让模型能够处理。

### Embedding (嵌入)
将离散的token ID转换为连续的向量表示。

### Attention (注意力)
计算序列中各位置之间的关联权重。

### Transformer Block
包含 Multi-Head Attention + Feed Forward Network + 残差连接。

### Pretraining (预训练)
在大规模文本上训练模型预测下一个token。

### Fine-tuning (微调)
用预训练模型 + 少量有标签数据，完成特定任务。

---

## 已完成的学习

### 第2章：文本数据处理
- BPE 分词器
- Token ID → Embedding
- 滑动窗口生成训练样本

### 第3章：注意力机制
- Self-Attention: Q/K/V 投影
- Causal Mask: 防止看到未来
- Multi-Head: 多个头学不同关系

### 第4章：GPT 模型架构
- LayerNorm: 归一化激活
- GELU: 平滑激活函数
- TransformerBlock: Attention + FFN + 残差
- GPTModel: 完整 124M 参数模型

### 第5章：预训练
- 损失函数: CrossEntropy
- 训练循环: 清梯度 → 算损失 → 反向 → 更新
- 优化器: AdamW
- 实际训练了模型，看到损失下降

---

## 实践项目：诡秘之主 GPT ⭐

### 项目概述
从零训练一个中文小说 GPT 模型，使用《诡秘之主》和《宿命之环》作为训练数据。

### 最终配置
| 参数 | 值 |
|------|-----|
| 模型参数量 | 123M |
| Transformer层数 | 12 |
| 嵌入维度 | 768 |
| 注意力头数 | 12 |
| 词表大小 | 50000 |
| 上下文长度 | 512 |
| 批次大小 | 8 |
| 学习率 | 3e-4 |

### 训练数据
| 文件 | 字符数 |
|------|--------|
| 诡秘之主.txt | 4,859,151 |
| 宿命之环.txt | 4,045,421 |
| **总计** | **8,904,574** |

### 训练结果
| 指标 | 值 |
|------|-----|
| 最佳验证损失 | 8.6741 |
| 训练轮数 | 10轮（早停） |
| 每轮耗时 | ~4分钟 |

### 项目文件
```
projects/guimi-gpt/
├── train.py      # 训练脚本（支持多文件、断点续训、早停）
├── generate.py   # 生成脚本
├── output/       # 输出目录
│   └── model/    # 训练好的模型
└── README.md     # 项目说明
```

### 实现功能
- ✅ BPE 中文分词器训练
- ✅ 多文件数据加载（支持目录输入）
- ✅ 断点续训（checkpoint保存/加载）
- ✅ 早停机制（防止过拟合）
- ✅ 混合精度训练（可选）
- ✅ 文本生成

### 使用方法
```bash
# 训练
python train.py -d ../../data/ -lr 3e-4

# 生成
python generate.py
```

---

## 关键洞察

### 理论层面
1. **Q/K/V 不是指定的，是训练学会的**：三个矩阵通过损失函数反向传播自然分化
2. **过拟合现象**：训练损失下降，验证损失先降后升
3. **数据量很重要**：小数据 + 大模型 = 死记硬背

### 实践层面
4. **模型容量匹配数据量**：
   - 123M 参数 + 890万字符 = 效果好
   - 253M 参数 + 890万字符 = 效果差（数据不够）
   
5. **学习率影响**：
   - 5e-4：训练快，可能不稳定
   - 3e-4：更稳定，效果更好

6. **数据量提升效果显著**：
   | 数据量 | 最佳Loss |
   |--------|----------|
   | 480万字符 | 9.23 |
   | 890万字符 | 8.67 |

7. **早停机制必要**：防止过拟合，节省训练时间

8. **断点续训实用**：中断后可继续训练，不丢失进度

---

## 后续学习计划

- [ ] Ch06：微调（分类任务）
- [ ] Ch07：指令微调（对话能力）
- [ ] Ch08：RLHF（对齐人类偏好）
- [ ] 更大模型训练（1B+参数）
- [ ] 预训练模型微调（Qwen/ChatGLM）
