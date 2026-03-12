# Llama3 from Scratch

从零手写实现 Meta Llama3-8B 大语言模型，逐行理解每个组件的工作原理。

> 📌 **致谢**：本教程改编自 [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)，感谢原作者 Naklecha 的精彩实现和分享！
> 
> 原项目地址：https://github.com/naklecha/llama3-from-scratch

## 📖 教程内容

### 1. Tokenizer
- 使用 tiktoken 加载 BPE 分词器
- 理解子词切分（subword tokenization）
- 特殊 token 的处理

### 2. 模型架构
```
输入 → Embedding → [Transformer Block × 32] → RMSNorm → Linear → Softmax → 输出
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              RMSNorm + GQA      SwiGLU FFN
                    ↓                   ↓
                 RoPE 位置编码      门控激活
```

### 3. 核心组件详解

#### RMSNorm（均方根层归一化）
```python
# 相比 LayerNorm 更简单高效
output = x / sqrt(mean(x^2) + eps) * weight
```

#### RoPE（旋转位置编码）
- 相对位置编码的一种实现
- 通过旋转矩阵注入位置信息
- 支持任意长度的外推

#### GQA（Grouped Query Attention）
- 多头注意力的优化版本
- 多个查询共享相同的 Key/Value
- 减少内存占用，加速推理

#### SwiGLU（Swish-Gated Linear Unit）
```python
# 前馈网络的激活函数
output = (x @ W1 * silu(x @ W3)) @ W2
```

### 4. 推理流程
1. 文本 → Token IDs
2. Embedding 查找
3. 逐层 Transformer 计算
4. 输出 logits
5. Softmax 采样生成下一个 token
6. 重复直到生成结束符

## 🚀 快速开始

### 1. 下载权重
访问 https://llama.meta.com/llama-downloads/ 申请下载 Llama3-8B 权重。

### 2. 目录结构
```
llama3/
├── tutorial.ipynb          # 主教程
├── README.md               # 本文件
└── images/                 # 40+ 张图解
    ├── archi.png           # 整体架构
    ├── attention.png       # 注意力机制
    ├── rope.png            # 旋转位置编码
    └── ...
```

### 3. 运行
```bash
jupyter notebook tutorial.ipynb
```

## 📂 权重文件结构

```
Meta-Llama-3-8B/
├── consolidated.00.pth     # 模型权重
├── tokenizer.model         # 分词器
└── params.json             # 配置参数
```

## 🎯 学习重点

1. **理解维度变化**：跟踪每个张量的 shape
2. **掌握矩阵运算**：理解 einsum 和 matmul
3. **可视化数据流**：配合图解理解信息流动
4. **对比标准实现**：与 transformers 库的实现对比

## 💡 进阶探索

1. 实现 KV-Cache 加速推理
2. 添加温度采样和 Top-p 采样
3. 实现批量推理
4. 量化模型（INT8/INT4）
5. 微调模型（LoRA）

## 📚 参考资料

- [Llama3 论文](https://arxiv.org/abs/2407.21783)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [GQA 论文](https://arxiv.org/abs/2305.13245)
- [SwiGLU 论文](https://arxiv.org/abs/2002.05202)

---

**Previous ← [01-basics](../../01-basics/)** | **Next → [03-practice](../../03-practice/)**
