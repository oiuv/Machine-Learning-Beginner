"""
阶段 10: 整合与总结
回顾整个 Llama3 架构
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 10: 整合与总结")
print("=" * 70)

# ============================================
# 第一部分：完整架构回顾
# ============================================

print("\n" + "=" * 70)
print("第一部分：Llama3 完整架构回顾")
print("=" * 70)

print("""
【Llama3-8B 完整架构】

┌─────────────────────────────────────────────────────────────────┐
│                         Llama3-8B                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【输入层】                                                       │
│   Token IDs: [batch, seq_len]                                    │
│     ↓                                                            │
│   Embedding: [128256, 4096] → [batch, seq_len, 4096]             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【Transformer 层 × 32】 (逐层抽象)                               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 1: 局部特征 (词级别)                               │    │
│  │   RMS Norm → Attention → 残差 → RMS Norm → FFN → 残差   │    │
│  └─────────────────────────────────────────────────────────┘    │
│     ↓                                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 2-10: 短语特征                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│     ↓                                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 11-20: 句子结构                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│     ↓                                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 21-31: 语义理解                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│     ↓                                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 32: 推理与生成                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【输出层】                                                       │
│   Linear: [4096, 128256]                                         │
│     ↓                                                            │
│   Softmax → 概率分布                                             │
│     ↓                                                            │
│   采样 → 下一个 Token                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

【关键参数】

词汇表大小: 128,256
嵌入维度: 4,096
注意力头数: 32 (Query) / 8 (Key/Value)
层数: 32
FFN 隐藏维度: 11,008
总参数量: ~8B (80亿)
""")

# ============================================
# 第二部分：组件总结
# ============================================

print("\n" + "=" * 70)
print("第二部分：组件总结")
print("=" * 70)

components = """
【1. Tokenization (阶段1)】
   作用: 文本 → Token IDs
   方法: BPE (Byte Pair Encoding)
   词汇表: 128,256
   关键概念: 子词、特殊Token(<START>, <END>)

【2. Embedding (阶段2)】
   作用: Token ID → 向量
   形状: [vocab_size, dim]
   意义: 语义空间中的表示
   训练: 预训练学习

【3. Attention (阶段3)】
   作用: 学习词间关系
   核心: Query, Key, Value
   计算: Softmax(Q @ K^T / √d) @ V
   意义: "关注"其他词

【4. RoPE (阶段4)】
   作用: 添加位置信息
   方法: 旋转向量
   公式: 旋转角度 = 位置 × 频率
   优点: 相对位置编码，可外推

【5. Multi-Head Attention (阶段5)】
   作用: 多视角观察
   头数: 32
   每头维度: 128
   GQA: Query多，KV少，节省计算

【6. SwiGLU FFN (阶段6)】
   作用: 非线性变换
   结构: (silu(x @ w1) * (x @ w3)) @ w2
   隐藏维度: 11,008
   意义: 增加表达能力

【7. RMS Norm & Residual (阶段7)】
   RMS Norm: x / sqrt(mean(x^2) + eps) * gamma
   残差连接: y = x + F(x)
   位置: Pre-Norm (先归一化，再计算)
   作用: 训练稳定，梯度流通

【8. Transformer Layer (阶段8)】
   结构: Attention子层 + FFN子层
   每个子层: Norm → 计算 → 残差
   输入输出: 形状相同 [seq_len, dim]

【9. Multi-Layer & Output (阶段9)】
   层数: 32
   抽象: 逐层加深
   输出: Linear → Softmax → 概率
   生成: 采样策略(Greedy, Top-k, Temperature)
"""

print(components)

# ============================================
# 第三部分：数据流形状变化
# ============================================

print("\n" + "=" * 70)
print("第三部分：数据流形状变化")
print("=" * 70)

print("""
【形状变化全流程】

阶段              形状                          说明
─────────────────────────────────────────────────────────────
输入文本          "我爱学习"                    原始文本
  ↓
Tokenization      [1, 3, 4, 5, 6, 2]            Token IDs
  ↓
Embedding         [6, 4096]                     嵌入向量
  ↓
Layer 1           [6, 4096]                     局部特征
  ↓
Layer 2           [6, 4096]                     短语特征
  ↓
...               ...                           ...
  ↓
Layer 32          [6, 4096]                     语义推理
  ↓
Output Linear     [6, 128256]                   Logits
  ↓
Softmax           [6, 128256]                   概率分布
  ↓
采样              7                             下一个Token
  ↓
解码              "习"                          生成的词

【关键观察】

1. 序列长度不变: 始终 [seq_len, ...]
2. 维度不变: Transformer层保持 [..., dim]
3. 最后扩展: 输出层扩展到词汇表大小
4. 形状变化: [seq_len] → [seq_len, dim] → [seq_len, vocab]
""")

# 可视化形状变化
fig, ax = plt.subplots(figsize=(14, 8))

stages = ['Text', 'Tokens', 'Embed', 'Layer1', 'Layer16', 'Layer32', 'Logits', 'Probs', 'Token']
shapes = ['str', '6', '6×4096', '6×4096', '6×4096', '6×4096', '6×128K', '6×128K', '1']
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightcoral', 
          'lightcoral', 'lightpink', 'lightpink', 'lightgreen']

y_pos = range(len(stages))
bars = ax.barh(y_pos, [1]*len(stages), color=colors, edgecolor='black', linewidth=1.5)

for i, (stage, shape) in enumerate(zip(stages, shapes)):
    ax.text(0.5, i, f'{stage:10s} → [{shape}]', 
            ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(stages)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title('Llama3 数据流形状变化', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# 添加箭头
for i in range(len(stages) - 1):
    ax.annotate('', xy=(1.05, i+0.5), xytext=(1.05, i-0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

plt.tight_layout()
plt.savefig('shape_flow.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'shape_flow.png'")

# ============================================
# 第四部分：关键概念总结
# ============================================

print("\n" + "=" * 70)
print("第四部分：关键概念总结")
print("=" * 70)

concepts = """
【核心概念】

1. 自注意力 (Self-Attention)
   ✓ 每个词都能看到其他所有词
   ✓ 通过 Query-Key-Value 计算
   ✓ 学习词与词的关系

2. 位置编码 (RoPE)
   ✓ 让模型知道词的位置
   ✓ 通过旋转向量实现
   ✓ 编码相对位置

3. 多头注意力 (Multi-Head)
   ✓ 多个视角同时观察
   ✓ 每个头学习不同模式
   ✓ GQA 优化计算

4. 残差连接 (Residual)
   ✓ y = x + F(x)
   ✓ 解决梯度消失
   ✓ 训练深层网络

5. 归一化 (RMS Norm)
   ✓ 保持数值稳定
   ✓ Pre-Norm 结构
   ✓ 加速训练

6. 逐层抽象
   ✓ 浅层: 局部特征
   ✓ 中层: 结构特征
   ✓ 深层: 语义推理

【关键公式】

注意力:    Attention = Softmax(Q @ K^T / √d) @ V
RoPE:      旋转角度 = 位置 × 频率
SwiGLU:    (silu(x @ w1) * (x @ w3)) @ w2
RMS Norm:  x / sqrt(mean(x^2) + eps) * gamma
残差:      y = x + F(x)
"""

print(concepts)

# ============================================
# 第五部分：学习路径回顾
# ============================================

print("\n" + "=" * 70)
print("第五部分：学习路径回顾")
print("=" * 70)

print("""
【10个阶段学习路径】

阶段1: Tokenization
   └─ 理解文本如何变成数字
   
阶段2: Embedding
   └─ 理解数字如何变成向量
   
阶段3: Attention Basics
   └─ 理解 Query, Key, Value
   
阶段4: RoPE
   └─ 理解位置编码
   
阶段5: Multi-Head Attention
   └─ 理解多视角观察
   
阶段6: SwiGLU
   └─ 理解非线性变换
   
阶段7: RMS Norm & Residual
   └─ 理解训练稳定性
   
阶段8: Transformer Layer
   └─ 组装完整层
   
阶段9: Multi-Layer & Output
   └─ 理解逐层抽象和生成
   
阶段10: Summary
   └─ 整合所有知识

【从简单到复杂】

文本 → Token → 向量 → 注意力 → 位置 → 多头 → FFN → 归一化 → 残差 → 多层 → 输出

【从局部到全局】

词级别 → 短语级别 → 句子级别 → 语义级别 → 推理级别
""")

# ============================================
# 第六部分：关键数字记忆
# ============================================

print("\n" + "=" * 70)
print("第六部分：关键数字记忆")
print("=" * 70)

print("""
【Llama3-8B 关键数字】

128,256  │ 词汇表大小
4,096    │ 嵌入维度
32       │ 注意力头数 (Query)
8        │ Key/Value 头数
128      │ 每头维度 (4096/32)
32       │ Transformer 层数
11,008   │ FFN 隐藏维度
~8B      │ 总参数量 (80亿)
500,000  │ RoPE 旋转基数
1e-5     │ RMS Norm epsilon

【形状记忆】

[batch, seq_len]           │ Token IDs
[batch, seq_len, dim]      │ 嵌入和Transformer层
[batch, seq_len, vocab]    │ 输出概率
[dim, dim]                 │ 投影矩阵
[n_heads, seq_len, head_dim] │ 多头分割
""")

# ============================================
# 第七部分：与其他模型对比
# ============================================

print("\n" + "=" * 70)
print("第七部分：与其他模型对比")
print("=" * 70)

print("""
【不同 Transformer 模型对比】

模型        │ 参数量 │ 层数 │ 维度 │ 头数 │ 特点
────────────┼────────┼──────┼──────┼──────┼────────────────
GPT-3       │ 175B   │ 96   │ 12288│ 96   │ 早期大模型
GPT-4       │ ?      │ ?    │ ?    │ ?    │ 闭源
Llama2-7B   │ 7B     │ 32   │ 4096 │ 32   │ 开源
Llama2-70B  │ 70B    │ 80   │ 8192 │ 64   │ 大版本
Llama3-8B   │ 8B     │ 32   │ 4096 │ 32   │ 新架构
Llama3-70B  │ 70B    │ 80   │ 8192 │ 64   │ 大版本

【Llama3 改进】

1. 词汇表更大: 128K (vs 32K)
2. 上下文更长: 128K (vs 4K)
3. RoPE 基数更大: 500K (vs 10K)
4. GQA 优化: 节省 KV 缓存
5. 训练数据更多: 15T tokens
""")

# ============================================
# 第八部分：实践建议
# ============================================

print("\n" + "=" * 70)
print("第八部分：实践建议")
print("=" * 70)

print("""
【下一步学习建议】

1. 深入阅读
   └─ 原始论文: "Attention Is All You Need"
   └─ Llama3 论文 (Meta)
   └─ RoPE 论文

2. 动手实践
   └─ 用 PyTorch 从头实现小模型
   └─ 在简单数据集上训练
   └─ 可视化注意力权重

3. 阅读源码
   └─ transformers 库 (Hugging Face)
   └─ llama.cpp (推理优化)
   └─ 本项目的原始 notebook

4. 进阶学习
   └─ 模型量化 (INT8, INT4)
   └─ 推理优化 (KV Cache, Flash Attention)
   └─ 微调 (Fine-tuning, LoRA)

【推荐资源】

- Andrej Karpathy 的 "Let's build GPT"
- Jay Alammar 的 Transformer 可视化博客
- 3Blue1Brown 的神经网络视频
""")

# ============================================
# 第九部分：总结
# ============================================

print("\n" + "=" * 70)
print("第九部分：总结")
print("=" * 70)

print("""
【学习成果】

通过这10个阶段，你已经理解了：

✓ Tokenization: 文本如何变成数字
✓ Embedding: 数字如何变成向量
✓ Attention: 模型如何"关注"其他词
✓ RoPE: 模型如何知道位置
✓ Multi-Head: 多视角观察
✓ SwiGLU: 非线性变换
✓ RMS Norm: 训练稳定性
✓ Residual: 梯度流通
✓ Transformer Layer: 完整层结构
✓ Multi-Layer: 逐层抽象
✓ Output: 生成下一个词

【核心理解】

Llama3 的本质是:
"通过多层 Transformer，将输入文本逐层抽象，
 最终预测下一个词的概率"

每一层都在学习:
- 词与词的关系 (Attention)
- 复杂的非线性模式 (FFN)
- 从局部到全局的抽象

【关键洞察】

1. Transformer 是通用的序列转换器
2. 注意力机制是核心创新
3. 深度带来抽象能力
4. 大规模数据 + 大模型 = 涌现能力

【恭喜完成学习！】

你已经从零开始理解了 Llama3 的完整架构！
这是理解现代大语言模型的坚实基础。
""")

# ============================================
# 第十部分：最终测试
# ============================================

print("\n" + "=" * 70)
print("第十部分：最终测试")
print("=" * 70)

questions = """
【自我检测】

请回答以下问题，检验学习成果：

1. Tokenization 的作用是什么？
   答案: 文本 → Token IDs

2. Embedding 层的作用是什么？
   答案: Token ID → 语义向量

3. Attention 的核心是什么？
   答案: Query, Key, Value 计算

4. RoPE 的作用是什么？
   答案: 添加位置信息（通过旋转）

5. 为什么需要多头注意力？
   答案: 多视角学习不同关系

6. SwiGLU 的作用是什么？
   答案: 非线性变换，增加表达能力

7. 残差连接的作用是什么？
   答案: 解决梯度消失

8. RMS Norm 的作用是什么？
   答案: 训练稳定性

9. 为什么需要32层？
   答案: 逐层抽象，从局部到全局

10. 输出层的作用是什么？
    答案: 向量 → 词汇概率

【如果都能回答，说明你已经掌握了 Llama3！】
"""

print(questions)

# ============================================
# 结束
# ============================================

print("\n" + "=" * 70)
print("🎉 恭喜完成 Llama3 从零实现学习！")
print("=" * 70)

print("""
【课程总结】

10个阶段，从 Tokenization 到完整模型输出
你已经理解了现代大语言模型的核心机制！

关键收获:
- 理解了 Transformer 架构
- 掌握了 Attention 机制
- 了解了训练稳定性技巧
- 认识了 Llama3 的设计

【感谢学习！】

希望这个课程对你理解大语言模型有所帮助。
继续深入学习，探索 AI 的无限可能！
""")

print("\n" + "=" * 70)
print("课程结束")
print("=" * 70)
