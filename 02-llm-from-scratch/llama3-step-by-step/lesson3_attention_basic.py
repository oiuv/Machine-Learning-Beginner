"""
阶段 3: 注意力机制基础 (Attention Basics)
理解 Self-Attention：让模型学会"关注"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 3: 注意力机制基础 (Attention Basics)")
print("=" * 70)

# ============================================
# 第一部分：理解 Query, Key, Value
# ============================================

print("\n" + "=" * 70)
print("第一部分：Query, Key, Value 概念理解")
print("=" * 70)

print("""
【图书馆比喻】

你在图书馆找书：
- Query (查询): 你的问题 "我想找Python编程书"
- Key (键): 书的标签 "Python入门", "Java进阶", "机器学习"
- Value (值): 书的内容（实际知识）

过程：
1. 你的 Query 对比所有书的 Key
2. "Python入门"匹配度高 → 注意力权重高
3. 取出这本书的 Value 阅读

【在 Transformer 中】

每个词都有 Q, K, V 三个向量：
- Query: "我在找谁？"
- Key: "我是谁？"
- Value: "我有什么信息？"

"爱"的 Query 会问："我应该关注哪些词？"
"我"的 Key 会回答："我是'我'，如果你需要主语，请关注我"
"学习"的 Key 会回答："我是'学习'，如果你需要宾语，请关注我"
""")

# ============================================
# 第二部分：最简单的注意力计算
# ============================================

print("\n" + "=" * 70)
print("第二部分：手动实现注意力（极简版）")
print("=" * 70)

# 定义一个小句子
words = ["我", "爱", "学", "习"]
seq_len = len(words)
dim = 8  # 嵌入维度（演示用，Llama3用4096）

print(f"\n【输入句子】: {' '.join(words)}")
print(f"【序列长度】: {seq_len}")
print(f"【嵌入维度】: {dim}")

# 创建模拟的嵌入向量（实际中来自嵌入层）
torch.manual_seed(42)
embeddings = torch.randn(seq_len, dim)

print(f"\n【输入嵌入矩阵】形状: {embeddings.shape}")
print(f"每行是一个词的 {dim} 维向量")

# Step 1: 生成 Q, K, V
print("\n" + "-" * 70)
print("Step 1: 生成 Query, Key, Value")
print("-" * 70)

# 定义权重矩阵（实际中是训练出来的）
W_q = torch.randn(dim, dim) * 0.1
W_k = torch.randn(dim, dim) * 0.1
W_v = torch.randn(dim, dim) * 0.1

# 计算 Q, K, V
Q = torch.matmul(embeddings, W_q)  # [seq_len, dim]
K = torch.matmul(embeddings, W_k)  # [seq_len, dim]
V = torch.matmul(embeddings, W_v)  # [seq_len, dim]

print(f"\nQ (Query) 形状: {Q.shape}")
print(f"K (Key)   形状: {K.shape}")
print(f"V (Value) 形状: {V.shape}")

print("\n每个词现在都有三个身份：")
for i, word in enumerate(words):
    print(f"  '{word}': Q[{i}], K[{i}], V[{i}]")

# Step 2: 计算注意力分数
print("\n" + "-" * 70)
print("Step 2: 计算注意力分数 (Attention Scores)")
print("-" * 70)

print("""
公式: Scores = Q × K^T

意义：每个词的 Query 与所有词的 Key 做点积
      结果表示"我有多关注你"
""")

scores = torch.matmul(Q, K.T)  # [seq_len, seq_len]

print(f"注意力分数矩阵形状: {scores.shape}")
print(f"\n【注意力分数矩阵】（每个词对其他词的关注度）:")
print(f"      {'  '.join([f'{w:>8}' for w in words])}")
for i, word in enumerate(words):
    row = scores[i].detach().numpy()
    print(f"{word:>3}: {row}")

# Step 3: 缩放 + Softmax
print("\n" + "-" * 70)
print("Step 3: 缩放 + Softmax")
print("-" * 70)

print("""
缩放: 除以 √dim，防止数值过大导致 Softmax 梯度消失
Softmax: 把分数变成概率（和为1）
""")

# 缩放
scores_scaled = scores / (dim ** 0.5)

# Softmax
attention_weights = F.softmax(scores_scaled, dim=-1)

print(f"缩放后的分数:")
for i, word in enumerate(words):
    row = scores_scaled[i].detach().numpy()
    print(f"{word:>3}: {row}")

print(f"\n【注意力权重】（Softmax后，每行和为1）:")
print(f"      {'  '.join([f'{w:>8}' for w in words])}")
for i, word in enumerate(words):
    row = attention_weights[i].detach().numpy()
    print(f"{word:>3}: {row}")
    print(f"     行和: {row.sum():.4f}")

# Step 4: 加权求和
print("\n" + "-" * 70)
print("Step 4: 加权求和 (Weighted Sum)")
print("-" * 70)

print("""
公式: Output = Attention_Weights × V

意义：根据注意力权重，组合所有词的 Value
""")

output = torch.matmul(attention_weights, V)  # [seq_len, dim]

print(f"输出形状: {output.shape}")
print(f'每个词现在都有了"上下文信息"！')

print("\n【输出向量】（包含注意力后的信息）:")
for i, word in enumerate(words):
    vec = output[i][:4].detach().numpy()  # 只显示前4维
    print(f"  '{word}': {vec}...")

# ============================================
# 第三部分：可视化注意力权重
# ============================================

print("\n" + "=" * 70)
print("第三部分：可视化注意力权重")
print("=" * 70)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(attention_weights.detach().numpy(), cmap='viridis', aspect='auto')

# 设置标签
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(words)
ax.set_yticklabels(words)

# 在每个格子里显示数值
for i in range(seq_len):
    for j in range(seq_len):
        text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                      ha="center", va="center", color="w" if attention_weights[i, j] > 0.5 else "k",
                      fontsize=10)

ax.set_xlabel('Key (被关注的词)', fontsize=12)
ax.set_ylabel('Query (关注的词)', fontsize=12)
ax.set_title('注意力权重热力图', fontsize=14)
plt.colorbar(im, ax=ax, label='注意力权重')
plt.tight_layout()
plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'attention_heatmap.png'")
print("观察：每行表示一个词对其他词的关注度")

# ============================================
# 第四部分：理解 Self-Attention
# ============================================

print("\n" + "=" * 70)
print("第四部分：理解 Self-Attention（自注意力）")
print("=" * 70)

print("""
【为什么叫"自"注意力？】

因为 Query, Key, Value 都来自同一个输入！

对比其他注意力：
- Self-Attention: Q, K, V 都来自同一个句子
- Cross-Attention: Q 来自句子A，K,V 来自句子B（如翻译时）

【自注意力的意义】

每个词都能"看到"其他所有词！

传统RNN: "我" → "爱" → "学" → "习"（顺序处理，远距离难关联）
Transformer: 每个词直接与其他所有词计算注意力（并行，远距离也能关联）

【注意力权重的含义】

在热力图中：
- 行: Query（谁在问）
- 列: Key（被谁关注）
- 值: 关注度（0-1之间）

理想情况下：
- "爱"应该高关注"我"（主语）和"学习"（宾语）
- "学"应该高关注"爱"（动作来源）
""")

# ============================================
# 第五部分：完整的 Attention 类
# ============================================

print("\n" + "=" * 70)
print("第五部分：完整的 Attention 实现")
print("=" * 70)

class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 定义 Q, K, V 的投影矩阵
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Step 1: 生成 Q, K, V
        Q = self.W_q(x)  # [batch, seq, dim]
        K = self.W_k(x)  # [batch, seq, dim]
        V = self.W_v(x)  # [batch, seq, dim]
        
        # Step 2: 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq, seq]
        
        # Step 3: 缩放 + Softmax
        scores = scores / (dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 4: 加权求和
        output = torch.matmul(attention_weights, V)  # [batch, seq, dim]
        
        return output, attention_weights

# 测试
print("\n【测试 SelfAttention 类】")
attention_layer = SelfAttention(dim)

# 添加 batch 维度
embeddings_batch = embeddings.unsqueeze(0)  # [1, seq_len, dim]
print(f"输入形状: {embeddings_batch.shape}")

output, weights = attention_layer(embeddings_batch)
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")

# ============================================
# 第六部分：Llama3 的注意力参数
# ============================================

print("\n" + "=" * 70)
print("第六部分：Llama3 的注意力参数")
print("=" * 70)

print("""
【Llama3-8B 的注意力配置】

嵌入维度 (dim): 4,096
注意力头数 (n_heads): 32
每个头的维度 (head_dim): 4,096 / 32 = 128

Q, K, V 权重矩阵形状:
- W_q: [4096, 4096]  → 生成 Query
- W_k: [1024, 4096]  → 生成 Key（使用 GQA，8个头共享）
- W_v: [1024, 4096]  → 生成 Value（使用 GQA，8个头共享）

参数量计算:
- W_q: 4096 × 4096 = 16,777,216
- W_k: 1024 × 4096 = 4,194,304
- W_v: 1024 × 4096 = 4,194,304
- W_o: 4096 × 4096 = 16,777,216 (输出投影)
- 总计: ~4200万参数（每层）
- 32层: ~13.4亿参数
""")

# 模拟 Llama3 规模
llama_dim = 4096
llama_heads = 32
llama_seq_len = 17  # 例如 "the answer to the ultimate question..."

print(f"\n【Llama3 注意力计算规模】")
print(f"序列长度: {llama_seq_len}")
print(f"嵌入维度: {llama_dim}")
print(f"注意力头数: {llama_heads}")

# 计算量
q_size = llama_seq_len * llama_dim
k_size = llama_seq_len * llama_dim
scores_ops = llama_seq_len * llama_seq_len * llama_dim

print(f"\nQ 矩阵大小: [{llama_seq_len}, {llama_dim}] = {q_size:,} 元素")
print(f"K 矩阵大小: [{llama_seq_len}, {llama_dim}] = {k_size:,} 元素")
print(f"注意力分数计算: Q×K^T 需要约 {scores_ops:,} 次乘法")
print(f"注意力权重矩阵: [{llama_seq_len}, {llama_seq_len}]")

# ============================================
# 第七部分：总结
# ============================================

print("\n" + "=" * 70)
print("第七部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. 注意力机制的作用：
   - 让模型理解词与词之间的关系
   - 每个词都能"看到"其他所有词

2. Query, Key, Value：
   - Query: "我在找谁？"
   - Key: "我是谁？"
   - Value: "我有什么信息？"

3. 计算步骤：
   Step 1: X × Wq/Wk/Wv → Q, K, V
   Step 2: Q × K^T → 注意力分数
   Step 3: Softmax → 注意力权重
   Step 4: Weights × V → 输出

4. Self-Attention 特点：
   - Q, K, V 来自同一输入
   - 并行计算，速度快
   - 远距离依赖也能捕捉

【形状变化】
输入:  [seq_len, dim]
Q,K,V: [seq_len, dim]
Scores: [seq_len, seq_len]
Weights: [seq_len, seq_len]
输出:  [seq_len, dim]

【下一步】
现在每个词都有了上下文信息，但还缺少位置信息！
"我"在句首和句尾应该有不同的表示。

阶段 4 将学习：位置编码 RoPE
""")

print("\n" + "=" * 70)
print("阶段 3 完成！")
print("=" * 70)
