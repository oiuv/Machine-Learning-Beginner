"""
阶段 5: 多头注意力 (Multi-Head Attention)
理解"多个视角同时观察"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 5: 多头注意力 (Multi-Head Attention)")
print("=" * 70)

# ============================================
# 第一部分：为什么需要多头？
# ============================================

print("\n" + "=" * 70)
print("第一部分：为什么需要多头？")
print("=" * 70)

print("""
【单头注意力的局限】

句子: "猫坐在垫子上，因为它很温暖"

一个句子包含多种关系：
1. 指代关系: "它" → "垫子"
2. 空间关系: "猫" → "坐在" → "垫子"
3. 属性关系: "垫子" → "很温暖"
4. 因果关系: "温暖" → "坐在上面"

【问题】
单头注意力只有一个 Query 向量，
很难同时学习所有类型的关系！

【解决方案：多头】
想象多个人同时读一句话：
- 读者 A: 关注语法结构
- 读者 B: 关注语义关系
- 读者 C: 关注指代关系
- 读者 D: 关注情感色彩

每个人（每个头）学习不同的模式！
""")

# ============================================
# 第二部分：多头注意力的结构
# ============================================

print("\n" + "=" * 70)
print("第二部分：多头注意力的结构")
print("=" * 70)

# 参数设置（模拟 Llama3）
seq_len = 8
dim = 64       # 总维度（Llama3用4096）
n_heads = 4    # 头数（Llama3用32）
head_dim = dim // n_heads  # 每个头的维度

print(f"\n【参数设置】")
print(f"序列长度 (seq_len): {seq_len}")
print(f"总维度 (dim): {dim}")
print(f"头数 (n_heads): {n_heads}")
print(f"每头维度 (head_dim): {head_dim}")

# 创建输入
x = torch.randn(seq_len, dim)
print(f"\n【输入】形状: {x.shape}")

# Step 1: 生成 Q, K, V
print("\n" + "-" * 70)
print("Step 1: 生成 Q, K, V")
print("-" * 70)

W_q = torch.randn(dim, dim) * 0.1
W_k = torch.randn(dim, dim) * 0.1
W_v = torch.randn(dim, dim) * 0.1

Q = torch.matmul(x, W_q)  # [seq_len, dim]
K = torch.matmul(x, W_k)  # [seq_len, dim]
V = torch.matmul(x, W_v)  # [seq_len, dim]

print(f"Q 形状: {Q.shape}")
print(f"K 形状: {K.shape}")
print(f"V 形状: {V.shape}")

# Step 2: 分成 n_heads 份
print("\n" + "-" * 70)
print("Step 2: 分成 n_heads 份")
print("-" * 70)

# reshape: [seq_len, dim] → [seq_len, n_heads, head_dim] → [n_heads, seq_len, head_dim]
Q_split = Q.view(seq_len, n_heads, head_dim).transpose(0, 1)
K_split = K.view(seq_len, n_heads, head_dim).transpose(0, 1)
V_split = V.view(seq_len, n_heads, head_dim).transpose(0, 1)

print(f"Q_split 形状: {Q_split.shape}  [n_heads, seq_len, head_dim]")
print(f"K_split 形状: {K_split.shape}")
print(f"V_split 形状: {V_split.shape}")

print("\n现在每个头有独立的 Q, K, V：")
for h in range(n_heads):
    print(f"  Head {h}: Q[{h}] 形状 {Q_split[h].shape}")

# Step 3: 每个头独立计算注意力
print("\n" + "-" * 70)
print("Step 3: 每个头独立计算注意力")
print("-" * 70)

def single_head_attention(Q_h, K_h, V_h):
    """单头注意力计算"""
    head_dim = Q_h.shape[-1]
    scores = torch.matmul(Q_h, K_h.T) / (head_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V_h)
    return output, weights

head_outputs = []
head_weights = []

for h in range(n_heads):
    out, w = single_head_attention(Q_split[h], K_split[h], V_split[h])
    head_outputs.append(out)
    head_weights.append(w)
    print(f"Head {h}: 输出形状 {out.shape}, 注意力权重形状 {w.shape}")

# Step 4: 合并所有头
print("\n" + "-" * 70)
print("Step 4: 合并所有头")
print("-" * 70)

# stack: [n_heads, seq_len, head_dim] → transpose: [seq_len, n_heads, head_dim] → reshape: [seq_len, dim]
concat_output = torch.stack(head_outputs, dim=0).transpose(0, 1).reshape(seq_len, dim)

print(f"所有头输出堆叠: {torch.stack(head_outputs).shape}")
print(f"转置后: {torch.stack(head_outputs, dim=0).transpose(0, 1).shape}")
print(f"合并后 (Concat): {concat_output.shape}")

# Step 5: 输出投影
print("\n" + "-" * 70)
print("Step 5: 输出投影")
print("-" * 70)

W_o = torch.randn(dim, dim) * 0.1
final_output = torch.matmul(concat_output, W_o)

print(f"输出投影矩阵 W_o 形状: {W_o.shape}")
print(f"最终输出形状: {final_output.shape}")

# ============================================
# 第三部分：可视化不同头的注意力模式
# ============================================

print("\n" + "=" * 70)
print("第三部分：可视化不同头的注意力模式")
print("=" * 70)

words = ["猫", "坐", "在", "垫子", "上", "，", "它", "很", "温暖"][:seq_len]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for h in range(n_heads):
    ax = axes[h]
    im = ax.imshow(head_weights[h].detach().numpy(), cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    
    ax.set_title(f'Head {h+1} 注意力模式', fontsize=12)
    ax.set_xlabel('Key', fontsize=10)
    ax.set_ylabel('Query', fontsize=10)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('多头注意力：每个头学习不同的关系', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('multihead_attention.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'multihead_attention.png'")
print("观察：每个头关注不同的模式！")

# ============================================
# 第四部分：GQA - Grouped Query Attention
# ============================================

print("\n" + "=" * 70)
print("第四部分：GQA - Grouped Query Attention (Llama3优化)")
print("=" * 70)

print("""
【标准多头注意力的计算量】

每个头都有独立的 Q, K, V：
- Q: [n_heads, seq_len, head_dim]
- K: [n_heads, seq_len, head_dim]
- V: [n_heads, seq_len, head_dim]

计算量：3 × n_heads × seq_len × head_dim

当 n_heads=32, seq_len=4096 时，计算量很大！

【GQA 优化思想】

Query 保持 n_heads 个（需要多样化查询）
Key 和 Value 共享，减少数量

Llama3-8B 配置：
- n_heads (Query): 32
- n_kv_heads (Key/Value): 8
- 每 4 个 Query 共享 1 组 Key/Value
""")

# GQA 参数
n_heads_q = 8     # Query 头数
n_heads_kv = 2    # Key/Value 头数（共享）
head_dim_gqa = dim // n_heads_q

print(f"\n【GQA 参数】")
print(f"总维度: {dim}")
print(f"Query 头数: {n_heads_q}")
print(f"Key/Value 头数: {n_heads_kv}")
print(f"每头维度: {head_dim_gqa}")
print(f"共享比例: 每 {n_heads_q // n_heads_kv} 个 Query 共享 1 组 KV")

# GQA 实现
print("\n【GQA 计算流程】")

# 生成 Q, K, V
Q_gqa = torch.matmul(x, torch.randn(dim, dim))  # [seq_len, dim]
K_gqa = torch.matmul(x, torch.randn(dim, dim * n_heads_kv // n_heads_q))  # 更小的 K
V_gqa = torch.matmul(x, torch.randn(dim, dim * n_heads_kv // n_heads_q))  # 更小的 V

print(f"Q 形状: {Q_gqa.shape}")
print(f"K 形状: {K_gqa.shape} (比标准多头小)")
print(f"V 形状: {V_gqa.shape} (比标准多头小)")

# 分割
Q_split_gqa = Q_gqa.view(seq_len, n_heads_q, head_dim_gqa).transpose(0, 1)  # [8, seq_len, head_dim]
K_split_gqa = K_gqa.view(seq_len, n_heads_kv, head_dim_gqa).transpose(0, 1)  # [2, seq_len, head_dim]
V_split_gqa = V_gqa.view(seq_len, n_heads_kv, head_dim_gqa).transpose(0, 1)  # [2, seq_len, head_dim]

print(f"\nQ_split: {Q_split_gqa.shape}")
print(f"K_split: {K_split_gqa.shape} (只有 {n_heads_kv} 组)")
print(f"V_split: {V_split_gqa.shape} (只有 {n_heads_kv} 组)")

# 计算注意力（Query 多，KV 少，需要重复 KV）
gqa_outputs = []
for h in range(n_heads_q):
    # 确定使用哪组 KV
    kv_idx = h // (n_heads_q // n_heads_kv)
    
    Q_h = Q_split_gqa[h]  # [seq_len, head_dim]
    K_h = K_split_gqa[kv_idx]  # [seq_len, head_dim]
    V_h = V_split_gqa[kv_idx]  # [seq_len, head_dim]
    
    out, _ = single_head_attention(Q_h, K_h, V_h)
    gqa_outputs.append(out)
    
print(f"\nGQA 输出数量: {len(gqa_outputs)} 个（与 Query 头数相同）")

# 计算节省的计算量
standard_compute = n_heads_q * seq_len * head_dim_gqa * 3  # Q, K, V
gqa_compute = (n_heads_q + n_heads_kv * 2) * seq_len * head_dim_gqa

print(f"\n【计算量对比】")
print(f"标准多头: {standard_compute:,} 次操作")
print(f"GQA:      {gqa_compute:,} 次操作")
print(f"节省:     {(1 - gqa_compute/standard_compute)*100:.1f}%")

# ============================================
# 第五部分：完整的 MultiHeadAttention 类
# ============================================

print("\n" + "=" * 70)
print("第五部分：完整的 MultiHeadAttention 类")
print("=" * 70)

class MultiHeadAttention(nn.Module):
    """多头注意力（支持 GQA）"""
    def __init__(self, dim, n_heads, n_kv_heads=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # 默认等于 n_heads
        self.head_dim = dim // n_heads
        
        # Q, K, V 投影
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.W_o = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Step 1: 生成 Q, K, V
        Q = self.W_q(x)  # [batch, seq, dim]
        K = self.W_k(x)  # [batch, seq, n_kv_heads * head_dim]
        V = self.W_v(x)  # [batch, seq, n_kv_heads * head_dim]
        
        # Step 2: reshape
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Q: [batch, n_heads, seq, head_dim]
        # K, V: [batch, n_kv_heads, seq, head_dim]
        
        # Step 3: 计算注意力（处理 GQA）
        outputs = []
        for h in range(self.n_heads):
            kv_idx = h // (self.n_heads // self.n_kv_heads) if self.n_heads != self.n_kv_heads else h
            
            Q_h = Q[:, h, :, :]  # [batch, seq, head_dim]
            K_h = K[:, kv_idx, :, :]  # [batch, seq, head_dim]
            V_h = V[:, kv_idx, :, :]  # [batch, seq, head_dim]
            
            scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / (self.head_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)
            out = torch.matmul(weights, V_h)  # [batch, seq, head_dim]
            outputs.append(out)
        
        # Step 4: 合并
        concat = torch.stack(outputs, dim=1)  # [batch, n_heads, seq, head_dim]
        concat = concat.transpose(1, 2).contiguous()  # [batch, seq, n_heads, head_dim]
        concat = concat.view(batch_size, seq_len, dim)  # [batch, seq, dim]
        
        # Step 5: 输出投影
        output = self.W_o(concat)
        return output

# 测试
print("\n【测试 MultiHeadAttention】")
mha = MultiHeadAttention(dim=64, n_heads=4, n_kv_heads=2)
x_test = torch.randn(2, 8, 64)  # [batch, seq, dim]
output = mha(x_test)
print(f"输入形状: {x_test.shape}")
print(f"输出形状: {output.shape}")

# ============================================
# 第六部分：Llama3 的多头注意力配置
# ============================================

print("\n" + "=" * 70)
print("第六部分：Llama3 的多头注意力配置")
print("=" * 70)

print("""
【Llama3-8B 配置】

总维度 (dim): 4,096
Query 头数 (n_heads): 32
Key/Value 头数 (n_kv_heads): 8
每头维度 (head_dim): 4,096 / 32 = 128

【参数量计算】

W_q: [4096, 4096] = 16,777,216
W_k: [4096, 1024] = 4,194,304  (8 heads × 128 dim)
W_v: [4096, 1024] = 4,194,304
W_o: [4096, 4096] = 16,777,216

每层注意力参数量: ~42M
32层总参数量: ~1.3B

【GQA 优势】

标准多头 (32 heads):
  K, V 参数量: 2 × 32 × 128 × 4096 = 33.5M
  
GQA (8 KV heads):
  K, V 参数量: 2 × 8 × 128 × 4096 = 8.4M
  
节省: 75% 的 KV 缓存！
""")

# 计算 Llama3 规模
llama_dim = 4096
llama_heads = 32
llama_kv_heads = 8
llama_head_dim = llama_dim // llama_heads

params_wq = llama_dim * llama_dim
params_wk = llama_dim * (llama_kv_heads * llama_head_dim)
params_wv = llama_dim * (llama_kv_heads * llama_head_dim)
params_wo = llama_dim * llama_dim

total_params = params_wq + params_wk + params_wv + params_wo

print(f"\n【Llama3-8B 注意力参数量】")
print(f"W_q: {params_wq:,}")
print(f"W_k: {params_wk:,}")
print(f"W_v: {params_wv:,}")
print(f"W_o: {params_wo:,}")
print(f"每层总计: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"32层总计: {total_params*32:,} ({total_params*32/1e9:.2f}B)")

# ============================================
# 第七部分：总结
# ============================================

print("\n" + "=" * 70)
print("第七部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. 为什么需要多头：
   - 一个句子有多种关系（语法、语义、指代等）
   - 每个头学习不同类型的关系
   - 多个视角同时观察

2. 多头注意力的流程：
   Step 1: 生成 Q, K, V
   Step 2: 分成 n_heads 份
   Step 3: 每个头独立计算注意力
   Step 4: 合并所有头的输出
   Step 5: 输出投影

3. GQA (Grouped Query Attention)：
   - Query: 32 个头（多样化查询）
   - Key/Value: 8 个头（共享，减少计算）
   - 节省 75% 的 KV 缓存

4. 形状变化：
   输入:        [seq_len, dim]
   Q, K, V:     [seq_len, dim]
   Split:       [n_heads, seq_len, head_dim]
   每头输出:    [seq_len, head_dim]
   合并:        [seq_len, dim]
   最终输出:    [seq_len, dim]

【下一步】

现在模型有了强大的注意力机制，
但还需要非线性变换来增加表达能力。

阶段 6 将学习：前馈网络 (FFN) 和 SwiGLU 激活函数
""")

print("\n" + "=" * 70)
print("阶段 5 完成！")
print("=" * 70)
