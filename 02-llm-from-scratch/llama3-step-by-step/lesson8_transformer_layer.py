"""
阶段 8: 组装完整 Transformer 层
把所有组件整合在一起
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 8: 组装完整 Transformer 层")
print("=" * 70)

# ============================================
# 第一部分：回顾所有组件
# ============================================

print("\n" + "=" * 70)
print("第一部分：回顾所有组件")
print("=" * 70)

print("""
【我们已经学习的组件】

1. 嵌入层 (Embedding)
   作用: Token ID → 向量
   形状: [seq_len] → [seq_len, dim]

2. RMS 归一化 (RMS Norm)
   作用: 保持数值稳定
   公式: y = x / sqrt(mean(x^2) + eps) * gamma
   位置: 每个子层之前

3. 多头注意力 (Multi-Head Attention)
   作用: 学习词间关系
   组件:
     - Q, K, V 投影
     - RoPE 位置编码
     - 多头并行计算
     - 输出投影
   形状: [seq_len, dim] → [seq_len, dim]

4. RoPE (Rotary Position Embedding)
   作用: 添加位置信息
   方法: 旋转向量
   位置: 注意力内部的 Q 和 K

5. 残差连接 (Residual Connection)
   作用: 解决梯度消失
   公式: y = x + F(x)
   位置: 每个子层之后

6. SwiGLU FFN
   作用: 非线性变换
   结构: (silu(x @ w1) * (x @ w3)) @ w2
   位置: 第二个子层

【现在要把它们组装起来！】
""")

# ============================================
# 第二部分：完整 Transformer 层的结构
# ============================================

print("\n" + "=" * 70)
print("第二部分：完整 Transformer 层的结构")
print("=" * 70)

print("""
【Transformer 层流程图】

输入: X [seq_len, dim]
    ↓
┌─────────────────────────────────────┐
│         注意力子层 (Attention)       │
│                                     │
│  ┌─ RMS Norm(X)                     │
│  │      ↓                           │
│  │  Multi-Head Attention            │
│  │  (with RoPE)                     │
│  │      ↓                           │
│  └─ X + Attention(X)  ← 残差连接    │
│         ↓                           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│           FFN 子层                   │
│                                     │
│  ┌─ RMS Norm(X)                     │
│  │      ↓                           │
│  │  SwiGLU FFN                      │
│  │      ↓                           │
│  └─ X + FFN(X)  ← 残差连接          │
│         ↓                           │
└─────────────────────────────────────┘
    ↓
输出: [seq_len, dim]

【关键设计】

1. Pre-Norm: 先归一化，再计算
2. 残差连接: 每个子层后都有
3. 两个子层: Attention(沟通) + FFN(思考)
""")

# ============================================
# 第三部分：逐步实现完整 Transformer 层
# ============================================

print("\n" + "=" * 70)
print("第三部分：逐步实现完整 Transformer 层")
print("=" * 70)

# 参数设置
dim = 64
n_heads = 4
seq_len = 8

print(f"\n【参数】")
print(f"维度 (dim): {dim}")
print(f"头数 (n_heads): {n_heads}")
print(f"序列长度 (seq_len): {seq_len}")

# 创建输入
torch.manual_seed(42)
x = torch.randn(seq_len, dim)
print(f"\n【输入】形状: {x.shape}")
print(f"输入范围: [{x.min():.3f}, {x.max():.3f}]")

# --------------------
# 子层 1: 注意力
# --------------------
print("\n" + "-" * 70)
print("子层 1: 多头注意力")
print("-" * 70)

# Step 1: RMS Norm
print("\nStep 1: RMS Norm")
x_norm = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
print(f"  RMS Norm 后: 均值={x_norm.mean():.3f}, 标准差={x_norm.std():.3f}")

# Step 2: 多头注意力（简化实现）
print("\nStep 2: 多头注意力")
head_dim = dim // n_heads

# Q, K, V 投影
W_q = torch.randn(dim, dim) * 0.1
W_k = torch.randn(dim, dim) * 0.1
W_v = torch.randn(dim, dim) * 0.1
W_o = torch.randn(dim, dim) * 0.1

Q = torch.matmul(x_norm, W_q)
K = torch.matmul(x_norm, W_k)
V = torch.matmul(x_norm, W_v)

print(f"  Q, K, V 形状: {Q.shape}")

# 分成多头
Q_heads = Q.view(seq_len, n_heads, head_dim).transpose(0, 1)
K_heads = K.view(seq_len, n_heads, head_dim).transpose(0, 1)
V_heads = V.view(seq_len, n_heads, head_dim).transpose(0, 1)

print(f"  分头后: {Q_heads.shape} [n_heads, seq_len, head_dim]")

# 简化的 RoPE（旋转位置编码）
def apply_rope_simple(x, positions):
    """简化的 RoPE"""
    seq_len, dim = x.shape
    x_pairs = x.float().reshape(seq_len, -1, 2)
    x_complex = torch.view_as_complex(x_pairs)
    
    # 简化的旋转
    angles = positions.float().unsqueeze(1) * 0.1
    rotation = torch.polar(torch.ones_like(angles), angles)
    
    x_rotated = x_complex * rotation
    return torch.view_as_real(x_rotated).reshape(seq_len, dim)

positions = torch.arange(seq_len)
Q_rotated = torch.stack([apply_rope_simple(Q_heads[h], positions) for h in range(n_heads)])
K_rotated = torch.stack([apply_rope_simple(K_heads[h], positions) for h in range(n_heads)])

print(f"  RoPE 后: {Q_rotated.shape}")

# 计算注意力
head_outputs = []
for h in range(n_heads):
    scores = torch.matmul(Q_rotated[h], K_rotated[h].T) / (head_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, V_heads[h])
    head_outputs.append(out)

# 合并头
concat = torch.stack(head_outputs, dim=1).transpose(0, 1).reshape(seq_len, dim)
attn_out = torch.matmul(concat, W_o)

print(f"  注意力输出: {attn_out.shape}")

# Step 3: 残差连接
print("\nStep 3: 残差连接")
x_after_attn = x + attn_out
print(f"  残差后: {x_after_attn.shape}")
print(f"  范围: [{x_after_attn.min():.3f}, {x_after_attn.max():.3f}]")

# --------------------
# 子层 2: FFN
# --------------------
print("\n" + "-" * 70)
print("子层 2: SwiGLU FFN")
print("-" * 70)

# Step 1: RMS Norm
print("\nStep 1: RMS Norm")
x_norm2 = x_after_attn / torch.sqrt(x_after_attn.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
print(f"  RMS Norm 后: 均值={x_norm2.mean():.3f}, 标准差={x_norm2.std():.3f}")

# Step 2: SwiGLU
print("\nStep 2: SwiGLU")
hidden_dim = dim * 4

w1 = torch.randn(dim, hidden_dim) * 0.1
w3 = torch.randn(dim, hidden_dim) * 0.1
w2 = torch.randn(hidden_dim, dim) * 0.1

x_w1 = torch.matmul(x_norm2, w1)
x_w3 = torch.matmul(x_norm2, w3)
silu_xw1 = F.silu(x_w1)
gated = silu_xw1 * x_w3
ffn_out = torch.matmul(gated, w2)

print(f"  FFN 输出: {ffn_out.shape}")

# Step 3: 残差连接
print("\nStep 3: 残差连接")
x_after_ffn = x_after_attn + ffn_out
print(f"  残差后: {x_after_ffn.shape}")
print(f"  范围: [{x_after_ffn.min():.3f}, {x_after_ffn.max():.3f}]")

# 最终输出
output = x_after_ffn
print(f"\n【最终输出】形状: {output.shape}")

# ============================================
# 第四部分：完整的 TransformerLayer 类
# ============================================

print("\n" + "=" * 70)
print("第四部分：完整的 TransformerLayer 类")
print("=" * 70)

class RMSNorm(nn.Module):
    """RMS 归一化"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma

class RoPEMultiHeadAttention(nn.Module):
    """带 RoPE 的多头注意力"""
    def __init__(self, dim, n_heads, n_kv_heads=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        
    def apply_rope(self, x, positions):
        """应用 RoPE - 简化版"""
        # x: [batch, n_heads, seq_len, head_dim]
        batch_size, n_heads, seq_len, head_dim = x.shape
        
        # reshape to [batch, n_heads, seq_len, head_dim//2, 2]
        x_pairs = x.float().reshape(batch_size, n_heads, seq_len, -1, 2)
        x_complex = torch.view_as_complex(x_pairs)
        
        # compute frequencies
        dim_half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim_half).float() / dim_half))
        angles = torch.outer(positions.float(), freqs)  # [seq_len, dim_half]
        
        # expand for broadcasting
        angles = angles.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim_half]
        rotation = torch.polar(torch.ones_like(angles), angles)
        
        # apply rotation
        x_rotated = x_complex * rotation
        return torch.view_as_real(x_rotated).reshape(batch_size, n_heads, seq_len, head_dim).to(x.dtype)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        positions = torch.arange(seq_len, device=x.device)
        
        # Q, K, V
        Q = self.W_q(x)  # [batch, seq, dim]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # reshape: [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # apply RoPE to Q and K
        Q_rotated = self.apply_rope(Q, positions)
        
        # repeat K for GQA
        K_repeated = K.repeat(1, self.n_heads // self.n_kv_heads, 1, 1)
        K_rotated = self.apply_rope(K_repeated, positions)
        
        # attention
        scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        
        # repeat V for GQA
        V_repeated = V.repeat(1, self.n_heads // self.n_kv_heads, 1, 1)
        out = torch.matmul(weights, V_repeated)
        
        # merge heads: [batch, n_heads, seq, head_dim] -> [batch, seq, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.W_o(out)

class SwiGLU(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerLayer(nn.Module):
    """完整的 Transformer 层"""
    def __init__(self, dim, n_heads, n_kv_heads=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # 注意力子层
        self.attn_norm = RMSNorm(dim)
        self.attn = RoPEMultiHeadAttention(dim, n_heads, n_kv_heads)
        
        # FFN 子层
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        # 注意力子层 (Pre-Norm + 残差)
        h = x + self.attn(self.attn_norm(x))
        
        # FFN 子层 (Pre-Norm + 残差)
        out = h + self.ffn(self.ffn_norm(h))
        
        return out

# 测试
print("\n【测试完整 TransformerLayer】")
transformer = TransformerLayer(dim=64, n_heads=4, n_kv_heads=2)
x_test = torch.randn(2, 8, 64)
output = transformer(x_test)

print(f"输入形状: {x_test.shape}")
print(f"输出形状: {output.shape}")
print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

# 计算参数量
def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"\n参数量: {count_params(transformer):,}")

# ============================================
# 第五部分：可视化数据流
# ============================================

print("\n" + "=" * 70)
print("第五部分：可视化数据流")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(5, 9.5, 'Transformer 层数据流', ha='center', fontsize=16, fontweight='bold')

# 输入
ax.text(5, 9, '输入 X [seq_len, dim]', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.annotate('', xy=(5, 8.5), xytext=(5, 8.8),
            arrowprops=dict(arrowstyle='->', lw=2))

# 注意力子层框
ax.add_patch(plt.Rectangle((1, 5.5), 8, 2.5, fill=True, 
                            facecolor='lightyellow', edgecolor='orange', linewidth=2))
ax.text(5, 7.5, '注意力子层', ha='center', fontsize=13, fontweight='bold')

ax.text(5, 7, 'RMS Norm', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(5, 6.5, 'Multi-Head Attention (with RoPE)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(5, 6, '残差连接: X + Attention(X)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.annotate('', xy=(5, 5.3), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2))

# FFN 子层框
ax.add_patch(plt.Rectangle((1, 2.5), 8, 2.5, fill=True,
                            facecolor='lightcyan', edgecolor='blue', linewidth=2))
ax.text(5, 4.5, 'FFN 子层', ha='center', fontsize=13, fontweight='bold')

ax.text(5, 4, 'RMS Norm', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(5, 3.5, 'SwiGLU FFN', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white'))
ax.text(5, 3, '残差连接: X + FFN(X)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.annotate('', xy=(5, 2.3), xytext=(5, 2.5),
            arrowprops=dict(arrowstyle='->', lw=2))

# 输出
ax.text(5, 1.5, '输出 [seq_len, dim]', ha='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('transformer_layer_flow.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'transformer_layer_flow.png'")

# ============================================
# 第六部分：Llama3 的 Transformer 层配置
# ============================================

print("\n" + "=" * 70)
print("第六部分：Llama3 的 Transformer 层配置")
print("=" * 70)

print("""
【Llama3-8B Transformer 层配置】

维度 (dim): 4,096
注意力头数 (n_heads): 32
KV 头数 (n_kv_heads): 8
FFN 隐藏维度: 11,008

【参数量计算】

1. RMS Norm (2个):
   gamma: 2 × 4,096 = 8,192

2. 多头注意力:
   W_q: 4096 × 4096 = 16,777,216
   W_k: 4096 × 1024 = 4,194,304
   W_v: 4096 × 1024 = 4,194,304
   W_o: 4096 × 4096 = 16,777,216
   小计: 41,942,040

3. SwiGLU FFN:
   W1: 4096 × 11008 = 45,088,768
   W3: 4096 × 11008 = 45,088,768
   W2: 11008 × 4096 = 45,088,768
   小计: 135,266,304

每层总计: ~177M 参数
32层总计: ~5.7B 参数
""")

# 计算 Llama3 规模
dim_llama = 4096
n_heads_llama = 32
n_kv_heads_llama = 8
hidden_dim_llama = 11008

params_norm = 2 * dim_llama
params_attn = (dim_llama * dim_llama + 
               dim_llama * n_kv_heads_llama * (dim_llama // n_heads_llama) * 2 +
               dim_llama * dim_llama)
params_ffn = (dim_llama * hidden_dim_llama * 2 + 
              hidden_dim_llama * dim_llama)

total_per_layer = params_norm + params_attn + params_ffn
total_all_layers = total_per_layer * 32

print(f"\n【Llama3-8B 每层参数量】")
print(f"RMS Norm: {params_norm:,}")
print(f"注意力: {params_attn:,}")
print(f"FFN: {params_ffn:,}")
print(f"每层总计: {total_per_layer:,} ({total_per_layer/1e6:.1f}M)")
print(f"32层总计: {total_all_layers:,} ({total_all_layers/1e9:.2f}B)")

# ============================================
# 第七部分：总结
# ============================================

print("\n" + "=" * 70)
print("第七部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. Transformer 层结构:
   输入
     ↓
   [注意力子层]
     - RMS Norm
     - Multi-Head Attention (with RoPE)
     - 残差连接
     ↓
   [FFN 子层]
     - RMS Norm
     - SwiGLU FFN
     - 残差连接
     ↓
   输出

2. 关键设计:
   - Pre-Norm: 先归一化，再计算
   - 残差连接: 每个子层后都有
   - 两个子层: Attention(沟通) + FFN(思考)

3. 形状变化:
   输入:  [batch, seq_len, dim]
   输出:  [batch, seq_len, dim]
   (形状不变，内容被转换)

4. Llama3-8B:
   - 每层: ~177M 参数
   - 32层: ~5.7B 参数
   - 总模型: ~8B 参数（包括嵌入层等）

【下一步】

现在我们有完整的 Transformer 层了！
但一层不够，需要堆叠多层。

阶段 9 将学习: 多层堆叠与模型输出
""")

print("\n" + "=" * 70)
print("阶段 8 完成！")
print("=" * 70)
