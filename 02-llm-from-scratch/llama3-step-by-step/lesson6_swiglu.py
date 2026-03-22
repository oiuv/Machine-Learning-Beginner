"""
阶段 6: 前馈网络 (FFN) 和 SwiGLU
理解非线性变换和门控机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 6: 前馈网络 (FFN) 和 SwiGLU")
print("=" * 70)

# ============================================
# 第一部分：为什么需要非线性？
# ============================================

print("\n" + "=" * 70)
print("第一部分：为什么需要非线性？")
print("=" * 70)

print("""
【注意力机制的局限】

注意力本质 = 加权平均（线性操作）

输入: [x1, x2, x3]
输出: w1*x1 + w2*x2 + w3*x3

无论怎么组合，都是线性的！

【为什么线性不够？】

例句 1: "我喜欢学习" → "喜欢"是正面
例句 2: "我不喜欢"   → "喜欢"是负面

线性模型很难理解这种"上下文依赖的语义变化"

【解决方案：非线性变换】

通过非线性激活函数，模型可以学习更复杂的模式：
- 曲线的决策边界
- 上下文敏感的语义
- 复杂的逻辑关系
""")

# 可视化线性与非线性
x = np.linspace(-3, 3, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 线性
axes[0].plot(x, x, 'b-', linewidth=2)
axes[0].set_title('线性: f(x) = x', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# ReLU
axes[1].plot(x, np.maximum(0, x), 'r-', linewidth=2)
axes[1].set_title('ReLU: f(x) = max(0, x)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# SiLU (Swish)
axes[2].plot(x, x * (1 / (1 + np.exp(-x))), 'g-', linewidth=2)
axes[2].set_title('SiLU: f(x) = x · σ(x)', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.suptitle('激活函数对比', fontsize=14)
plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'activation_functions.png'")

# ============================================
# 第二部分：传统 FFN 结构
# ============================================

print("\n" + "=" * 70)
print("第二部分：传统 FFN 结构")
print("=" * 70)

# 参数设置
dim = 64
hidden_dim = 256  # 扩展 4 倍
seq_len = 8

print(f"\n【参数】")
print(f"输入维度 (dim): {dim}")
print(f"隐藏维度 (hidden_dim): {hidden_dim}")
print(f"扩展比例: {hidden_dim / dim}x")

# 创建输入
x = torch.randn(seq_len, dim)
print(f"\n【输入】形状: {x.shape}")

# 传统 FFN
print("\n【传统 FFN 流程】")
print("输入 → Linear(扩展) → ReLU → Linear(压缩) → 输出")

W1 = torch.randn(dim, hidden_dim) * 0.1
W2 = torch.randn(hidden_dim, dim) * 0.1

# Step 1: 扩展
h = torch.matmul(x, W1)  # [seq_len, hidden_dim]
print(f"Step 1 - 扩展: {x.shape} × {W1.shape} → {h.shape}")

# Step 2: ReLU 激活
h_relu = F.relu(h)
print(f"Step 2 - ReLU: {h_relu.shape}")
print(f"  ReLU 后非零元素比例: {(h_relu > 0).float().mean():.2%}")

# Step 3: 压缩
output = torch.matmul(h_relu, W2)  # [seq_len, dim]
print(f"Step 3 - 压缩: {h_relu.shape} × {W2.shape} → {output.shape}")

print(f"\n【输出】形状: {output.shape}")

# ============================================
# 第三部分：SwiGLU 详解
# ============================================

print("\n" + "=" * 70)
print("第三部分：SwiGLU 详解")
print("=" * 70)

print("""
【SwiGLU 结构】

SwiGLU = Swish + Gated Linear Unit

结构:
输入 x
    ↓
┌─────────┬─────────┐
│         │         │
↓ W1      ↓ W3      │
├─────────┤         │
silu(·)   │         │
    ↓     ↓         │
    ⊗ (逐元素乘)    │
    ↓               │
    W2              │
    ↓               │
  输出              │

公式: SwiGLU(x) = (silu(x @ W1) ⊗ (x @ W3)) @ W2

其中:
- silu(x) = x · sigmoid(x)  (Swish 激活)
- ⊗: 逐元素乘法 (门控)
- W1, W3: 两个不同的投影矩阵
- W2: 输出投影
""")

# SiLU (Swish) 激活函数
print("\n【SiLU (Swish) 激活函数】")
print("公式: silu(x) = x · σ(x) = x / (1 + e^(-x))")

x_test = torch.linspace(-5, 5, 100)
silu_values = F.silu(x_test)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_test.numpy(), x_test.numpy(), 'b--', label='y=x (线性)', alpha=0.5)
ax.plot(x_test.numpy(), silu_values.numpy(), 'r-', linewidth=2, label='SiLU: x·σ(x)')
ax.plot(x_test.numpy(), F.relu(x_test).numpy(), 'g--', label='ReLU', alpha=0.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('SiLU vs ReLU', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('silu_activation.png', dpi=150, bbox_inches='tight')
print("【可视化】已保存为 'silu_activation.png'")

# SwiGLU 实现
print("\n【SwiGLU 计算流程】")

# 参数（Llama3 风格）
dim = 64
hidden_dim = 256  # 通常 dim * 3.5

W1 = torch.randn(dim, hidden_dim) * 0.1
W3 = torch.randn(dim, hidden_dim) * 0.1
W2 = torch.randn(hidden_dim, dim) * 0.1

print(f"W1 形状: {W1.shape}")
print(f"W3 形状: {W3.shape}")
print(f"W2 形状: {W2.shape}")

# 输入
x = torch.randn(seq_len, dim)
print(f"\n输入 x 形状: {x.shape}")

# Step 1: 两个分支
xW1 = torch.matmul(x, W1)  # [seq_len, hidden_dim]
xW3 = torch.matmul(x, W3)  # [seq_len, hidden_dim]

print(f"\nStep 1 - 分支1 (x @ W1): {xW1.shape}")
print(f"Step 1 - 分支2 (x @ W3): {xW3.shape}")

# Step 2: 分支1 过 SiLU
silu_xW1 = F.silu(xW1)
print(f"\nStep 2 - SiLU(x @ W1): {silu_xW1.shape}")

# Step 3: 门控（逐元素乘）
gated = silu_xW1 * xW3
print(f"Step 3 - 门控 (SiLU ⊗ xW3): {gated.shape}")

# Step 4: 输出投影
output_swiglu = torch.matmul(gated, W2)
print(f"Step 4 - 输出 (@ W2): {output_swiglu.shape}")

print(f"\n【SwiGLU 输出】形状: {output_swiglu.shape}")

# ============================================
# 第四部分：SwiGLU vs 传统 FFN
# ============================================

print("\n" + "=" * 70)
print("第四部分：SwiGLU vs 传统 FFN 对比")
print("=" * 70)

print("""
【对比】

传统 FFN:
  x → Linear → ReLU → Linear → output
  参数量: dim × hidden_dim + hidden_dim × dim

SwiGLU:
  x → ┬→ Linear → SiLU ─┐
      └→ Linear ───────→ ⊗ → Linear → output
  参数量: dim × hidden_dim × 2 + hidden_dim × dim
  (多一个 W3)

【为什么 SwiGLU 更好？】

1. 门控机制:
   - 可以"选择"哪些信息通过
   - 像 LSTM 的门，更灵活

2. SiLU 激活:
   - 平滑，处处可导
   - 没有 ReLU 的"死亡"问题

3. 表达能力:
   - 两个分支可以学习不同的特征
   - 组合后表达能力更强
""")

# 数值对比
def traditional_ffn(x, W1, W2):
    return torch.matmul(F.relu(torch.matmul(x, W1)), W2)

def swiglu_ffn(x, W1, W3, W2):
    return torch.matmul(F.silu(torch.matmul(x, W1)) * torch.matmul(x, W3), W2)

# 测试
x_test = torch.randn(4, 64)

# 传统 FFN
W1_trad = torch.randn(64, 256) * 0.1
W2_trad = torch.randn(256, 64) * 0.1
out_trad = traditional_ffn(x_test, W1_trad, W2_trad)

# SwiGLU
W1_swiglu = torch.randn(64, 256) * 0.1
W3_swiglu = torch.randn(64, 256) * 0.1
W2_swiglu = torch.randn(256, 64) * 0.1
out_swiglu = swiglu_ffn(x_test, W1_swiglu, W3_swiglu, W2_swiglu)

print(f"\n【输出对比】")
print(f"传统 FFN 输出形状: {out_trad.shape}")
print(f"SwiGLU 输出形状: {out_swiglu.shape}")
print(f"输出范围 (传统): [{out_trad.min():.3f}, {out_trad.max():.3f}]")
print(f"输出范围 (SwiGLU): [{out_swiglu.min():.3f}, {out_swiglu.max():.3f}]")

# ============================================
# 第五部分：完整的 SwiGLU 类
# ============================================

print("\n" + "=" * 70)
print("第五部分：完整的 SwiGLU FFN 类")
print("=" * 70)

class SwiGLU(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, dim, hidden_dim=None, multiple_of=256):
        super().__init__()
        self.dim = dim
        
        # Llama3 的 hidden_dim 计算
        # hidden_dim = 2/3 * 4 * dim (约 2.67x)
        # 然后向上取整到 multiple_of 的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of
        
        self.hidden_dim = hidden_dim
        
        # 三个投影矩阵
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        # SwiGLU: (silu(x @ w1) * (x @ w3)) @ w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# 测试
print("\n【测试 SwiGLU 类】")
swiglu = SwiGLU(dim=64, hidden_dim=256)
x_test = torch.randn(2, 8, 64)  # [batch, seq, dim]
output = swiglu(x_test)

print(f"输入形状: {x_test.shape}")
print(f"隐藏维度: {swiglu.hidden_dim}")
print(f"输出形状: {output.shape}")

# 计算参数量
params_w1 = 64 * 256
params_w3 = 64 * 256
params_w2 = 256 * 64
total = params_w1 + params_w3 + params_w2

print(f"\n参数量:")
print(f"  W1: {params_w1:,}")
print(f"  W3: {params_w3:,}")
print(f"  W2: {params_w2:,}")
print(f"  总计: {total:,} ({total/1e6:.2f}M)")

# ============================================
# 第六部分：Llama3 的 FFN 配置
# ============================================

print("\n" + "=" * 70)
print("第六部分：Llama3 的 FFN 配置")
print("=" * 70)

print("""
【Llama3-8B FFN 配置】

输入维度 (dim): 4,096
隐藏维度计算:
  1. 基础: 4 × 4096 = 16,384
  2. SwiGLU 调整: 2/3 × 16,384 = 10,922.67
  3. 向上取整到 256 倍数: 10,944
  
实际 hidden_dim: 11,008 (Llama3 配置)

参数量:
  W1: 4096 × 11008 = 45,088,768
  W3: 4096 × 11008 = 45,088,768
  W2: 11008 × 4096 = 45,088,768
  总计: 135,266,304 (约 135M)

每层 FFN 参数量: ~135M
32 层总参数量: ~4.3B

【为什么 hidden_dim 这么大？】

1. 扩展 → 增加表达能力
2. 非线性激活 → 学习复杂模式
3. 压缩 → 回到原始维度
""")

# 计算 Llama3 规模
dim_llama = 4096
hidden_llama = 11008

params_w1_llama = dim_llama * hidden_llama
params_w3_llama = dim_llama * hidden_llama
params_w2_llama = hidden_llama * dim_llama
total_llama = params_w1_llama + params_w3_llama + params_w2_llama

print(f"\n【Llama3-8B FFN 参数量】")
print(f"W1: {params_w1_llama:,}")
print(f"W3: {params_w3_llama:,}")
print(f"W2: {params_w2_llama:,}")
print(f"每层总计: {total_llama:,} ({total_llama/1e6:.1f}M)")
print(f"32层总计: {total_llama*32:,} ({total_llama*32/1e9:.2f}B)")

# ============================================
# 第七部分：在 Transformer 中的位置
# ============================================

print("\n" + "=" * 70)
print("第七部分：FFN 在 Transformer 中的位置")
print("=" * 70)

print("""
【Transformer 层的完整结构】

输入: X
    ↓
【注意力子层】
    RMS Norm
    Multi-Head Attention (with RoPE)
    残差连接: X + Attention(X)
    ↓
【FFN 子层】
    RMS Norm
    SwiGLU FFN
    残差连接: X + FFN(X)
    ↓
输出

【为什么 FFN 在注意力之后？】

1. 注意力: 收集上下文信息（"看"其他词）
2. FFN: 对每个位置独立变换（"思考"）

注意力是"沟通"，FFN 是"思考"
""")

# 模拟完整流程
print("\n【模拟 Transformer 层流程】")

seq_len = 8
dim = 64

# 输入
x = torch.randn(seq_len, dim)
print(f"输入: {x.shape}")

# 注意力子层（简化）
attn_output = torch.randn(seq_len, dim)  # 模拟注意力输出
x_after_attn = x + attn_output  # 残差连接
print(f"注意力后: {x_after_attn.shape}")

# FFN 子层
swiglu = SwiGLU(dim=dim, hidden_dim=256)
ffn_output = swiglu(x_after_attn.unsqueeze(0)).squeeze(0)
x_after_ffn = x_after_attn + ffn_output  # 残差连接
print(f"FFN 后: {x_after_ffn.shape}")

print(f"\n输出: {x_after_ffn.shape}")

# ============================================
# 第八部分：总结
# ============================================

print("\n" + "=" * 70)
print("第八部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. 为什么需要 FFN：
   - 注意力是线性的（加权平均）
   - 需要非线性变换学习复杂模式
   - 增加模型的表达能力

2. SwiGLU 结构：
   - 两个分支: x @ W1 和 x @ W3
   - 分支1 过 SiLU 激活
   - 门控: 逐元素相乘
   - 输出投影: @ W2

3. 公式:
   SwiGLU(x) = (silu(x @ W1) ⊗ (x @ W3)) @ W2

4. 优势:
   - 门控机制: 灵活控制信息流
   - SiLU: 平滑，处处可导
   - 比 ReLU 表达能力更强

5. Llama3 配置:
   - dim: 4,096
   - hidden_dim: 11,008
   - 每层参数量: ~135M

【形状变化】
输入:     [seq_len, dim]
分支1:    [seq_len, hidden_dim]
分支2:    [seq_len, hidden_dim]
SiLU:     [seq_len, hidden_dim]
门控:     [seq_len, hidden_dim]
输出:     [seq_len, dim]

【下一步】

现在模型有了：
✓ 嵌入层
✓ 注意力机制
✓ 位置编码
✓ 多头注意力
✓ FFN (SwiGLU)

但还缺少：
- 归一化（稳定训练）
- 残差连接（缓解梯度消失）

阶段 7 将学习: RMS 归一化与残差连接
""")

print("\n" + "=" * 70)
print("阶段 6 完成！")
print("=" * 70)
