"""
阶段 7: RMS 归一化与残差连接
理解训练稳定性的关键技术
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 7: RMS 归一化与残差连接")
print("=" * 70)

# ============================================
# 第一部分：为什么需要归一化？
# ============================================

print("\n" + "=" * 70)
print("第一部分：为什么需要归一化？")
print("=" * 70)

print("""
【深度网络的问题】

随着网络变深，数值会爆炸或消失：

第1层: [-1, 1]
第2层: [-10, 10]      ← 放大10倍
第3层: [-100, 100]    ← 继续放大
...
第32层: [-10^30, 10^30] ← 爆炸！

或者：
第1层: [-1, 1]
第2层: [-0.1, 0.1]    ← 缩小
第3层: [-0.01, 0.01]  ← 继续缩小
...
第32层: 接近 0        ← 消失！

【归一化的作用】

保持数值在稳定范围 [-1, 1] 附近
让每一层的输入分布一致
训练更稳定，收敛更快
""")

# 演示数值爆炸
print("\n【数值爆炸演示】")
x = torch.randn(10) * 0.5  # 初始小数值
print(f"初始: 均值={x.mean():.3f}, 标准差={x.std():.3f}")

for i in range(5):
    W = torch.randn(10, 10) * 1.5  # 权重
    x = torch.matmul(x, W)
    print(f"第{i+1}层: 均值={x.mean():.3f}, 标准差={x.std():.3f}, 范围=[{x.min():.1f}, {x.max():.1f}]")

print("\n数值迅速爆炸！")

# ============================================
# 第二部分：三种归一化对比
# ============================================

print("\n" + "=" * 70)
print("第二部分：三种归一化对比")
print("=" * 70)

print("""
【Batch Norm（批归一化）】

对每个特征，在 batch 维度上归一化

公式:
  mean = x.mean(dim=0)  # 在 batch 上求平均
  var = x.var(dim=0)
  y = (x - mean) / sqrt(var + eps)

缺点:
  - 依赖 batch size
  - 对序列数据不友好
  - 训练和推理行为不一致

【Layer Norm（层归一化）】⭐ Transformer 原版

对每个样本，在特征维度上归一化

公式:
  mean = x.mean(dim=-1, keepdim=True)  # 在特征上求平均
  var = x.var(dim=-1, keepdim=True)
  y = (x - mean) / sqrt(var + eps) * gamma + beta

优点:
  - 不依赖 batch size
  - 适合序列数据
  - 训练和推理一致

【RMS Norm（均方根归一化）】⭐ Llama3 使用

Layer Norm 的简化版，去掉 mean 计算

公式:
  rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
  y = x / rms * gamma

优点:
  - 计算更快（少一次 mean）
  - 效果相当
  - 现代 LLM 都用这个
""")

# 实现三种归一化
def batch_norm(x, eps=1e-5):
    """批归一化（在 batch 维度）"""
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

def layer_norm(x, gamma, beta, eps=1e-5):
    """层归一化（在特征维度）"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta

def rms_norm(x, gamma, eps=1e-5):
    """RMS 归一化（Llama3 使用）"""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * gamma

# 测试
print("\n【三种归一化对比】")
x = torch.randn(4, 8) * 10  # [batch, features]
print(f"输入: 均值={x.mean():.3f}, 标准差={x.std():.3f}")

gamma = torch.ones(8)
beta = torch.zeros(8)

bn_out = batch_norm(x)
print(f"Batch Norm: 均值={bn_out.mean():.6f}, 标准差={bn_out.std():.3f}")

ln_out = layer_norm(x, gamma, beta)
print(f"Layer Norm: 均值={ln_out.mean(dim=-1)}, 标准差={ln_out.std(dim=-1)}")

rms_out = rms_norm(x, gamma)
print(f"RMS Norm: 均值={rms_out.mean():.3f}, 标准差={rms_out.std():.3f}")

# ============================================
# 第三部分：RMS Norm 详解
# ============================================

print("\n" + "=" * 70)
print("第三部分：RMS Norm 详解")
print("=" * 70)

print("""
【RMS Norm 公式推导】

输入: x = [x_1, x_2, ..., x_d]

Step 1: 计算均方根 (RMS)
  rms = sqrt((x_1^2 + x_2^2 + ... + x_d^2) / d + eps)
      = sqrt(mean(x^2) + eps)

Step 2: 归一化
  x_norm = x / rms

Step 3: 缩放 (可学习参数 gamma)
  output = x_norm * gamma

【与 Layer Norm 的区别】

Layer Norm:
  y = (x - mean) / sqrt(var + eps) * gamma + beta
  需要计算 mean 和 var

RMS Norm:
  y = x / sqrt(mean(x^2) + eps) * gamma
  只需要计算 mean(x^2)，少一次操作

【为什么 RMS Norm 有效？】

1. 保持向量的方向（只缩放长度）
2. 计算更快（少一次 mean）
3. 不需要 beta 参数（中心化不是必须的）
""")

# 详细演示 RMS Norm
print("\n【RMS Norm 计算演示】")
x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [2.0, 4.0, 6.0, 8.0]])
gamma = torch.ones(4)
eps = 1e-5

print(f"输入 x:\n{x}")

# Step 1: 计算 RMS
mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
print(f"\nStep 1 - mean(x^2):\n{mean_sq}")

rms = torch.sqrt(mean_sq + eps)
print(f"RMS:\n{rms}")

# Step 2: 归一化
x_norm = x / rms
print(f"\nStep 2 - x / RMS:\n{x_norm}")

# Step 3: 缩放
output = x_norm * gamma
print(f"\nStep 3 - x_norm * gamma:\n{output}")

# 验证
print(f"\n验证 - 归一化后的 RMS:\n{torch.sqrt(x_norm.pow(2).mean(dim=-1, keepdim=True))}")
print("(应该接近 1.0)")

# ============================================
# 第四部分：残差连接（Residual Connection）
# ============================================

print("\n" + "=" * 70)
print("第四部分：残差连接（Residual Connection）")
print("=" * 70)

print("""
【问题：梯度消失】

深度网络反向传播时，梯度逐层衰减：

第32层梯度: 1.0
第31层梯度: 0.9
第30层梯度: 0.81
...
第1层梯度: 0.9^31 ≈ 0.04  ← 几乎为0！

前面层学不到东西！

【残差连接的解决方案】

原始: y = F(x)
残差: y = x + F(x)  ← 关键！加了一个"捷径"

反向传播时，梯度可以直接流回：
  ∂y/∂x = 1 + ∂F(x)/∂x

即使 ∂F(x)/∂x 很小，还有 1 保证梯度流通！

【形象比喻】

没有残差：走 32 层楼梯，每层都累加误差
有残差：  走 32 层楼梯，但每层都有"直达电梯"

梯度可以通过"电梯"直接传回前面层！
""")

# 可视化残差连接
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 没有残差
ax = axes[0]
ax.set_title('没有残差连接', fontsize=14)
ax.text(0.5, 0.9, 'y = F(x)', ha='center', fontsize=16, 
        bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.7, '↓', ha='center', fontsize=20)
ax.text(0.5, 0.5, 'F(x)', ha='center', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='lightcoral'))
ax.text(0.5, 0.3, '↓', ha='center', fontsize=20)
ax.text(0.5, 0.1, '梯度: dF/dx', ha='center', fontsize=12, color='red')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 有残差
ax = axes[1]
ax.set_title('有残差连接', fontsize=14)
ax.text(0.5, 0.9, 'y = x + F(x)', ha='center', fontsize=16,
        bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.7, '↓', ha='center', fontsize=20)

# 画分支
ax.annotate('', xy=(0.3, 0.5), xytext=(0.5, 0.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text(0.2, 0.5, 'x\n(捷径)', ha='center', fontsize=12, color='blue')

ax.annotate('', xy=(0.7, 0.5), xytext=(0.5, 0.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax.text(0.8, 0.5, 'F(x)', ha='center', fontsize=12, color='red',
        bbox=dict(boxstyle='round', facecolor='lightcoral'))

ax.text(0.5, 0.3, '↓', ha='center', fontsize=20)
ax.text(0.5, 0.1, '梯度: 1 + dF/dx', ha='center', fontsize=12, color='green')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.suptitle('残差连接的作用', fontsize=16)
plt.tight_layout()
plt.savefig('residual_connection.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'residual_connection.png'")

# ============================================
# 第五部分：残差连接的数值演示
# ============================================

print("\n" + "=" * 70)
print("第五部分：残差连接的数值演示")
print("=" * 70)

# 模拟深层网络
print("\n【模拟 10 层网络】")

# 没有残差
print("\n没有残差连接:")
x = torch.tensor([1.0])
gradients_no_residual = []

for i in range(10):
    # 模拟前向传播
    W = torch.tensor([0.9])  # 权重小于1
    x = x * W  # 没有残差
    
    # 模拟反向传播梯度
    grad = 0.9 ** (i + 1)
    gradients_no_residual.append(grad)
    print(f"第{i+1}层梯度: {grad:.6f}")

# 有残差
print("\n有残差连接:")
x = torch.tensor([1.0])
gradients_with_residual = []

for i in range(10):
    # 模拟前向传播
    W = torch.tensor([0.9])
    x = x + x * W  # 有残差: x + F(x)
    
    # 模拟反向传播梯度
    grad = 1 + 0.9 ** (i + 1)  # 1 + ∂F/∂x
    gradients_with_residual.append(grad)
    print(f"第{i+1}层梯度: {grad:.6f}")

# 可视化梯度对比
fig, ax = plt.subplots(figsize=(10, 6))
layers = range(1, 11)
ax.plot(layers, gradients_no_residual, 'r-o', label='没有残差', linewidth=2, markersize=8)
ax.plot(layers, gradients_with_residual, 'g-s', label='有残差', linewidth=2, markersize=8)
ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='y=1')
ax.set_xlabel('层数', fontsize=12)
ax.set_ylabel('梯度大小', fontsize=12)
ax.set_title('残差连接对梯度的影响', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'gradient_flow.png'")

# ============================================
# 第六部分：在 Transformer 中的位置
# ============================================

print("\n" + "=" * 70)
print("第六部分：在 Transformer 中的位置")
print("=" * 70)

print("""
【Pre-Norm vs Post-Norm】

有两种放置归一化的方式：

Post-Norm（原始 Transformer）:
  x → Attention → Norm → +x → FFN → Norm → +x
  问题：深层时梯度仍然不稳定

Pre-Norm（Llama3 使用）⭐:
  x → Norm → Attention → +x → Norm → FFN → +x
  优点：更稳定，现代 LLM 都用这个

【Llama3 的结构】

输入: X
    ↓
【注意力子层】
    RMS Norm(X)
    Multi-Head Attention
    X + Attention(X)  ← 残差连接
    ↓
【FFN 子层】
    RMS Norm(X)
    SwiGLU FFN
    X + FFN(X)  ← 残差连接
    ↓
输出

【关键点】

1. 先归一化，再计算（Pre-Norm）
2. 每个子层都有残差连接
3. 残差连接在归一化之后
""")

# 实现 Pre-Norm Transformer 层
class TransformerBlock(nn.Module):
    """Transformer 块（Pre-Norm）"""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        
        # 归一化
        self.attn_norm = nn.LayerNorm(dim)  # 实际用 RMSNorm
        self.ffn_norm = nn.LayerNorm(dim)
        
        # 注意力（简化版）
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        # FFN（简化版）
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        """
        x: [batch, seq, dim]
        """
        # 注意力子层（Pre-Norm + 残差）
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out  # 残差连接
        
        # FFN 子层（Pre-Norm + 残差）
        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out  # 残差连接
        
        return x

# 测试
print("\n【测试 Transformer Block】")
block = TransformerBlock(dim=64, n_heads=4)
x = torch.randn(2, 8, 64)
output = block(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

# ============================================
# 第七部分：Llama3 的配置
# ============================================

print("\n" + "=" * 70)
print("第七部分：Llama3 的归一化配置")
print("=" * 70)

print("""
【Llama3-8B 配置】

归一化类型: RMS Norm
归一化位置: Pre-Norm（子层之前）
eps (epsilon): 1e-5

【为什么 eps 很重要？】

RMS = sqrt(mean(x^2) + eps)

如果没有 eps:
  当 x 全为 0 时，RMS = 0
  导致除以 0 错误！

eps 保证数值稳定性

【参数量】

每个 RMS Norm 层有一个可学习的 gamma 参数：
  gamma: [dim] = 4096

每层有 2 个 RMS Norm（Attention 前和 FFN 前）：
  每层: 2 × 4096 = 8192

32 层总计: 32 × 8192 = 262,144 (约 0.26M)

相对于总参数量（8B），归一化参数可以忽略
""")

# ============================================
# 第八部分：总结
# ============================================

print("\n" + "=" * 70)
print("第八部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. 为什么需要归一化：
   - 防止数值爆炸或消失
   - 保持每层输入分布一致
   - 训练更稳定

2. RMS Norm：
   - 公式: y = x / sqrt(mean(x^2) + eps) * gamma
   - 比 Layer Norm 更快（少一次 mean）
   - Llama3 使用

3. 残差连接：
   - 公式: y = x + F(x)
   - 解决梯度消失问题
   - 梯度可以通过"捷径"直接传回

4. Pre-Norm 结构：
   - 先归一化，再计算
   - 每个子层后都有残差连接
   - 现代 LLM 标准做法

5. 完整 Transformer 块：
   Input
     ↓
   RMS Norm → Attention → +残差
     ↓
   RMS Norm → FFN → +残差
     ↓
   Output

【下一步】

现在我们已经学习了所有组件：
✓ 嵌入层
✓ 注意力机制（多头 + RoPE）
✓ FFN (SwiGLU)
✓ RMS 归一化
✓ 残差连接

阶段 8 将把所有组件组装成完整的 Transformer 层！
""")

print("\n" + "=" * 70)
print("阶段 7 完成！")
print("=" * 70)
