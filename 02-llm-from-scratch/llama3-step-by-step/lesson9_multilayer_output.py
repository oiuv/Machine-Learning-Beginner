"""
阶段 9: 多层堆叠与模型输出
理解逐层抽象和输出生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 9: 多层堆叠与模型输出")
print("=" * 70)

# ============================================
# 第一部分：为什么需要多层？
# ============================================

print("\n" + "=" * 70)
print("第一部分：为什么需要多层？")
print("=" * 70)

print("""
【单层 Transformer 的局限】

一个 Transformer 层能做什么？
- 看到相邻词的关系
- 提取局部特征

【但语言有层次结构！】

例句: "猫坐在垫子上，因为它很温暖"

第1层: 学习词与词的关系
       "猫"-"坐", "坐"-"垫子"
       
第2层: 学习短语关系
       "猫坐"-"垫子"
       
第3层: 学习句子结构
       "猫坐在垫子上"-"它很温暖"
       
...
第32层: 理解整体语义和推理
        理解"它"指"垫子"
        理解因果关系

【逐层抽象】

就像人类理解语言：
- 第1层: 识别字母
- 第2层: 识别单词
- 第3层: 理解语法
- 第4层: 理解语义
- ...
- 第32层: 理解隐含意义、推理
""")

# 可视化逐层抽象
fig, ax = plt.subplots(figsize=(12, 8))

layers = list(range(1, 33))
abstraction_level = [i/32 for i in layers]  # 抽象程度递增

# 绘制抽象程度
ax.barh(layers, abstraction_level, color=plt.cm.viridis(np.linspace(0, 1, 32)))
ax.set_xlabel('抽象程度', fontsize=12)
ax.set_ylabel('层数', fontsize=12)
ax.set_title('Transformer 逐层抽象', fontsize=14)
ax.set_xlim(0, 1.2)

# 添加注释
ax.text(0.5, 5, '局部特征\n(词级别)', ha='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(0.5, 15, '短语特征', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(0.5, 25, '句子结构', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(0.5, 31, '语义推理', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig('layer_abstraction.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'layer_abstraction.png'")

# ============================================
# 第二部分：多层堆叠的结构
# ============================================

print("\n" + "=" * 70)
print("第二部分：多层堆叠的结构")
print("=" * 70)

print("""
【Llama3-8B 的多层结构】

输入: Token Embeddings [seq_len, dim]
    ↓
【第1层 Transformer】
    提取局部特征（词级别）
    ↓
【第2层 Transformer】
    提取短语特征
    ↓
【第3层 Transformer】
    提取句子特征
    ↓
...
    ↓
【第32层 Transformer】
    提取语义和推理特征
    ↓
【输出层】
    Linear: [dim, vocab_size]
    Softmax → 概率分布
    ↓
输出: 下一个 Token 的概率

【关键设计】

1. 每层输入输出形状相同: [seq_len, dim]
   方便堆叠

2. 残差连接保证梯度流通
   可以训练深层网络

3. 32层足够学习复杂模式
   但又不至于过深导致训练困难
""")

# ============================================
# 第三部分：简化的多层实现
# ============================================

print("\n" + "=" * 70)
print("第三部分：简化的多层实现")
print("=" * 70)

# 参数
dim = 64
n_heads = 4
n_layers = 4  # 演示用4层，Llama3用32层
seq_len = 8

print(f"\n【参数】")
print(f"维度: {dim}")
print(f"头数: {n_heads}")
print(f"层数: {n_layers} (演示用，Llama3用32)")

# 简化的 Transformer 层
class SimpleTransformerLayer(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        # 简化的注意力
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # 简化的 FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Attention + 残差
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        
        # FFN + 残差
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        
        return x

# 创建多层模型
print("\n【创建多层模型】")
layers = nn.ModuleList([SimpleTransformerLayer(dim, n_heads) for _ in range(n_layers)])
print(f"创建了 {n_layers} 个 Transformer 层")

# 测试前向传播
x = torch.randn(1, seq_len, dim)
print(f"\n输入形状: {x.shape}")

# 逐层传播
for i, layer in enumerate(layers):
    x = layer(x)
    print(f"第{i+1}层输出: {x.shape}, 范围=[{x.min():.3f}, {x.max():.3f}]")

print(f"\n最终输出: {x.shape}")

# ============================================
# 第四部分：输出层
# ============================================

print("\n" + "=" * 70)
print("第四部分：输出层")
print("=" * 70)

print("""
【输出层的作用】

Transformer 层输出: [seq_len, dim] 的向量
需要转换成: [seq_len, vocab_size] 的概率

【结构】

输入: [seq_len, dim]
    ↓
Linear: [dim, vocab_size]
    ↓
Logits: [seq_len, vocab_size]
    ↓
Softmax
    ↓
概率分布: [seq_len, vocab_size]
    (每个位置对每个词的概率)

【为什么需要输出层？】

Transformer 学习的是"语义表示"
输出层把它映射回"词汇空间"

类比：
- Transformer: 理解问题
- 输出层: 选择回答的词语
""")

# 实现输出层
vocab_size = 1000  # 简化的词汇表

class OutputLayer(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        return: [batch, seq_len, vocab_size]
        """
        logits = self.linear(x)  # [batch, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)  # 概率分布
        return logits, probs

print(f"\n【输出层】")
print(f"输入维度: {dim}")
print(f"词汇表大小: {vocab_size}")

output_layer = OutputLayer(dim, vocab_size)
logits, probs = output_layer(x)

print(f"\n输入: {x.shape}")
print(f"Logits: {logits.shape}")
print(f"概率分布: {probs.shape}")
print(f"概率和 (应该≈1): {probs[0, 0].sum():.6f}")

# 查看最高概率的词
print(f"\n【第1个位置的最高概率词】")
top5_probs, top5_indices = torch.topk(probs[0, 0], 5)
print(f"Top 5 概率: {top5_probs.tolist()}")
print(f"Top 5 词索引: {top5_indices.tolist()}")

# ============================================
# 第五部分：Softmax 和温度参数
# ============================================

print("\n" + "=" * 70)
print("第五部分：Softmax 和温度参数")
print("=" * 70)

print("""
【Softmax 公式】

softmax(x_i) = exp(x_i) / sum(exp(x_j))

作用:
1. 把任意数值变成概率 (0-1之间)
2. 所有概率和为1
3. 放大差异（大的更大，小的更小）

【温度参数 (Temperature)】

softmax(x_i / T)

T > 1: 分布更平滑（随机性高，创造性）
T = 1: 标准分布
T < 1: 分布更尖锐（确定性高，保守）
""")

# 可视化温度影响
logits_demo = torch.tensor([2.0, 1.0, 0.5, 0.1, -0.5])
temperatures = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, T in enumerate(temperatures):
    probs_temp = F.softmax(logits_demo / T, dim=0)
    
    axes[idx].bar(range(len(logits_demo)), probs_temp.numpy())
    axes[idx].set_title(f'Temperature = {T}', fontsize=12)
    axes[idx].set_xlabel('词索引', fontsize=10)
    axes[idx].set_ylabel('概率', fontsize=10)
    axes[idx].set_ylim(0, 1)
    
    # 显示最高概率
    max_idx = probs_temp.argmax()
    axes[idx].text(max_idx, probs_temp[max_idx] + 0.05, 
                   f'{probs_temp[max_idx]:.2f}', 
                   ha='center', fontsize=10, fontweight='bold')

plt.suptitle('温度参数对概率分布的影响', fontsize=14)
plt.tight_layout()
plt.savefig('temperature_effect.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'temperature_effect.png'")

# ============================================
# 第六部分：生成下一个 Token
# ============================================

print("\n" + "=" * 70)
print("第六部分：生成下一个 Token")
print("=" * 70)

print("""
【生成过程】

输入: "我爱学"
    ↓
模型预测下一个词的概率:
    "习": 0.6
    "习": 0.2
    "习": 0.1
    ...
    
【采样策略】

1. Greedy (贪心):
   总是选择概率最高的词
   缺点: 缺乏多样性

2. Random Sampling:
   按概率随机选择
   缺点: 可能选到低概率的奇怪词

3. Top-k Sampling:
   只从概率最高的 k 个词中选择
   平衡质量和多样性

4. Top-p (Nucleus) Sampling:
   从累积概率达到 p 的最小集合中选择
   更灵活
""")

# 模拟生成
def generate_next_token(logits, strategy='greedy', temperature=1.0, top_k=5):
    """
    生成下一个token
    """
    logits = logits / temperature
    
    if strategy == 'greedy':
        # 贪心：选概率最高的
        return torch.argmax(logits).item()
    
    elif strategy == 'top_k':
        # Top-k 采样
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        # 重新归一化
        top_k_probs = top_k_probs / top_k_probs.sum()
        # 采样
        sampled_idx = torch.multinomial(top_k_probs, 1).item()
        return top_k_indices[sampled_idx].item()
    
    elif strategy == 'random':
        # 随机采样
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

# 测试不同策略
print("\n【不同采样策略对比】")
last_token_logits = logits[0, -1, :]  # 最后一个位置的logits

print(f"Logits 形状: {last_token_logits.shape}")
print(f"Top 5 词: {torch.topk(last_token_logits, 5).indices.tolist()}")

strategies = ['greedy', 'top_k', 'random']
for strategy in strategies:
    tokens = []
    for _ in range(5):  # 生成5次看随机性
        token = generate_next_token(last_token_logits, strategy=strategy)
        tokens.append(token)
    print(f"{strategy:10s}: {tokens}")

# ============================================
# 第七部分：完整的 Llama3 模型
# ============================================

print("\n" + "=" * 70)
print("第七部分：完整的 Llama3 模型")
print("=" * 70)

class SimpleLlama3(nn.Module):
    """简化的 Llama3 模型"""
    def __init__(self, vocab_size, dim, n_heads, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(dim, n_heads) 
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(dim, vocab_size)
        
    def forward(self, tokens):
        """
        tokens: [batch, seq_len]
        return: logits [batch, seq_len, vocab_size]
        """
        # 嵌入
        x = self.embedding(tokens)  # [batch, seq_len, dim]
        
        # 通过所有 Transformer 层
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        logits = self.output(x)  # [batch, seq_len, vocab_size]
        
        return logits

# 创建模型
vocab_size = 1000
dim = 64
n_heads = 4
n_layers = 4

print(f"\n【创建 Llama3 模型】")
print(f"词汇表: {vocab_size}")
print(f"维度: {dim}")
print(f"头数: {n_heads}")
print(f"层数: {n_layers}")

model = SimpleLlama3(vocab_size, dim, n_heads, n_layers)

# 计算参数量
def count_params(model):
    return sum(p.numel() for p in model.parameters())

total_params = count_params(model)
print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# 测试
print("\n【测试模型】")
tokens = torch.randint(0, vocab_size, (1, 10))  # [batch=1, seq_len=10]
print(f"输入 tokens: {tokens.shape}")

logits = model(tokens)
print(f"输出 logits: {logits.shape}")

probs = F.softmax(logits, dim=-1)
print(f"输出概率: {probs.shape}")
print(f"概率和: {probs[0, 0].sum():.6f}")

# ============================================
# 第八部分：Llama3-8B 的规模
# ============================================

print("\n" + "=" * 70)
print("第八部分：Llama3-8B 的规模")
print("=" * 70)

print("""
【Llama3-8B 配置】

词汇表: 128,256
维度: 4,096
头数: 32
KV头数: 8
层数: 32
FFN隐藏维度: 11,008

【参数量分解】

1. 嵌入层:
   128,256 × 4,096 = 525,336,576 (~525M)

2. 32层 Transformer:
   每层: ~177M
   总计: ~5.7B

3. 输出层:
   4,096 × 128,256 = 525,336,576 (~525M)

总参数量: ~8B (80亿)

【为什么叫 8B？】

总参数量 ≈ 8 billion (80亿)

【内存占用】

bf16 (2字节/参数):
  8B × 2 = 16 GB

int8 (1字节/参数):
  8B × 1 = 8 GB

int4 (0.5字节/参数):
  8B × 0.5 = 4 GB
""")

# 计算 Llama3 规模
llama_vocab = 128256
llama_dim = 4096
llama_heads = 32
llama_kv_heads = 8
llama_layers = 32
llama_hidden = 11008

# 嵌入层
params_embedding = llama_vocab * llama_dim

# 每层
params_attn_wq = llama_dim * llama_dim
params_attn_wk = llama_dim * (llama_kv_heads * llama_dim // llama_heads)
params_attn_wv = llama_dim * (llama_kv_heads * llama_dim // llama_heads)
params_attn_wo = llama_dim * llama_dim
params_ffn_w1 = llama_dim * llama_hidden
params_ffn_w3 = llama_dim * llama_hidden
params_ffn_w2 = llama_hidden * llama_dim
params_norm = llama_dim * 2  # 两个 RMS Norm

params_per_layer = (params_attn_wq + params_attn_wk + params_attn_wv + 
                    params_attn_wo + params_ffn_w1 + params_ffn_w3 + 
                    params_ffn_w2 + params_norm)

# 输出层
params_output = llama_dim * llama_vocab

# 总计
total_llama = params_embedding + params_per_layer * llama_layers + params_output

print(f"\n【Llama3-8B 参数量计算】")
print(f"嵌入层: {params_embedding:,} ({params_embedding/1e9:.2f}B)")
print(f"每层: {params_per_layer:,} ({params_per_layer/1e6:.1f}M)")
print(f"32层: {params_per_layer * llama_layers:,} ({params_per_layer * llama_layers/1e9:.2f}B)")
print(f"输出层: {params_output:,} ({params_output/1e9:.2f}B)")
print(f"总计: {total_llama:,} ({total_llama/1e9:.2f}B)")

# ============================================
# 第九部分：总结
# ============================================

print("\n" + "=" * 70)
print("第九部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. 为什么需要多层:
   - 逐层抽象
   - 浅层: 局部特征
   - 深层: 语义推理
   - Llama3: 32层

2. 多层堆叠:
   - 每层输入输出形状相同
   - 残差连接保证梯度流通
   - 可以堆叠很深

3. 输出层:
   - Linear: [dim, vocab_size]
   - Softmax: 概率分布
   - 生成下一个token

4. 采样策略:
   - Greedy: 确定性
   - Top-k: 平衡
   - Temperature: 控制随机性

5. 完整模型:
   输入 → 嵌入 → 32层Transformer → 输出层 → 概率

【完整流程回顾】

Token IDs
    ↓
Embedding [vocab, dim]
    ↓
Layer 1: Attention + FFN
    ↓
Layer 2: Attention + FFN
    ↓
...
    ↓
Layer 32: Attention + FFN
    ↓
Output Linear [dim, vocab]
    ↓
Softmax
    ↓
概率分布 → 采样 → 下一个Token

【下一步】

阶段 10: 整合与总结
回顾整个 Llama3 架构！
""")

print("\n" + "=" * 70)
print("阶段 9 完成！")
print("=" * 70)
