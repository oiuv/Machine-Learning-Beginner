# 09-02: MoE基础 - 门控网络与专家网络

## MoE的两大核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                        MoE Layer                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   门控网络 (Gating Network)                                 │
│   - 决定哪个专家被调用                                     │
│   - 输入：hidden states                                    │
│   - 输出：每个专家的权重                                   │
│                                                             │
│   专家网络 (Expert Network)                                 │
│   - 执行实际的计算                                         │
│   - 通常是FFN层                                            │
│   - 每个专家独立权重                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. 门控网络 (Gating Network)

### 结构

```python
class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        # 门控矩阵：将hidden_dim映射到num_experts维度
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        # 输出: [batch, seq_len, num_experts] 每个专家的分数
        return self.gate(x)
```

### 计算过程

```
输入: x (hidden state)
   │
   ↓
gate(x) = x @ W_gate
   │       (线性变换)
   ↓
[score_1, score_2, ..., score_N]  (N个专家的原始分数)
   │
   ↓
softmax(scores)  (归一化为概率)
   │
   ↓
[prob_1, prob_2, ..., prob_N]  (每个专家被选中的概率)
```

### 例子

```python
# 假设有8个专家
hidden_dim = 768
num_experts = 8

gate = GatingNetwork(768, 8)
x = torch.randn(1, 1, 768)  # 一个token的hidden state

scores = gate(x)
# scores: [1, 1, 8] = [0.5, 0.1, 0.3, 0.05, 0.02, 0.01, 0.01, 0.01]

probs = F.softmax(scores, dim=-1)
# probs: [0.29, 0.14, 0.21, 0.13, 0.12, 0.05, 0.03, 0.03]
# 总和 = 1.0
```

## 2. 专家网络 (Expert Network)

### 结构

每个专家本质上是一个FFN（与标准Transformer中的FFN相同）：

```python
class Expert(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

### FFN标准结构

```
input (768)
    │
    ↓
Linear(768 → 3072)  # 扩展维度
    │
    ↓
  GELU激活
    │
    ↓
Linear(3072 → 768)  # 压缩回原维度
    │
    ↓
output (768)
```

### 为什么专家是FFN？

| 结构 | 作用 | 原因 |
|------|------|------|
| Attention | 捕捉序列关系 | Self-Attention |
| **FFN** | **非线性变换** | **每个位置的独立变换** |

MoE替换的是FFN部分，因为：
1. FFN占Transformer参数的大部分
2. FFN的运算是逐位置的，不需要跨位置交互

## 3. MoE Layer的组合

```python
class MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts, ffn_dim):
        super().__init__()
        self.gate = GatingNetwork(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) 
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        
        # 1. 计算门控分数
        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch, seq_len, num_experts]
        
        # 2. 选择Top-K专家
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=2, dim=-1)
        
        # 3. 对选中的专家加权求和
        # ... (详见Top-K路由章节)
```

## 4. 完整的MoE Transformer Block

```python
class MoETransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim)
        self.moe = MoELayer(hidden_dim, num_experts, top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Self-Attention + 残差连接
        x = x + self.attention(self.norm1(x))
        
        # MoE + 残差连接
        x = x + self.moe(self.norm2(x))
        
        return x
```

## 5. 密集型MoE vs 稀疏型MoE

### 密集型 (Dense MoE)

```python
# 每个专家都参与计算，然后加权求和
for expert in experts:
    outputs.append(expert(x))
output = sum(outputs) / num_experts
```

### 稀疏型 (Sparse MoE) - 常用

```python
# 只选择Top-K个专家参与计算
top_k_probs, top_k_indices = torch.topk(gate_probs, k=2)
output =加权求和(selected_experts)
```

| 类型 | 计算量 | 效果 | 常用场景 |
|------|--------|------|----------|
| Dense | 大（全部计算） | 好 | 小规模MoE |
| **Sparse** | **小（只算Top-K）** | **相当** | **大规模MoE** |

## 总结

### 门控网络
- 决定哪些专家被调用
- 输出每个专家的权重
- 是MoE的"大脑"

### 专家网络
- 执行实际计算
- 通常是FFN结构
- 每个专家独立权重

### 稀疏激活
- 只选择Top-K专家
- 大幅降低计算量
- 效果与Dense相当

---

## 下一步

学习 [09-03: Top-K路由机制](./03_topk_routing.md)
