# 09-03: Top-K路由机制

## 为什么需要Top-K？

### 全选的问题

如果每次都让所有专家参与计算：

```python
# 所有专家都计算
for expert in experts:  # 8个专家
    outputs.append(expert(x))
output = weighted_sum(outputs)
```

| 问题 | 影响 |
|------|------|
| 计算量大 | 失去MoE优势 |
| 内存占用高 | 每个专家都要保存中间结果 |
| 扩展性差 | 专家越多，计算越多 |

### Top-K的解决方案

**只选择最相关的K个专家参与计算**

```python
# 只选Top-2专家
top_k_probs, top_k_indices = torch.topk(gate_probs, k=2)
# 只计算这2个专家
```

## Top-K路由计算过程

### 完整流程

```
Step 1: 输入
─────────────────────────────────
x: [batch, seq_len, hidden_dim]
    │
    ↓
Step 2: 门控计算
─────────────────────────────────
gate_logits = gate(x)  # [batch, seq_len, num_experts]
gate_probs = softmax(gate_logits, dim=-1)  # [batch, seq_len, num_experts]
    │
    ↓
Step 3: Top-K选择
─────────────────────────────────
top_k_probs, top_k_indices = torch.topk(gate_probs, k=2)
# top_k_probs: [batch, seq_len, 2]  选中的概率
# top_k_indices: [batch, seq_len, 2] 选中的专家索引
    │
    ↓
Step 4: 归一化概率
─────────────────────────────────
top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    │
    ↓
Step 5: 加权求和
─────────────────────────────────
output = sum(top_k_probs[i] * expert_i(x))
```

### 具体例子

```python
import torch
import torch.nn.functional as F

# 假设有8个专家，1个token
num_experts = 8
hidden_dim = 768
k = 2

# Step 1: 门控分数 (模拟)
gate_logits = torch.tensor([2.0, 0.5, 1.5, 0.1, 0.2, 0.1, 0.1, 0.1])
gate_probs = F.softmax(gate_logits, dim=0)

print("原始概率:")
print(gate_probs)
# tensor([0.70, 0.016, 0.040, 0.011, 0.012, 0.011, 0.011, 0.011])

# Step 2: Top-2选择
top_k_probs, top_k_indices = torch.topk(gate_probs, k=2)
print("\nTop-2专家:")
print(f"索引: {top_k_indices}")  # [0, 2]
print(f"概率: {top_k_probs}")     # [0.70, 0.040]

# Step 3: 归一化
top_k_probs = top_k_probs / top_k_probs.sum()
print(f"\n归一化后: {top_k_probs}")  # [0.945, 0.055]

# Step 4: 含义
# 专家0被选中概率94.5%
# 专家2被选中概率5.5%
```

## Top-K实现代码

```python
import torch
import torch.nn.functional as F
import torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x):
        """
        x: [batch, seq_len, hidden_dim]
        returns: 
            - selected_experts: 选中的专家索引 [batch, seq_len, top_k]
            - weights: 归一化权重 [batch, seq_len, top_k]
        """
        # 1. 计算门控分数
        gate_logits = self.gate(x)  # [B, L, E]
        
        # 2. Softmax转概率
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 3. Top-K选择
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        
        # 4. 归一化权重
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_indices, top_k_probs
```

## 处理批量数据

```python
def moe_forward(x, experts, top_k_indices, top_k_weights):
    """
    x: [batch, seq_len, hidden_dim]
    experts: List of Expert modules
    top_k_indices: [batch, seq_len, top_k]
    top_k_weights: [batch, seq_len, top_k]
    """
    batch_size, seq_len, hidden_dim = x.shape
    num_selected = top_k_indices.shape[-1]
    
    # 展平以便批量处理
    x_flat = x.view(-1, hidden_dim)  # [B*L, H]
    indices_flat = top_k_indices.view(-1, num_selected)  # [B*L, top_k]
    weights_flat = top_k_weights.view(-1, num_selected)  # [B*L, top_k]
    
    # 初始化输出
    output = torch.zeros_like(x_flat)
    
    # 对每个token，选择top_k专家加权求和
    for i in range(num_selected):
        expert_idx = indices_flat[:, i]  # 这个token选择的专家
        weight = weights_flat[:, i]       # 这个专家的权重
        
        # 计算被选中的专家输出
        for j, expert in enumerate(experts):
            # 找出这个token是选择专家j的
            mask = (expert_idx == j)
            if mask.any():
                expert_output = expert(x_flat[mask])
                # 加权累加
                output[mask] += weight[mask].unsqueeze(-1) * expert_output
    
    # 恢复形状
    return output.view(batch_size, seq_len, hidden_dim)
```

## Top-K的K如何选择？

| K值 | 计算量 | 效果 | 适用场景 |
|-----|--------|------|----------|
| 1 | 最低 | 可能不够 | 极端稀疏 |
| 2 | 低 | 常用 | 平衡 |
| 4 | 中等 | 较好 | 大模型 |
| 8+ | 较高 | 好 | MoE with shared experts |

### K=1的问题

```python
# K=1意味着只有一个专家被选中
# 风险：单一路由可能不够稳定

top_k_probs, top_k_indices = torch.topk(gate_probs, k=1)
# 只有一个专家，100%权重
# 如果选错专家，没有其他专家补充
```

### K=2的优点

```python
# K=2是常用选择
# 优点：有一定多样性，同时计算量不大

# 专家0: 85% + 专家2: 12% = 97%的信息
# 专家1: 3% 被忽略
```

## 路由的随机性

### 带噪声的路由

```python
class NoisyTopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.noise = nn.Linear(hidden_dim, num_experts)  # 添加噪声
    
    def forward(self, x):
        # 原始分数
        gate_logits = self.gate(x)
        
        # 添加噪声（训练时使用）
        if self.training:
            noise_logits = self.noise(x)
            noise = torch.randn_like(gate_logits) * F.softplus(noise_logits)
            gate_logits = gate_logits + noise
        
        # Softmax + TopK
        gate_probs = F.softmax(gate_logits, dim=-1)
        return torch.topk(gate_probs, k=self.top_k)
```

### 噪声的作用

| 阶段 | 噪声效果 |
|------|----------|
| 训练时 | 探索不同专家组合，避免过早收敛到单一专家 |
| 推理时 | 关闭噪声，保证确定性输出 |

## 总结

### Top-K路由流程

```
门控分数 → Softmax → Top-K选择 → 归一化权重 → 加权求和
```

### 关键点

- K控制稀疏程度：K越小越稀疏
- 归一化保证权重和为1
- 噪声有助于训练稳定性
- K=2是常用选择

---

## 下一步

学习 [09-04: 负载均衡](./04_load_balancing.md)
