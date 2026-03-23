# 09-04: 负载均衡

## 负载不均衡问题

### 问题描述

如果门控网络总是选择相同的专家：

```
理想情况：
专家1: 12.5% 被选中
专家2: 12.5% 被选中
专家3: 12.5% 被选中
...
专家8: 12.5% 被选中

实际情况：
专家1: 80%   被选中  ← 专家1过载
专家2: 10%
专家3: 5%
...
专家8: 1%    ← 专家8空闲
```

### 问题影响

| 影响 | 说明 |
|------|------|
| 计算瓶颈 | 专家1成为瓶颈，速度受限于它 |
| 专家浪费 | 部分专家几乎不被使用 |
| 训练不稳定 | 部分专家训练不足 |
| 效果下降 | 专业分工的优势消失 |

## Auxiliary Loss（辅助损失）

### 解决方案

添加一个辅助损失函数，惩罚负载不均衡：

```python
def load_balancing_loss(gate_probs, top_k_indices, num_experts):
    """
    鼓励每个专家被选中的概率接近 1/num_experts
    """
    # 方法：让gate概率的均值接近均匀分布
    
    # 计算每个专家被选中的频率
    expert_counts = torch.zeros(num_experts, device=gate_probs.device)
    
    for expert_idx in range(num_experts):
        # 统计有多少token选择了这个专家
        count = (top_k_indices == expert_idx).float().sum()
        expert_counts[expert_idx] = count
    
    # 计算平均概率
    avg_prob = expert_counts / expert_counts.sum()
    
    # 理想概率
    ideal_prob = 1.0 / num_experts
    
    # 计算差异（用均方误差）
    # 正确的辅助损失：使用均方误差让分布接近均匀
    # L = num_experts * sum((p_i - 1/num_experts)^2)
    loss = num_experts * ((avg_prob - ideal_prob) ** 2).sum()
    
    return loss
```

### 更简洁的写法

```python
def aux_load_balancing_loss(gate_probs, top_k_indices, num_experts, top_k):
    """
    论文中的标准实现
    """
    # gate_probs: [batch, seq_len, num_experts]
    # top_k_indices: [batch, seq_len, top_k]
    
    # 方法1：基于概率的辅助损失
    # 鼓励所有专家的概率分布均匀
    probs_mean = gate_probs.mean(dim=[0, 1])  # [num_experts]
    ideal_prob = 1.0 / num_experts
    
    # L = sum(probs_mean * ideal_prob) * num_experts
    # = num_experts * sum(p_i / num_experts)
    # = sum(p_i) = 1  (理想情况)
    # 注意：上面的公式是错误的！它总是返回1.0，无论分布如何
    # 正确的辅助损失使用均方误差：
    # L = num_experts * sum((p_i - 1/num_experts)^2)
    aux_loss = num_experts * ((probs_mean - ideal_prob) ** 2).sum()
    
    return aux_loss
```

### Auxiliary Loss的另一种形式

```python
def aux_load_balancing_loss_v2(fire_probs, top_k_indices, num_experts):
    """
    Switch Transformer使用的形式
    """
    # fire_probs: 每个专家被选中的频率
    fire_probs = fire_probs.mean(dim=[0, 1])  # [num_experts]
    fire_probs = fire_probs / fire_probs.sum()  # 归一化
    
    # 理想情况：每个专家被选中的概率相等
    ideal = 1.0 / num_experts
    
    # 辅助损失 = num_experts * 概率分布与均匀分布的差异（均方误差形式）
    aux_loss = num_experts * ((fire_probs - ideal) ** 2).sum()
    
    return aux_loss
```

## 完整的MoE训练损失

```python
def total_loss(main_loss, aux_loss, alpha=0.01):
    """
    main_loss: 主损失（如语言模型的交叉熵）
    aux_loss: 负载均衡辅助损失
    alpha: 辅助损失的权重（通常0.01-0.05）
    """
    return main_loss + alpha * aux_loss
```

### 辅助损失权重的影响

| alpha值 | 效果 |
|---------|------|
| 太小 (0.001) | 负载均衡不够 |
| 合适 (0.01) | 均衡且不影响主损失 |
| 太大 (0.1) | 主任务被干扰 |

##Expert选择频率监控

### 训练时监控

```python
class LoadBalancer:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.expert_counts = torch.zeros(num_experts)
        self.total_tokens = 0
    
    def update(self, top_k_indices):
        # 统计每个专家被选中的次数
        for idx in range(self.num_experts):
            count = (top_k_indices == idx).float().sum().item()
            self.expert_counts[idx] += count
        self.total_tokens += top_k_indices.numel()
    
    def get_rates(self):
        # 每个专家被选中的频率
        return self.expert_counts / self.total_tokens
    
    def print_stats(self):
        rates = self.get_rates()
        print("专家选择频率:")
        for i, rate in enumerate(rates):
            ideal = 1.0 / self.num_experts
            status = "✓" if abs(rate - ideal) < 0.05 else "✗"
            print(f"  专家{i}: {rate:.3f} (理想: {ideal:.3f}) {status}")
```

### 理想vs实际的例子

```python
# 训练前（不均衡）
专家选择频率:
专家0: 0.450 ████████████  ✗ (应该0.125)
专家1: 0.200 ██████        ✗
专家2: 0.150 ████          ✗
专家3: 0.100 ███           ✗
专家4: 0.050 █             ✗
专家5: 0.030 ▏            ✗
专家6: 0.015 ▎            ✗
专家7: 0.005 ▏            ✗

# 训练后（均衡）
专家选择频率:
专家0: 0.130 ████         ✓
专家1: 0.125 ████         ✓
专家2: 0.128 ████         ✓
专家3: 0.122 ████         ✓
专家4: 0.125 ████         ✓
专家5: 0.120 ████         ✓
专家6: 0.128 ████         ✓
专家7: 0.122 ████         ✓
```

## 其他负载均衡策略

### 1. 温度采样 (Temperature Sampling)

```python
def gate_with_temperature(logits, temperature=0.7):
    # 温度越高，分布越平滑（接近均匀）
    return F.softmax(logits / temperature, dim=-1)
```

### 2. 随机路由 (Random Routing)

```python
def random_gate(num_experts, top_k):
    # 以一定概率随机选择专家
    probs = torch.ones(num_experts) / num_experts
    return torch.multinomial(probs, top_k, replacement=False)
```

### 3. Expert Capacity

```python
def forward_with_capacity(x, experts, gate_probs, max_capacity_per_expert=1000):
    """
    限制每个专家的最大处理量
    """
    batch_size, seq_len, num_experts = gate_probs.shape
    
    # 计算每个专家当前被分配了多少token
    expert_load = torch.zeros(num_experts, device=x.device)
    
    outputs = []
    for i in range(num_experts):
        # 检查这个专家是否还有容量
        if expert_load[i] < max_capacity_per_expert:
            # 处理请求
            pass
    
    return outputs
```

## 总结

### 负载不均衡问题
- 门控可能总是选择相同专家
- 导致计算瓶颈和训练不稳定

### Auxiliary Loss解决方案
- 添加额外的损失函数
- 惩罚不均匀的专家选择
- 让每个专家被选中的频率接近 1/num_experts

### 关键参数
- alpha: 辅助损失权重（通常0.01）

---

## 下一步

学习 [09-05: MoE Transformer](./05_moe_transformer.md)
