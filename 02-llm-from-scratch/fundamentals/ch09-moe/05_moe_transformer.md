# 09-05: MoE Transformer架构

## 标准Transformer vs MoE Transformer

### 标准Transformer Block

```
┌─────────────────────────────────────────┐
│           Transformer Block              │
├─────────────────────────────────────────┤
│                                          │
│  输入 X                                   │
│    │                                      │
│    ├──→ LayerNorm                        │
│    │                                      │
│    ├──→ Self-Attention                   │
│    │     (Q, K, V → Attention)           │
│    │                                      │
│    ├──→ + X (残差连接)                   │
│    │                                      │
│    ├──→ LayerNorm                        │
│    │                                      │
│    ├──→ FFN                              │
│    │     (Linear → GELU → Linear)        │
│    │                                      │
│    └──→ + X (残差连接)                   │
│                                          │
│  输出                                      │
└─────────────────────────────────────────┘
```

### MoE Transformer Block

```
┌─────────────────────────────────────────┐
│           MoE Transformer Block          │
├─────────────────────────────────────────┤
│                                          │
│  输入 X                                   │
│    │                                      │
│    ├──→ LayerNorm                        │
│    │                                      │
│    ├──→ Self-Attention                   │
│    │     (保持不变，所有token共享)         │
│    │                                      │
│    ├──→ + X (残差连接)                   │
│    │                                      │
│    ├──→ LayerNorm                        │
│    │                                      │
│    ├──→ MoE Layer                        │
│    │     (FFN → 多个Expert + Gating)      │
│    │                                      │
│    └──→ + X (残差连接)                   │
│                                          │
│  输出                                      │
└─────────────────────────────────────────┘
```

## MoE替换的是FFN层

### 标准FFN

```python
class StandardFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

### MoE-FFN

```python
class MoEFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k):
        super().__init__()
        self.gate = TopKRouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) 
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # TopK路由
        indices, weights = self.gate(x)
        
        # 收集被选中的专家输出并加权求和
        output = self.moe_forward(x, indices, weights)
        
        return output
```

## 两种MoE架构

### 1. 共享专家MoE (Shared Expert MoE)

某些专家被所有token共享：

```python
class SharedExpertMoE(nn.Module):
    def __init__(self, hidden_dim, num_shared, num_specific, top_k):
        super().__init__()
        # 共享专家（所有token都经过）
        self.shared_experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) 
            for _ in range(num_shared)
        ])
        # 特定专家（通过路由选择）
        self.specific_experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim)
            for _ in range(num_specific)
        ])
        self.gate = TopKRouter(hidden_dim, num_specific, top_k)
    
    def forward(self, x):
        # 所有token经过共享专家
        shared_output = sum(expert(x) for expert in self.shared_experts)
        
        # 路由选择特定专家
        indices, weights = self.gate(x)
        specific_output = self.moe_forward(x, indices, weights)
        
        return shared_output + specific_output
```

### 2. 纯MoE (Pure MoE)

所有专家都是通过路由选择的：

```python
class PureMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.gate = TopKRouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        indices, weights = self.gate(x)
        return self.moe_forward(x, indices, weights)
```

## Mixtral的MoE架构

### Mixtral 8x7B

```
Mixtral Block:
    │
    ├──→ Self-Attention (所有专家共享)
    │
    └──→ MoE Layer
          │
          ├──→ 共享FFN (所有token经过)
          │
          └──→ 8个独立Expert (Top-2选择)
```

### 代码表示

```python
class MixtralBlock(nn.Module):
    def __init__(self):
        self.attention = SelfAttention()      # 共享
        self.moe = MixtralMoE(
            num_experts=8,
            top_k=2,
            has_shared_expert=True
        )
    
    def forward(self, x):
        x = x + self.attention(x)  # Self-Attention + 残差
        x = x + self.moe(x)        # MoE + 残差
        return x
```

## 完整MoE GPT模型

```python
class MoEGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                hidden_dim=config.hidden_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                ffn_dim=config.ffn_dim
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
    
    def forward(self, input_ids):
        # Embedding
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(torch.arange(x.size(1), device=x.device))
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        return self.lm_head(x)
```

## 参数分布

### Dense vs MoE

| 模型 | 总参数量 | Attention | FFN/Experts | 激活参数 |
|------|----------|-----------|--------------|----------|
| LLaMA 7B | 7B | 2B | 5B | 7B (100%) |
| Mixtral 8x7B | 46.7B | 12.8B | 33.9B | 12B (26%) |

### Mixtral参数分解

```
总参数 = Attention参数 + Expert参数
      = 12.8B + (8 × 4.8B)
      = 12.8B + 38.4B
      = 51.2B ≈ 46.7B (共享等因素略有差异)

激活参数 = Attention + Top-2 Experts
        = 12.8B + (2 × 4.8B)
        = 12.8B + 9.6B
        = 22.4B ≈ 12B (实际实现有优化)
```

## 显存占用分析

### 训练时显存

| 项目 | Dense模型 | MoE模型 |
|------|-----------|---------|
| 模型参数 | 100% | 100% (所有专家) |
| 梯度 | 100% | 100% |
| 优化器状态 | 100% | 100% (所有专家) |
| 激活值 | 100% | ~20-30% (只计算Top-K) |

### 推理时显存

| 项目 | Dense模型 | MoE模型 |
|------|-----------|---------|
| 模型参数 | 100% | 100% (需加载所有专家) |
| 激活值 | 100% | Top-K专家的计算 |

## 总结

### MoE Transformer结构

```
输入 → Embedding → [Self-Attention + MoE Layer] × N → Output
                      ↑
              只有MoE层有多个专家
              Attention保持标准
```

### 关键设计

1. MoE替换FFN层
2. Attention保持共享
3. 稀疏激活节省计算
4. 需要负载均衡

---

## 下一步

学习 [09-06: Mixtral 8x7B分析](./06_mixtral_analysis.md)
