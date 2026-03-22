# 09-06: Mixtral 8x7B 架构分析

## Mixtral简介

Mixtral 8x7B 是由Mistral AI发布的稀疏专家混合模型，被认为是"GPT-4级别"的开源模型。

| 规格 | 值 |
|------|-----|
| 发布时间 | 2023年12月 |
| 总参数量 | 46.7B |
| 专家数量 | 8 |
| 每个专家大小 | ~5.8B |
| 激活参数量 | ~12B |
| 上下文长度 | 32K |
| 许可协议 | Apache 2.0 |

## 架构概览

```
Mixtral 8x7B
┌─────────────────────────────────────────────────────────────┐
│                      32层 MixtralBlock                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  每层结构:                                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ LayerNorm                                              │ │
│  │    ↓                                                   │ │
│  │ Self-Attention (所有token共享)                         │ │
│  │    ↓                                                   │ │
│  │ + Residual                                             │ │
│  │    ↓                                                   │ │
│  │ LayerNorm                                              │ │
│  │    ↓                                                   │ │
│  │ MoE Layer:                                             │ │
│  │    Gate (Top-2)                                        │ │
│  │    ├── Expert 1 (5.8B) ─────────────────┐              │ │
│  │    ├── Expert 2 (5.8B) ─────────────────┼─→ 加权求和   │ │
│  │    ├── ...                             │              │ │
│  │    └── Expert 8 (5.8B) ────────────────┘              │ │
│  │    ↓                                                   │ │
│  │ + Residual                                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 与LLaMA 2 7B的对比

| 对比项 | LLaMA 2 7B | Mixtral 8x7B |
|--------|-------------|---------------|
| 总参数量 | 7B | 46.7B |
| 激活参数 | 7B (100%) | 12B (26%) |
| 层数 | 32 | 32 |
| 隐藏维度 | 4096 | 4096 |
| Attention头数 | 32 | 32 |
| FFN中间维度 | 11008 | 14336 |
| 专家数 | 1 (Dense) | 8 |
| 上下文长度 | 4K | 32K |

## 计算量对比

### 推理时的计算量

```
LLaMA 7B: 每层 = Self-Attn + FFN = 7B参数参与计算

Mixtral 8x7B: 每层 = Self-Attn + Top-2 Experts
            = Self-Attn + 2 × 5.8B
            ≈ 12B参数参与计算 (比LLaMA多70%)
```

### 但存储量

```
LLaMA 7B: 需要存储 7B 参数

Mixtral 8x7B: 需要存储 46.7B 参数
             (但每个GPU只需加载一次)
```

## 专家路由示例

```python
# Mixtral的Top-2路由示例

输入: "如何用Python写一个快速排序算法？"

专家选择:
├── 专家3 (编程)     权重: 0.72  ← 主要
├── 专家5 (数学/算法) 权重: 0.23  ← 辅助
├── 其他专家         权重: <0.05  ← 不参与
└── 共享FFN         权重: 1.00   ← 始终参与

输出 = 0.72 × Expert3(x) + 0.23 × Expert5(x) + SharedFFN(x)
```

## 共享专家机制

Mixtral包含一个**共享专家(Shared Expert)**，所有token都会经过：

```python
class MixtralMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        # 共享专家（始终参与）
        self.shared_expert = Expert(hidden_dim, ffn_dim)
        
        # 路由专家（稀疏激活）
        self.gate = TopKRouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) 
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 共享专家输出（始终计算）
        shared_output = self.shared_expert(x)
        
        # Top-K专家输出（稀疏）
        indices, weights = self.gate(x)
        expert_output = self.moe_forward(x, indices, weights)
        
        return shared_output + expert_output
```

## 性能基准

### 基准测试结果

| 基准 | LLaMA 2 7B | LLaMA 2 13B | Mixtral 8x7B | GPT-3.5 |
|------|-------------|-------------|-------------|---------|
| MMLU | 68.5 | 75.0 | **70.7** | 70.0 |
| HumanEval | 29.9 | 38.1 | **40.2** | 48.1 |
| GSM8K | 56.1 | 59.8 | **58.4** | 57.1 |

### 关键洞察

1. **效率提升**: 12B激活参数 vs 70B参数的LLaMA 2 70B
2. **效果超越**: Mixtral 8x7B 接近或超越 LLaMA 2 13B
3. **开源优势**: Apache 2.0许可，可商用

## 训练数据

Mixtral的训练数据：

| 数据源 | 描述 |
|--------|------|
| Web数据 | 去重和清洗的网页文本 |
| 代码 | 代码数据集 |
| 数学 | 数学论文和教科书 |
| 英语 | 主要语言 |
| 其他 | 多语言数据 |

## 使用Mixtral

### 通过HuggingFace加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自动将不同层分配到不同GPU
    torch_dtype=torch.float16,
)

# 推理
input_text = "如何学习深度学习？"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### 显存需求

| 精度 | 单GPU显存 | 说明 |
|------|-----------|------|
| FP16 | ~48GB | 需要A100 |
| INT8 | ~24GB | 需要A10或双卡 |
| INT4 | ~12GB | 可在高端消费级GPU运行 |

## 总结

### Mixtral的核心设计

| 设计 | 实现 |
|------|------|
| 稀疏激活 | Top-2路由 |
| 共享专家 | 所有token经过shared Expert |
| Attention | 所有token共享 |
| 层数 | 32层 |
| 专家FFN维度 | 14336 |

### 为什么Mixtral效果好？

1. **专业分工**: 不同专家学习不同类型的知识
2. **稀疏激活**: 每次只激活部分专家，保持效率
3. **共享专家**: 保证基础能力的传递
4. **规模效应**: 46.7B总参数，但计算成本只有12B

---

## 下一步

学习 [09-07: MoE代码实现](./07_implementation.md)
