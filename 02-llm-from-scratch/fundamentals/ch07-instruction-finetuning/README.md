# 第7章：指令微调（Instruction Fine-tuning）

## 概述

**指令微调** = 让模型学会理解并执行人类指令

```
第6章（分类微调）: 短信 → "是垃圾" 或 "正常"
第7章（指令微调）: 指令 → 自由文本回复
```

---

## 对比

| 类型 | 输入 | 输出 | 例子 |
|------|------|------|------|
| **预训练** | 文本 | 下一个词 | "今天天气" → "很好" |
| **分类微调** | 文本 | 类别 | "中奖了" → "垃圾" |
| **指令微调** | 指令+输入 | 文本回复 | "翻译: Hello" → "你好" |

---

## 指令微调的作用

### 预训练模型的问题

```
用户: 请把"Hello"翻译成中文
模型: 请把"Hello"翻译成中文，请问您需要什么帮助？...
      ↑ 只是继续生成文本，不理解这是"翻译任务"
```

### 指令微调后

```
用户: 请把"Hello"翻译成中文
模型: 你好
      ↑ 理解了指令，执行翻译任务
```

---

## 核心概念

### 1. 指令数据格式

```json
{
  "instruction": "请把下面的句子翻译成中文",
  "input": "Hello world",
  "output": "你好世界"
}
```

### 2. Prompt Template（提示模板）

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
请把下面的句子翻译成中文

### Input:
Hello world

### Response:
你好世界
```

### 3. 训练目标

```
输入: "### Instruction: 翻译 Hello ### Input: Hello ### Response:"
目标: "你好世界"

损失 = 预测的词 vs 真实的词
```

---

## 指令微调流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 准备指令数据集                                          │
│     - 多种任务（翻译、摘要、问答等）                        │
│     - 统一格式（instruction + input + output）              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 格式化输入                                              │
│     添加 Prompt Template                                    │
│     "### Instruction: ... ### Input: ... ### Response:"     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 微调训练                                                │
│     用预训练 GPT-2 权重                                     │
│     在指令数据上继续训练                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 测试                                                    │
│     输入新指令 → 模型生成回复                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 代码实现

### 1. 指令数据集类

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            # 格式化输入
            instruction_text = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            
            # 拼接完整文本
            full_text = instruction_text + response_text
            
            # Tokenize
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, idx):
        return self.encoded_texts[idx]
```

### 2. 格式化输入

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    # 如果有输入，添加 input 部分
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    
    return instruction_text + input_text
```

### 3. 训练

```python
# 和预训练类似，但用指令数据
for batch in train_loader:
    input_ids = batch["input_ids"]
    targets = batch["targets"]
    
    logits = model(input_ids)
    loss = cross_entropy(logits, targets)
    
    loss.backward()
    optimizer.step()
```

---

## 数据整理（Collate Function）

### 问题：不同长度的序列

```
样本1: [1, 2, 3]           (长度3)
样本2: [4, 5, 6, 7, 8]     (长度5)
样本3: [9, 10]             (长度2)
```

### 解决：Padding + Mask

```python
def custom_collate_fn(batch):
    # 找到批次中最长的序列
    batch_max_length = max(len(item) for item in batch)
    
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        # Padding
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        
        # 输入：去掉最后一个token
        inputs = padded[:-1]
        
        # 目标：向右移一位
        targets = padded[1:]
        
        # 关键：padding部分不计算损失
        targets[targets == pad_token_id] = -100  # ignore_index
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    return torch.stack(inputs_lst), torch.stack(targets_lst)
```

### 为什么 targets 要移位？

```
输入:  "### Instruction: 翻译 ### Input: Hello ### Response:"
目标:  "### Instruction: 翻译 ### Input: Hello ### Response: 你"

下一步:
输入:  "... Response: 你"
目标:  "... Response: 你好"

再下一步:
输入:  "... Response: 你好"
目标:  "... Response: 你好世界"
```

**每个位置都预测下一个词！**

---

## 关键技术点

### 1. Masking Loss（只计算 Response 部分）

```python
# 指令部分不计算损失，只训练 Response 生成
if targets == pad_token_id:
    targets = -100  # PyTorch 会忽略 -100
```

### 2. 多任务训练

```
同时训练多种任务:
- 翻译
- 摘要
- 问答
- 改写
- 分类

→ 模型学会"理解指令"而非"记住特定任务"
```

### 3. Prompt Engineering

```python
# 好的 Prompt Template 很重要
template = """
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
```

---

## 与前几章的关系

```
第2-4章: 搭建 GPT 架构
第5章: 预训练（学会语言）
第6章: 分类微调（学会分类）
第7章: 指令微调（学会执行指令）← 你在这里
第8章: 偏好对齐（RLHF，学会人类偏好）
```

---

## 效果对比

### 预训练模型

```
输入: 请总结以下文本：今天天气很好...
输出: 今天天气很好，阳光明媚，温度适宜，非常适合外出...
      ↑ 只是继续生成，不是总结
```

### 指令微调后

```
输入: 请总结以下文本：今天天气很好...
输出: 天气晴朗，适合外出。
      ↑ 理解了"总结"指令
```

---

## 常见指令数据集

| 数据集 | 任务类型 | 规模 |
|--------|----------|------|
| Alpaca | 多任务 | 52K |
| Dolly | 多任务 | 15K |
| FLAN | 多任务 | 1.8M |
| Self-Instruct | 多任务 | 82K |

---

## Q&A 洞察

*（会在你提问后填充）*

---

## 下一步

- 第8章：人类反馈强化学习（RLHF）
- 让模型不仅执行指令，还要符合人类偏好

---

## 运行代码

```bash
cd learning/ch07-instruction-finetuning
python ch07_code.py
```
