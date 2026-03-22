# 垃圾短信分类器

基于预训练 GPT-2 的垃圾短信分类项目。

---

## 项目成果

| 指标 | 数值 |
|------|------|
| **准确率** | 96.7% |
| **精确率** | 87.3% |
| **召回率** | 87.9% |
| **F1 分数** | 87.6% |

---

## 技术要点

1. **预训练模型**: GPT-2 (124M 参数)
2. **微调策略**: 冻结 GPT-2，只训练分类头 (0.16% 参数)
3. **解决数据不平衡**: 使用类别权重（垃圾短信权重 6.5x）
4. **评估指标**: F1 分数（比准确率更适合不平衡数据）

---

## 文件结构

```
spam-classifier/
├── README.md              # 本文件
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── SMSSpamCollection      # 数据集（自动下载）
└── model/
    └── best_model.pt      # 训练好的模型
```

---

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程：
- 自动下载数据集（5572 条短信）
- 训练 5 个 epoch
- 保存最佳模型到 `model/best_model.pt`

### 2. 预测

```bash
# 测试模式
python predict.py

# 预测单条短信
python predict.py --text "WINNER!! You won $1000!"
```

---

## 核心代码

### 模型结构

```python
class SpamClassifier(nn.Module):
    def __init__(self):
        # 预训练 GPT-2
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2类: 正常/垃圾
        )
        
        # 冻结 GPT-2
        for param in self.gpt2.parameters():
            param.requires_grad = False
```

### 类别权重

```python
# 解决数据不平衡
class_weights = torch.tensor([0.58, 3.73])  # 正常/垃圾

# 垃圾短信错误惩罚 6.5x 更重
loss = F.cross_entropy(logits, labels, weight=class_weights)
```

---

## 学到的知识

1. **微调** = 预训练模型 + 少量有标签数据
2. **冻结参数** = 保留预训练知识，防止过拟合
3. **类别权重** = 解决数据不平衡问题
4. **F1 分数** = 精确率和召回率的调和平均

---

## 数据集

**SMS Spam Collection**
- 来源: UCI Machine Learning Repository
- 规模: 5,572 条短信
- 分布: 86.6% 正常, 13.4% 垃圾

---

## 依赖

```
torch
transformers
pandas
scikit-learn
tqdm
requests
```

安装：
```bash
pip install torch transformers pandas scikit-learn tqdm requests
```

---

## 训练输出示例

```
Epoch 1/5
  Train Loss: 0.8445, Train Acc: 0.6379
  Val Acc: 0.8816, Precision: 0.5311, Recall: 0.9732, F1: 0.6872

Epoch 5/5
  Train Loss: 0.2760, Train Acc: 0.9096
  Val Acc: 0.9668, Precision: 0.8733, Recall: 0.8792, F1: 0.8763
  ✓ 保存最佳模型

最佳 F1 分数: 0.8763
```

---

## 下一步

- [ ] 尝试更大的模型（gpt2-medium）
- [ ] 数据增强（同义词替换）
- [ ] 交叉验证
- [ ] 部署为 Web API

---

## 参考

- [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
