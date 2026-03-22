"""
第6章：微调（Fine-tuning）演示
展示如何使用预训练模型进行分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ===========================================
# GPT 组件（简化版，只用于演示）
# ===========================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GPTModel(nn.Module):
    """简化版 GPT 模型，只有 embedding 和输出层"""
    def __init__(self, vocab_size=50257, emb_dim=768):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(1024, emb_dim)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds
        return x


# ===========================================
# 分类器
# ===========================================
class TextClassifier(nn.Module):
    """
    文本分类器
    结构: Embedding → Pooling → Linear → Output
    """
    def __init__(self, vocab_size=50257, emb_dim=768, num_classes=2):
        super().__init__()
        
        # Embedding 层（可以用预训练权重）
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        output: (batch, num_classes)
        """
        # Embedding
        embeddings = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        
        # Pooling: 取最后一个 token（或平均）
        # 方法1: 最后一个 token
        pooled = embeddings[:, -1, :]  # (batch, emb_dim)
        
        # 方法2: 平均池化（可以试试）
        # pooled = embeddings.mean(dim=1)  # (batch, emb_dim)
        
        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits


# ===========================================
# 演示：微调
# ===========================================
def demo_finetuning():
    print("\n" + "="*60)
    print("微调演示：文本分类")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 模型参数
    vocab_size = 50257
    emb_dim = 768
    num_classes = 2  # 垃圾短信分类
    
    # 创建模型
    model = TextClassifier(vocab_size, emb_dim, num_classes).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    
    # 模拟数据
    print("\n模拟数据:")
    print("  0 = 正常短信 (ham)")
    print("  1 = 垃圾短信 (spam)")
    
    # 示例文本
    texts = [
        "Hey, are you coming tonight?",          # ham
        "WINNER!! You won $1000!",               # spam
        "Can you pick up some milk?",            # ham
        "FREE entry to WIN $5000 cash!",         # spam
        "See you at 7pm",                        # ham
        "URGENT: Your account has been compromised!",  # spam
    ]
    labels = [0, 1, 0, 1, 0, 1]
    
    # Tokenize
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = [tokenizer.encode(text) for text in texts]
    
    # 填充到相同长度
    max_len = max(len(e) for e in encoded)
    padded = [e + [50256] * (max_len - len(e)) for e in encoded]
    
    # 转为 tensor
    input_ids = torch.tensor(padded).to(device)
    labels_tensor = torch.tensor(labels).to(device)
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"标签: {labels_tensor.tolist()}")
    
    # 训练前
    print("\n" + "-"*60)
    print("训练前:")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels_tensor).float().mean()
    
    print(f"预测: {predictions.tolist()}")
    print(f"真实: {labels_tensor.tolist()}")
    print(f"准确率: {accuracy.item():.2%}")
    
    # 训练
    print("\n" + "-"*60)
    print("开始训练...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits, labels_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels_tensor).float().mean()
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.2%}")
    
    # 训练后
    print("\n" + "-"*60)
    print("训练后:")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels_tensor).float().mean()
    
    print(f"预测: {predictions.tolist()}")
    print(f"真实: {labels_tensor.tolist()}")
    print(f"准确率: {accuracy.item():.2%}")
    
    # 测试新数据
    print("\n" + "-"*60)
    print("测试新数据:")
    
    test_texts = [
        "Call me back ASAP",
        "Congratulations! You've won a prize!",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        padded = encoded + [50256] * (max_len - len(encoded))
        input_tensor = torch.tensor([padded]).to(device)
        
        with torch.no_grad():
            logit = model(input_tensor)
            pred = torch.argmax(logit, dim=-1).item()
        
        label_name = "垃圾短信" if pred == 1 else "正常短信"
        print(f"  '{text}' → {label_name}")


# ===========================================
# 演示：冻结策略
# ===========================================
def demo_freezing():
    print("\n" + "="*60)
    print("冻结策略演示")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = TextClassifier().to(device)
    
    print("\n策略1: 只训练分类头（冻结 embedding）")
    print("-"*60)
    
    # 冻结 embedding
    for param in model.embedding.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    print("\n策略2: 训练所有参数")
    print("-"*60)
    
    # 解冻
    for param in model.embedding.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,} / {total:,} (100.00%)")
    
    print("\n" + "-"*60)
    print("选择建议:")
    print("  数据少 → 策略1（只训练分类头，防过拟合）")
    print("  数据多 → 策略2（训练全部，效果更好）")
    print("-"*60)


# ===========================================
# 主程序
# ===========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("第6章：微调（Fine-tuning）")
    print("="*60)
    
    demo_finetuning()
    demo_freezing()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    
    print("\n关键要点:")
    print("1. 微调 = 预训练模型 + 少量有标签数据")
    print("2. 需要修改输出层用于分类任务")
    print("3. 可以选择冻结部分参数")
    print("4. 冻结越多，训练越快，但可能效果越差")
    print("\n下一步:")
    print("  - 下载 OpenAI GPT-2 预训练权重")
    print("  - 加载到我们的模型")
    print("  - 在真实数据集上微调")
