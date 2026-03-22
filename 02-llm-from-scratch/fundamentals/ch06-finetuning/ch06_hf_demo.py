"""
第6章：使用 Hugging Face 加载 GPT-2 预训练权重
不需要 TensorFlow，只需要 transformers 库
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer


# ===========================================
# 1. 加载预训练 GPT-2
# ===========================================
def load_gpt2_from_hf(model_size="gpt2"):
    """
    从 Hugging Face 加载 GPT-2
    
    model_size 选项:
    - "gpt2"      : 124M 参数
    - "gpt2-medium": 355M 参数  
    - "gpt2-large" : 774M 参数
    - "gpt2-xl"    : 1.5B 参数
    """
    print(f"正在加载 {model_size}...")
    
    # 加载模型和分词器
    model = GPT2Model.from_pretrained(model_size)
    tokenizer = GPT2Tokenizer.from_pretrained(model_size)
    
    # 设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    
    return model, tokenizer


# ===========================================
# 2. GPT-2 分类器
# ===========================================
class GPT2Classifier(nn.Module):
    """
    GPT-2 + 分类头
    """
    def __init__(self, model_size="gpt2", num_classes=2, freeze_gpt=True):
        super().__init__()
        
        # 加载预训练 GPT-2
        self.gpt2 = GPT2Model.from_pretrained(model_size)
        self.config = self.gpt2.config
        
        # 分类头
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # 是否冻结 GPT-2 参数
        if freeze_gpt:
            for param in self.gpt2.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        output: (batch, num_classes)
        """
        # GPT-2 输出
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取最后一个 token 的隐藏状态
        last_hidden = outputs.last_hidden_state[:, -1, :]  # (batch, hidden_size)
        
        # 分类
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        
        return logits


# ===========================================
# 3. 演示
# ===========================================
def demo_load_gpt2():
    print("\n" + "="*60)
    print("演示1: 加载预训练 GPT-2")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 加载模型
    model, tokenizer = load_gpt2_from_hf("gpt2")
    model = model.to(device)
    
    # 测试编码
    text = "Hello, I am a language model"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print(f"\n输入文本: {text}")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"输出形状: {outputs.last_hidden_state.shape}")
    print(f"  (batch=1, seq_len={outputs.last_hidden_state.shape[1]}, hidden_size={outputs.last_hidden_state.shape[2]})")


def demo_classifier():
    print("\n" + "="*60)
    print("演示2: GPT-2 分类器")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 创建分类器（冻结 GPT-2）
    print("创建分类器（冻结 GPT-2 参数）...")
    classifier = GPT2Classifier(model_size="gpt2", num_classes=2, freeze_gpt=True).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 统计参数
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    
    # 测试数据
    texts = [
        "Hey, are you coming tonight?",
        "WINNER!! You won $1000! Click here!",
        "Can you pick up some milk?",
        "FREE entry to WIN $5000 cash prize!",
    ]
    labels = torch.tensor([0, 1, 0, 1]).to(device)  # 0=正常, 1=垃圾
    
    # 编码
    print("\n编码文本...")
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    print(f"输入形状: {inputs['input_ids'].shape}")
    
    # 训练前
    print("\n" + "-"*60)
    print("训练前:")
    classifier.eval()
    with torch.no_grad():
        logits = classifier(inputs['input_ids'], inputs['attention_mask'])
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
    
    for text, pred, label in zip(texts, predictions.tolist(), labels.tolist()):
        pred_str = "垃圾" if pred == 1 else "正常"
        label_str = "垃圾" if label == 1 else "正常"
        print(f"  '{text[:30]}...' → 预测: {pred_str}, 真实: {label_str}")
    
    print(f"\n准确率: {accuracy.item():.2%}")
    
    # 训练
    print("\n" + "-"*60)
    print("开始微调...")
    
    classifier.train()
    optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=5e-4)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        logits = classifier(inputs['input_ids'], inputs['attention_mask'])
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
            print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.2%}")
    
    # 训练后
    print("\n" + "-"*60)
    print("训练后:")
    classifier.eval()
    with torch.no_grad():
        logits = classifier(inputs['input_ids'], inputs['attention_mask'])
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
    
    for text, pred, label in zip(texts, predictions.tolist(), labels.tolist()):
        pred_str = "垃圾" if pred == 1 else "正常"
        label_str = "垃圾" if label == 1 else "正常"
        match = "✓" if pred == label else "✗"
        print(f"  '{text[:30]}...' → 预测: {pred_str}, 真实: {label_str} {match}")
    
    print(f"\n准确率: {accuracy.item():.2%}")
    
    # 测试新数据
    print("\n" + "-"*60)
    print("测试新数据:")
    
    test_texts = [
        "Call me back when you can",
        "URGENT: Your account has been compromised!",
    ]
    
    test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = classifier(test_inputs['input_ids'], test_inputs['attention_mask'])
        predictions = torch.argmax(logits, dim=-1)
    
    for text, pred in zip(test_texts, predictions.tolist()):
        pred_str = "垃圾短信" if pred == 1 else "正常短信"
        print(f"  '{text}' → {pred_str}")


def demo_model_sizes():
    print("\n" + "="*60)
    print("演示3: 不同模型大小对比")
    print("="*60)
    
    print("\nGPT-2 模型规格:")
    print("-"*60)
    print(f"{'模型':<15} {'参数量':>12} {'显存(推理)':>12}")
    print("-"*60)
    print(f"{'gpt2':<15} {'124M':>12} {'~500MB':>12}")
    print(f"{'gpt2-medium':<15} {'355M':>12} {'~1.4GB':>12}")
    print(f"{'gpt2-large':<15} {'774M':>12} {'~3GB':>12}")
    print(f"{'gpt2-xl':<15} {'1.5B':>12} {'~6GB':>12}")
    print("-"*60)
    
    print(f"\n你的显存: 16GB")
    print("✓ 可以运行所有 GPT-2 模型！")
    print("\n推荐:")
    print("  - 快速实验: gpt2 (124M)")
    print("  - 更好效果: gpt2-medium (355M)")


# ===========================================
# 主程序
# ===========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("第6章：使用 Hugging Face 加载 GPT-2")
    print("="*60)
    
    # 检查 transformers 是否安装
    try:
        import transformers
        print(f"transformers 版本: {transformers.__version__}")
    except ImportError:
        print("\n错误: 未安装 transformers 库")
        print("请运行: pip install transformers")
        exit(1)
    
    demo_load_gpt2()
    demo_classifier()
    demo_model_sizes()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    
    print("\n关键要点:")
    print("1. Hugging Face 让加载预训练模型变得简单")
    print("2. 不需要 TensorFlow，只需要 transformers 库")
    print("3. 可以选择不同大小的模型")
    print("4. 冻结预训练参数 + 训练分类头 = 快速微调")
