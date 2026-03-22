"""
垃圾短信分类器 - 改进版训练脚本
解决数据不平衡问题
"""

import os
import requests
import zipfile
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np


# ===========================================
# 1. 数据集类
# ===========================================
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ===========================================
# 2. 分类模型
# ===========================================
class SpamClassifier(nn.Module):
    def __init__(self, model_name="gpt2", num_classes=2, freeze_gpt=True):
        super().__init__()
        
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.config = self.gpt2.config
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        if freeze_gpt:
            for param in self.gpt2.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits


# ===========================================
# 3. 辅助函数
# ===========================================
def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "data.zip"
    data_path = "SMSSpamCollection"
    
    if os.path.exists(data_path):
        print("✓ 数据已存在")
        return data_path
    
    print("下载数据...")
    response = requests.get(url, timeout=60)
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    print("解压...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(zip_path)
    print("✓ 下载完成")
    return data_path


def load_data(data_path):
    print("加载数据...")
    df = pd.read_csv(data_path, sep='\t', names=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"总数据: {len(df)}")
    print(f"  正常短信: {(df['label']==0).sum()}")
    print(f"  垃圾短信: {(df['label']==1).sum()}")
    
    return df['text'].values, df['label'].values


def train_epoch(model, dataloader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # 使用类别权重
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }


# ===========================================
# 4. 主程序
# ===========================================
def main():
    print("\n" + "="*60)
    print("垃圾短信分类器 - 改进版训练（解决数据不平衡）")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 1. 下载数据
    data_path = download_data()
    texts, labels = load_data(data_path)
    
    # 计算类别权重
    print("\n计算类别权重（解决数据不平衡）...")
    class_counts = np.bincount(labels)
    total = len(labels)
    
    # 权重 = 总样本数 / (类别数 * 该类别样本数)
    class_weights = torch.tensor(
        [total / (2 * count) for count in class_counts],
        dtype=torch.float32
    ).to(device)
    
    print(f"  正常短信权重: {class_weights[0]:.4f}")
    print(f"  垃圾短信权重: {class_weights[1]:.4f}")
    print(f"  → 垃圾短信错误会被惩罚 {class_weights[1]/class_weights[0]:.1f}x 更重！")
    
    # 2. 划分数据
    print("\n划分数据...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"训练集: {len(train_texts)}")
    print(f"验证集: {len(val_texts)}")
    
    # 3. 创建数据集
    print("\n加载 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
    val_dataset = SpamDataset(val_texts, val_labels, tokenizer)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. 创建模型
    print("\n创建模型...")
    model = SpamClassifier(model_name="gpt2", freeze_gpt=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    
    # 5. 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    num_epochs = 5
    best_val_f1 = 0
    
    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("-"*60)
    
    os.makedirs("model", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 保存 F1 最好的模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'class_weights': class_weights.cpu(),
            }, "model/best_model.pt")
            print(f"  ✓ 保存最佳模型 (F1: {best_val_f1:.4f})")
    
    # 6. 完成
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳 F1 分数: {best_val_f1:.4f}")
    print(f"模型已保存到: model/best_model.pt")
    
    # 最终评估
    print("\n最终评估:")
    checkpoint = torch.load("model/best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = evaluate(model, val_loader, device)
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    
    print("\n下一步: 运行 python predict.py 进行预测")
    print("\n改进点:")
    print("  1. 使用类别权重平衡损失函数")
    print("  2. 增加 epoch 数（5轮）")
    print("  3. 使用 F1 分数选择最佳模型")


if __name__ == "__main__":
    main()
