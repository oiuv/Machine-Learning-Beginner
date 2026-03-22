"""
指令微调训练脚本
使用 GPT-2 微调中文指令任务
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


# ===========================================
# 1. 数据集类
# ===========================================
class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载数据: {len(self.data)} 条指令")
        
        # 预处理
        self.inputs = []
        self.labels = []
        
        for entry in self.data:
            # 格式化 prompt
            prompt = self.format_prompt(entry)
            full_text = prompt + entry['output']
            
            # Tokenize
            encoded = tokenizer.encode(
                full_text,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            # 创建 input 和 labels
            input_ids = encoded[:-1]
            label_ids = encoded[1:]
            
            # 找到 Response 开始位置
            prompt_encoded = tokenizer.encode(prompt, max_length=max_length, truncation=True)
            prompt_length = len(prompt_encoded)
            
            # Prompt 部分设为 -100（不计算损失）
            for i in range(min(prompt_length - 1, len(label_ids))):
                label_ids[i] = -100
            
            # Padding 部分也设为 -100
            for i in range(len(label_ids)):
                if input_ids[i] == tokenizer.pad_token_id:
                    label_ids[i] = -100
            
            self.inputs.append(input_ids)
            self.labels.append(label_ids)
    
    def format_prompt(self, entry):
        prompt = f"### 指令:\n{entry['instruction']}\n\n"
        
        if entry.get('input'):
            prompt += f"### 输入:\n{entry['input']}\n\n"
        
        prompt += "### 回答:\n"
        
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.inputs[idx]),
            'labels': torch.tensor(self.labels[idx])
        }


# ===========================================
# 2. 训练函数
# ===========================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="训练中"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


# ===========================================
# 3. 主程序
# ===========================================
def main():
    print("\n" + "="*60)
    print("指令微调训练 - 中文指令")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 1. 加载 tokenizer 和模型
    print("加载模型...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}\n")
    
    # 2. 创建数据集
    print("加载数据集...")
    dataset = InstructionDataset("data.json", tokenizer)
    
    # 划分训练/验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"训练集: {train_size} 条")
    print(f"验证集: {val_size} 条\n")
    
    # 3. 训练设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    # 4. 训练
    print(f"开始训练 ({num_epochs} epochs)...")
    print("-"*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
    
    # 5. 完成
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存: best_model.pt")
    print("\n下一步: 运行 python predict.py 测试效果")


if __name__ == "__main__":
    main()
