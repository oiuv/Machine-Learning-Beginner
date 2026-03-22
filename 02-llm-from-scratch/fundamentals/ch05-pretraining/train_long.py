"""
第5章：训练 GPT 模型 - 更长时间的训练
看看效果能有多好
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ch04-gpt-model'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time

from torch.utils.data import Dataset, DataLoader
from ch04_code import GPTModel, generate_text_simple


# ===========================================
# 数据加载
# ===========================================
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=shuffle, drop_last=drop_last)
    return dataloader, tokenizer


# ===========================================
# 训练函数
# ===========================================
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size
        )
    
    text = token_ids_to_text(token_ids, tokenizer)
    model.train()
    return text


def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs, eval_freq, start_context, tokenizer, save_path="model_best.pth"):
    
    train_losses, val_losses = [], []
    tokens_seen = 0
    global_step = 0
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            # 定期评估
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_input, val_target in val_loader:
                        val_loss += calc_loss_batch(val_input, val_target, model, device).item()
                    val_loss /= len(val_loader)
                
                train_loss = epoch_loss / (global_step % len(train_loader) + 1)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch
                    }, save_path)
                
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:2d} | Step {global_step:4d} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"Best: {best_val_loss:.4f} | Time: {elapsed:.1f}s")
                
                model.train()
        
        # 每个 epoch 结束后生成样本
        sample = generate_sample(model, tokenizer, device, start_context, max_new_tokens=50)
        print(f"\n生成样本: {sample[:100]}...\n")
    
    return train_losses, val_losses, best_val_loss


# ===========================================
# 主程序
# ===========================================
def main():
    print("="*70)
    print("训练 GPT 模型 - 使用 the-verdict.txt")
    print("="*70)
    
    # 读取数据
    data_file = os.path.join(os.path.dirname(__file__), "the-verdict.txt")
    with open(data_file, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"数据: {len(text_data):,} 字符")
    
    # 配置
    GPT_CONFIG = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,      # 使用完整的 768
        "n_heads": 12,
        "n_layers": 12,      # 使用完整的 12 层
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    TRAINING_CONFIG = {
        "learning_rate": 5e-4,
        "num_epochs": 20,    # 训练 20 轮
        "batch_size": 4,
        "weight_decay": 0.1,
        "eval_freq": 10
    }
    
    # 划分数据
    split_idx = int(0.90 * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    # 创建数据加载器
    train_loader, tokenizer = create_dataloader_v1(
        train_data, 
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        shuffle=True,
        drop_last=True
    )
    
    val_loader, _ = create_dataloader_v1(
        val_data,
        batch_size=TRAINING_CONFIG["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        shuffle=False,
        drop_last=False
    )
    
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    
    # 初始化模型
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    model = GPTModel(GPT_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    # 训练前生成
    start_context = "Every effort moves"
    print("\n" + "-"*70)
    print("训练前:")
    sample = generate_sample(model, tokenizer, device, start_context, max_new_tokens=50)
    print(f"{sample}")
    print("-"*70)
    
    # 开始训练
    print("\n开始训练...\n")
    save_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
    
    train_losses, val_losses, best_loss = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=TRAINING_CONFIG["num_epochs"],
        eval_freq=TRAINING_CONFIG["eval_freq"],
        start_context=start_context,
        tokenizer=tokenizer,
        save_path=save_path
    )
    
    # 加载最佳模型并生成
    print("\n" + "="*70)
    print("训练完成！加载最佳模型...")
    print("="*70)
    
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳验证损失: {checkpoint['val_loss']:.4f} (Epoch {checkpoint['epoch']+1})")
    
    print("\n生成更多样本:")
    print("-"*70)
    
    prompts = [
        "Every effort moves",
        "I had seen",
        "The beauty of",
        "She looked at"
    ]
    
    model.eval()
    for prompt in prompts:
        sample = generate_sample(model, tokenizer, device, prompt, max_new_tokens=80)
        print(f"\n输入: {prompt}")
        print(f"输出: {sample}")
        print("-"*70)


if __name__ == "__main__":
    main()
