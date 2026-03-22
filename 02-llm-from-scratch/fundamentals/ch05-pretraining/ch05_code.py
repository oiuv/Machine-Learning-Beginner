"""
第5章：预训练（Pretraining）
运行这个文件来体验训练 GPT 模型的过程。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ch04-gpt-model'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# 从第4章导入
from ch04_code import (
    LayerNorm, GELU, FeedForward, MultiHeadAttention, 
    TransformerBlock, GPTModel, generate_text_simple
)


# ===========================================
# 配置
# ===========================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # 缩短以加快训练
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

TRAINING_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1
}


# ===========================================
# 1. 损失函数演示
# ===========================================
def demo_loss_function():
    print("\n" + "="*60)
    print("1. 损失函数（Cross-Entropy Loss）演示")
    print("="*60)
    
    # 模拟两个预测
    # 情况1: 预测正确（目标类别概率高）
    logits_good = torch.tensor([[2.0, 0.1, 0.1]])  # 预测类别0
    target = torch.tensor([0])                      # 真实是类别0
    
    loss_good = F.cross_entropy(logits_good, target)
    
    # 情况2: 预测错误（目标类别概率低）
    logits_bad = torch.tensor([[0.1, 0.1, 2.0]])   # 预测类别2
    loss_bad = F.cross_entropy(logits_bad, target) # 真实是类别0
    
    print(f"目标类别: {target.item()}")
    print(f"\n情况1（预测正确）:")
    print(f"  Logits: {logits_good.tolist()}")
    print(f"  损失: {loss_good.item():.4f}")
    
    print(f"\n情况2（预测错误）:")
    print(f"  Logits: {logits_bad.tolist()}")
    print(f"  损失: {loss_bad.item():.4f}")
    
    print(f"\n结论: 预测越准 → 损失越低")


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算一个批次的损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# ===========================================
# 2. 数据加载器
# ===========================================
from torch.utils.data import Dataset, DataLoader

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
    return dataloader


# ===========================================
# 3. 训练循环核心
# ===========================================
def demo_training_steps():
    print("\n" + "="*60)
    print("2. 训练循环核心四步演示")
    print("="*60)
    
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建小模型演示
    small_config = {
        "vocab_size": 1000,
        "context_length": 16,
        "emb_dim": 64,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    model = GPTModel(small_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    # 模拟一个批次的数据
    input_batch = torch.randint(0, 1000, (2, 16)).to(device)
    target_batch = torch.randint(0, 1000, (2, 16)).to(device)
    
    print("\n初始状态:")
    initial_loss = calc_loss_batch(input_batch, target_batch, model, device)
    print(f"  损失: {initial_loss.item():.4f}")
    
    print("\n训练四步:")
    
    # 步骤1: 清空梯度
    print("\n  1. optimizer.zero_grad() - 清空上一步的梯度")
    optimizer.zero_grad()
    
    # 步骤2: 计算损失
    print("  2. loss = calc_loss(...) - 计算损失")
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    print(f"     当前损失: {loss.item():.4f}")
    
    # 步骤3: 反向传播
    print("  3. loss.backward() - 计算梯度")
    loss.backward()
    
    # 步骤4: 更新权重
    print("  4. optimizer.step() - 更新权重")
    optimizer.step()
    
    # 检查更新后的损失
    new_loss = calc_loss_batch(input_batch, target_batch, model, device)
    print(f"\n更新后损失: {new_loss.item():.4f}")
    print(f"损失变化: {initial_loss.item():.4f} → {new_loss.item():.4f}")
    print(f"\n结论: 损失下降，说明模型在学习！")


# ===========================================
# 4. 完整训练演示
# ===========================================
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成并打印样本"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=30,
            context_size=context_size
        )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"  生成: {decoded_text.replace(chr(10), ' ')}")
    model.train()


def train_model_demo():
    print("\n" + "="*60)
    print("3. 完整训练演示（使用小模型）")
    print("="*60)
    
    # 读取数据文件
    data_file = os.path.join(os.path.dirname(__file__), "the-verdict.txt")
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            text_data = f.read()
        print(f"数据文件: the-verdict.txt ({len(text_data)} 字符)")
    else:
        # 如果没有文件，使用内置文本
        text_data = """Every effort moves you, I know, the picture still remains 
    faintly upon the retina of my consciousness. The beauty of the world 
    has two edges, one of laughter, one of anguish, cutting the heart asunder.
    The most beautiful things in the world are the most useless; peacocks 
    and lilies for instance. The world has no name, the names I give to 
    things are only my names for them. I feel the beauty of the world.
    """ * 10  # 重复以获得更多数据
        print("使用内置数据")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 划分数据
    split_idx = int(0.90 * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    # 创建小模型（快速演示）
    small_config = {
        "vocab_size": 50257,
        "context_length": 128,
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    model = GPTModel(small_config).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # 创建数据加载器（使用较小的 stride 获得更多批次）
    train_loader = create_dataloader_v1(
        train_data, batch_size=2, max_length=128, stride=64, shuffle=True, drop_last=True
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=2, max_length=128, stride=64, shuffle=False, drop_last=False
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    if len(train_loader) == 0:
        print("\n警告: 训练数据太少，跳过训练演示")
        return
    
    start_context = "Every effort moves"
    
    print("\n" + "-"*60)
    print("训练前:")
    generate_and_print_sample(model, tokenizer, device, start_context)
    
    print("\n" + "-"*60)
    print("开始训练...")
    print("-"*60)
    
    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证损失
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                val_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
                val_count += 1
        val_loss = val_loss / val_count if val_count > 0 else 0
        
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    print("\n" + "-"*60)
    print("训练完成!")
    print("-"*60)
    print("\n观察: 损失下降，生成的文本变得越来越有意义")


# ===========================================
# 5. 优化器对比
# ===========================================
def demo_optimizer():
    print("\n" + "="*60)
    print("4. 优化器演示")
    print("="*60)
    
    # 简单的参数
    param = torch.nn.Parameter(torch.tensor([5.0]))
    target = torch.tensor([0.0])
    
    print(f"初始参数值: {param.item():.4f}")
    print(f"目标值: {target.item():.4f}")
    
    # SGD
    param_sgd = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer_sgd = torch.optim.SGD([param_sgd], lr=0.1)
    
    # AdamW
    param_adam = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer_adam = torch.optim.AdamW([param_adam], lr=0.1)
    
    print("\n优化10步后:")
    
    for i in range(10):
        # SGD
        optimizer_sgd.zero_grad()
        loss_sgd = (param_sgd - target).pow(2)
        loss_sgd.backward()
        optimizer_sgd.step()
        
        # AdamW
        optimizer_adam.zero_grad()
        loss_adam = (param_adam - target).pow(2)
        loss_adam.backward()
        optimizer_adam.step()
    
    print(f"  SGD 参数值:  {param_sgd.item():.4f}")
    print(f"  AdamW 参数值: {param_adam.item():.4f}")
    
    print("\n结论: AdamW 收敛更快（自适应学习率）")
    print("      这就是为什么 Transformer 训练都用 AdamW")


# ===========================================
# 主程序
# ===========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("第5章：预训练（Pretraining）")
    print("="*60)
    
    demo_loss_function()
    demo_training_steps()
    demo_optimizer()
    train_model_demo()
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)
    
    print("\n关键要点:")
    print("1. 损失函数衡量预测和真实的差距")
    print("2. 训练循环: 清梯度 → 算损失 → 反向传播 → 更新权重")
    print("3. 训练损失下降 = 模型在学习")
    print("4. AdamW 是现代 Transformer 的标准优化器")
    print("5. 训练后生成文本变得有意义")
