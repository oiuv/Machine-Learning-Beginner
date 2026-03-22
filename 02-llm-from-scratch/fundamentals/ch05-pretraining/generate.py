"""
使用训练好的模型生成文本
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ch04-gpt-model'))

import torch
import tiktoken
from ch04_code import GPTModel, generate_text_simple


# ===========================================
# 配置（必须和训练时一致）
# ===========================================
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(model, tokenizer, device, prompt, max_new_tokens=100):
    """生成文本"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size
        )
    
    return token_ids_to_text(token_ids, tokenizer)


def main():
    print("="*60)
    print("使用训练好的模型生成文本")
    print("="*60)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
    
    if not os.path.exists(model_path):
        print(f"\n错误: 找不到模型文件 {model_path}")
        print("请先运行 train_long.py 训练模型")
        return
    
    # 初始化模型
    model = GPTModel(GPT_CONFIG).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n模型加载成功!")
    print(f"训练轮次: Epoch {checkpoint['epoch']+1}")
    print(f"验证损失: {checkpoint['val_loss']:.4f}")
    
    # 分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 交互式生成
    print("\n" + "="*60)
    print("输入提示词，模型会继续生成")
    print("输入 'quit' 退出")
    print("="*60)
    
    # 默认提示词
    default_prompts = [
        "Every effort moves",
        "I had seen",
        "The beauty of",
        "She looked at",
        "He said to me"
    ]
    
    print("\n默认提示词示例:")
    for i, prompt in enumerate(default_prompts):
        print(f"  {i+1}. {prompt}")
    
    while True:
        print("\n" + "-"*60)
        user_input = input("输入提示词 (或数字1-5选择默认): ").strip()
        
        if user_input.lower() == 'quit':
            print("再见!")
            break
        
        # 选择默认提示词
        if user_input in ['1', '2', '3', '4', '5']:
            prompt = default_prompts[int(user_input) - 1]
        else:
            prompt = user_input
        
        if not prompt:
            continue
        
        # 生成
        print(f"\n输入: {prompt}")
        print("-"*60)
        
        # 生成不同长度
        for length in [50, 100]:
            output = generate(model, tokenizer, device, prompt, max_new_tokens=length)
            print(f"\n生成 ({length} tokens):")
            print(output)
        
        print("-"*60)


if __name__ == "__main__":
    main()
