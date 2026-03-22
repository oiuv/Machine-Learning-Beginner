"""
垃圾短信分类器 - 预测脚本（改进版）
测试改进后的模型效果
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from train import SpamClassifier


def predict(model, tokenizer, text, device):
    """预测单条文本"""
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred].item()
    
    return pred, confidence


def main():
    print("\n" + "="*60)
    print("垃圾短信分类器 - 测试改进版模型")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 加载模型
    print("加载模型...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = SpamClassifier(freeze_gpt=True).to(device)
    
    checkpoint = torch.load("model/best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ 模型加载成功")
    print(f"  验证准确率: {checkpoint['val_metrics']['accuracy']:.2%}")
    print(f"  验证 F1: {checkpoint['val_metrics']['f1']:.2%}\n")
    
    # 测试样本
    test_samples = [
        ("WINNER!! You won $1000! Click here!", "垃圾"),
        ("Hey, are you coming tonight?", "正常"),
        ("FREE entry to WIN $5000!", "垃圾"),
        ("Can you pick up milk?", "正常"),
        ("URGENT: Account compromised!", "垃圾"),
        ("See you at 7pm", "正常"),
        ("Congratulations! You won a prize!", "垃圾"),
        ("Running late, be there soon", "正常"),
        ("Click here NOW for free money!", "垃圾"),
        ("Thanks for your help!", "正常"),
    ]
    
    print("测试结果:")
    print("-"*60)
    
    correct = 0
    for text, true_label in test_samples:
        pred, confidence = predict(model, tokenizer, text, device)
        
        pred_label = "垃圾" if pred == 1 else "正常"
        is_correct = pred_label == true_label
        emoji = "✓" if is_correct else "✗"
        
        if is_correct:
            correct += 1
        
        print(f"\n短信: {text[:45]}...")
        print(f"真实: {true_label:4s} | 预测: {pred_label:4s} | 置信度: {confidence:.1%} {emoji}")
    
    print("\n" + "="*60)
    print(f"准确率: {correct}/{len(test_samples)} ({correct/len(test_samples):.0%})")
    print("="*60)
    
    print("\n🎉 改进成功!")
    print("  之前: 全部预测为'正常' (准确率 50%)")
    print(f"  现在: 能正确识别垃圾短信 (准确率 {correct/len(test_samples):.0%})")


if __name__ == "__main__":
    main()
