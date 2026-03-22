"""
指令微调测试脚本
测试训练好的模型能否执行指令
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class InstructionPredictor:
    def __init__(self, model_path="best_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("加载模型...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        
        # 加载训练好的权重
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ 模型加载成功\n")
    
    def predict(self, instruction, input_text=""):
        # 格式化 prompt
        prompt = f"### 指令:\n{instruction}\n\n"
        
        if input_text:
            prompt += f"### 输入:\n{input_text}\n\n"
        
        prompt += "### 回答:\n"
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 提取 Response 部分
        if "### 回答:" in full_output:
            response = full_output.split("### 回答:")[-1].strip()
        else:
            response = full_output
        
        return response


def demo():
    print("\n" + "="*60)
    print("指令微调模型测试")
    print("="*60)
    
    predictor = InstructionPredictor()
    
    # 测试样本
    test_cases = [
        {
            "instruction": "请把下面的句子翻译成中文",
            "input": "Good luck",
            "expected": "祝你好运"
        },
        {
            "instruction": "请把下面的句子翻译成英文",
            "input": "早上好",
            "expected": "Good morning"
        },
        {
            "instruction": "请回答问题",
            "input": "水的化学式是什么？",
            "expected": "H2O"
        },
        {
            "instruction": "请判断下面的情感倾向",
            "input": "这个产品非常好用！",
            "expected": "正面"
        },
        {
            "instruction": "请改写成更礼貌的表达",
            "input": "等一下",
            "expected": "请稍等"
        }
    ]
    
    print("\n测试结果:")
    print("-"*60)
    
    for i, test in enumerate(test_cases, 1):
        response = predictor.predict(test['instruction'], test['input'])
        
        print(f"\n测试 {i}:")
        print(f"  指令: {test['instruction']}")
        print(f"  输入: {test['input']}")
        print(f"  期望: {test['expected']}")
        print(f"  实际: {response}")
    
    print("\n" + "="*60)
    print("注意: 模型只训练了 3 个 epoch，效果可能不完美")
    print("      更多数据和更长训练可以提升效果")
    print("="*60)


def interactive():
    print("\n" + "="*60)
    print("交互模式 - 输入 'quit' 退出")
    print("="*60 + "\n")
    
    predictor = InstructionPredictor()
    
    while True:
        print("\n请输入指令和输入（用 | 分隔）：")
        print("示例: 请翻译成中文 | Hello")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("再见!")
            break
        
        if '|' in user_input:
            parts = user_input.split('|')
            instruction = parts[0].strip()
            input_text = parts[1].strip() if len(parts) > 1 else ""
        else:
            instruction = user_input
            input_text = ""
        
        if not instruction:
            continue
        
        response = predictor.predict(instruction, input_text)
        print(f"\n回答: {response}\n")
        print("-"*60)


def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '-i':
            interactive()
        else:
            # 单次预测
            predictor = InstructionPredictor()
            instruction = sys.argv[1]
            input_text = sys.argv[2] if len(sys.argv) > 2 else ""
            response = predictor.predict(instruction, input_text)
            print(f"\n回答: {response}")
    else:
        demo()


if __name__ == "__main__":
    main()
