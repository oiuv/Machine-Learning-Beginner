"""
第7章：指令微调演示
展示指令微调的核心概念（无需实际训练）
"""


# ===========================================
# 1. 指令数据格式
# ===========================================
def demo_instruction_format():
    """演示指令数据格式"""
    print("\n" + "="*60)
    print("演示1: 指令数据格式")
    print("="*60)
    
    # 示例数据
    examples = [
        {
            "instruction": "请把下面的句子翻译成中文",
            "input": "Hello world",
            "output": "你好世界"
        },
        {
            "instruction": "请总结下面的文本",
            "input": "今天天气很好，阳光明媚，温度适宜，非常适合外出散步。",
            "output": "天气晴朗，适合外出。"
        },
        {
            "instruction": "请回答问题",
            "input": "中国的首都是哪里？",
            "output": "中国的首都是北京。"
        }
    ]
    
    print("\n原始数据:")
    print("-"*60)
    for i, ex in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print(f"  指令: {ex['instruction']}")
        print(f"  输入: {ex['input']}")
        print(f"  输出: {ex['output']}")
    
    print("\n\n格式化后的 Prompt:")
    print("-"*60)
    
    for i, ex in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        prompt = format_prompt(ex)
        print(prompt)


def format_prompt(entry):
    """格式化指令输入"""
    prompt = f"### Instruction:\n{entry['instruction']}\n\n"
    
    if entry.get('input'):
        prompt += f"### Input:\n{entry['input']}\n\n"
    
    prompt += "### Response:\n"
    
    return prompt


# ===========================================
# 2. 微调前后对比
# ===========================================
def demo_before_after():
    """演示微调前后对比"""
    print("\n" + "="*60)
    print("演示2: 微调前后对比")
    print("="*60)
    
    print("\n【微调前】预训练模型:")
    print("-"*60)
    print("用户: 请把'Hello world'翻译成中文")
    print("模型: 请把'Hello world'翻译成中文，请问您需要...")
    print("      ↑ 只是继续生成文本，不理解指令")
    
    print("\n【微调后】指令微调模型:")
    print("-"*60)
    print("用户: 请把'Hello world'翻译成中文")
    print("模型: 你好世界")
    print("      ✓ 理解了翻译指令，执行任务")


# ===========================================
# 3. 训练过程
# ===========================================
def demo_training_process():
    """演示训练过程"""
    print("\n" + "="*60)
    print("演示3: 训练过程")
    print("="*60)
    
    print("\n1. 准备数据")
    print("-"*60)
    print("   原始格式:")
    print('   {"instruction": "翻译", "input": "Hello", "output": "你好"}')
    print("\n   格式化后:")
    print('   ### Instruction:')
    print('   翻译')
    print('   ')
    print('   ### Input:')
    print('   Hello')
    print('   ')
    print('   ### Response:')
    print('   你好')
    
    print("\n2. 训练目标")
    print("-"*60)
    print("   输入: ### Instruction: 翻译 ### Input: Hello ### Response:")
    print("   目标: 你好")
    print("   ")
    print("   损失 = CrossEntropy(模型预测, 真实答案)")
    print("   ")
    print("   关键: 只计算 Response 部分的损失！")
    print("         Instruction 和 Input 部分不计算损失")
    
    print("\n3. 批量训练")
    print("-"*60)
    print("   样本1: [1, 2, 3, 4, 5]  (短)")
    print("   样本2: [6, 7, 8, 9, 10, 11, 12]  (长)")
    print("   ")
    print("   Padding 到相同长度:")
    print("   样本1: [1, 2, 3, 4, 5, 0, 0]")
    print("   样本2: [6, 7, 8, 9, 10, 11, 12]")
    print("   ")
    print("   Mask:")
    print("   样本1 labels: [1, 2, 3, 4, 5, -100, -100]")
    print("                ↑ 只计算前5个位置的损失")


# ===========================================
# 4. 多任务学习
# ===========================================
def demo_multi_task():
    """演示多任务学习"""
    print("\n" + "="*60)
    print("演示4: 多任务学习")
    print("="*60)
    
    tasks = [
        ("翻译", "Translate to Chinese: Hello", "你好"),
        ("摘要", "Summarize: 今天天气...", "天气晴朗"),
        ("问答", "Question: 首都在哪？", "北京"),
        ("改写", "Rewrite politely: 滚开", "请让一下"),
        ("分类", "Classify sentiment: 很棒！", "正面")
    ]
    
    print("\n同时学习多种任务:")
    print("-"*60)
    
    for task_name, example_input, example_output in tasks:
        print(f"\n任务: {task_name}")
        print(f"  输入: {example_input}")
        print(f"  输出: {example_output}")
    
    print("\n" + "-"*60)
    print("关键: 模型学会'理解指令'而非'记住任务'")
    print("      可以泛化到没见过的新指令！")


# ===========================================
# 5. 代码示例
# ===========================================
def demo_code_examples():
    """演示代码示例"""
    print("\n" + "="*60)
    print("演示5: 代码示例")
    print("="*60)
    
    print("\n1. 数据集类")
    print("-"*60)
    print("""
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # 格式化
        prompt = format_prompt(entry)
        full_text = prompt + entry['output']
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_text)
        
        return input_ids
""")
    
    print("\n2. 训练循环")
    print("-"*60)
    print("""
for batch in train_loader:
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    # 前向传播
    logits = model(input_ids)
    
    # 计算损失（只计算 Response 部分）
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100  # 忽略 padding
    )
    
    # 反向传播
    loss.backward()
    optimizer.step()
""")
    
    print("\n3. 推理")
    print("-"*60)
    print("""
# 格式化输入
prompt = format_prompt({
    'instruction': '翻译成中文',
    'input': 'Hello'
})

# 生成
input_ids = tokenizer.encode(prompt)
output = model.generate(input_ids)
response = tokenizer.decode(output)
""")


# ===========================================
# 6. 关键技术
# ===========================================
def demo_key_techniques():
    """演示关键技术"""
    print("\n" + "="*60)
    print("演示6: 关键技术")
    print("="*60)
    
    techniques = [
        ("Prompt Template", "统一格式，让模型识别指令"),
        ("Loss Masking", "只计算 Response 部分损失"),
        ("Padding & Truncation", "处理不同长度序列"),
        ("Multi-task Learning", "同时训练多种任务"),
        ("Few-shot Learning", "少量样本快速学习"),
    ]
    
    for name, desc in techniques:
        print(f"\n{name}:")
        print(f"  {desc}")
    
    print("\n" + "-"*60)
    print("提示工程技巧:")
    print("  1. 清晰的指令描述")
    print("  2. 明确的输入输出格式")
    print("  3. 适当的示例（Few-shot）")
    print("  4. 一致的模板风格")


# ===========================================
# 7. 主程序
# ===========================================
def main():
    print("\n" + "="*60)
    print("第7章：指令微调（Instruction Fine-tuning）")
    print("="*60)
    
    demo_instruction_format()
    demo_before_after()
    demo_training_process()
    demo_multi_task()
    demo_code_examples()
    demo_key_techniques()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    
    print("\n关键要点:")
    print("1. 指令微调让模型学会理解和执行指令")
    print("2. 使用统一的 Prompt Template 格式化输入")
    print("3. 只计算 Response 部分的损失")
    print("4. 多任务训练提升泛化能力")
    print("5. 可以泛化到没见过的新指令")
    
    print("\n学习进度:")
    print("  ✓ 第2章: 文本数据处理")
    print("  ✓ 第3章: 注意力机制")
    print("  ✓ 第4章: GPT 模型架构")
    print("  ✓ 第5章: 预训练")
    print("  ✓ 第6章: 分类微调")
    print("  ✓ 第7章: 指令微调 ← 你在这里")
    print("  ○ 第8章: RLHF（人类反馈强化学习）")


if __name__ == "__main__":
    main()
