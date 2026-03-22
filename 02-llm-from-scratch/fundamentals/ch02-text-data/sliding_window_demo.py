"""
Sliding Window 滑动窗口原理演示

详细解释如何用滑动窗口准备训练数据
"""

import torch


def demo_why_sliding_window():
    """
    演示为什么需要滑动窗口
    """
    print("=" * 60)
    print("为什么需要滑动窗口？")
    print("=" * 60)
    
    print("""
LLM 的训练目标：预测下一个 token

例子：
  文本: "The cat sat on the mat"
  
  我们希望模型学会：
    "The" → 预测 "cat"
    "The cat" → 预测 "sat"
    "The cat sat" → 预测 "on"
    ...
    
  这就是"下一个词预测"任务
    """)
    
    print("\n问题：如何从一段文本构造训练样本？")
    print("答案：滑动窗口！")


def demo_basic_sliding_window():
    """
    演示基本的滑动窗口
    """
    print("\n" + "=" * 60)
    print("基本滑动窗口演示")
    print("=" * 60)
    
    # 假设这是 tokenized 后的文本
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    
    print(f"\nToken 序列: {tokens}")
    print("位置:        [0,  1,  2,  3,  4,  5,  6,  7]")
    
    context_size = 4  # 窗口大小
    
    print(f"\n窗口大小 (context_size): {context_size}")
    print("\n滑动窗口过程:")
    print("-" * 60)
    
    for i in range(len(tokens) - context_size):
        input_window = tokens[i:i + context_size]
        target_window = tokens[i + 1:i + context_size + 1]
        
        print(f"\n位置 {i}:")
        print(f"  输入 (x):  {input_window}")
        print(f"  目标 (y):     {target_window}")
        print(f"  解释: 用 {input_window} 预测 {target_window}")
    
    print("\n" + "-" * 60)
    print("\n观察规律:")
    print("  目标 = 输入向右移动一位")
    print("  输入: [a, b, c, d]")
    print("  目标:    [b, c, d, e]")
    print("           ↑ 预测这个")


def demo_sliding_window_visual():
    """
    可视化滑动窗口
    """
    print("\n" + "=" * 60)
    print("滑动窗口可视化")
    print("=" * 60)
    
    tokens = ["The", "cat", "sat", "on", "the", "mat", ".", ""]
    
    print(f"\nToken 序列: {tokens}")
    
    context_size = 4
    
    print(f"\n窗口大小: {context_size}")
    print("\n滑动过程:")
    print("-" * 60)
    
    for i in range(len(tokens) - context_size):
        input_tokens = tokens[i:i + context_size]
        target_tokens = tokens[i + 1:i + context_size + 1]
        
        # 可视化
        print(f"\n步骤 {i + 1}:")
        
        # 显示位置
        positions = " ".join([f"{j:5d}" for j in range(i, i + context_size + 1)])
        print(f"  位置:      {positions}")
        
        # 显示 token
        all_tokens = " ".join([f"{t:5s}" for t in tokens[i:i + context_size + 1]])
        print(f"  Tokens:    {all_tokens}")
        
        # 显示窗口
        input_str = "[" + ", ".join([f"{t:3s}" for t in input_tokens]) + "]"
        target_str = "   [" + ", ".join([f"{t:3s}" for t in target_tokens]) + "]"
        
        print(f"  输入:      {input_str}")
        print(f"  目标:      {target_str}")


def demo_stride_parameter():
    """
    演示 stride 参数的作用
    """
    print("\n" + "=" * 60)
    print("Stride (步长) 参数")
    print("=" * 60)
    
    tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    context_size = 4
    
    print(f"\nToken 序列: {tokens}")
    print(f"窗口大小: {context_size}")
    
    # Stride = 1 (最大重叠)
    print("\n" + "-" * 60)
    print("Stride = 1 (最大重叠，最多训练样本):")
    print("-" * 60)
    
    stride = 1
    for i in range(0, len(tokens) - context_size, stride):
        input_window = tokens[i:i + context_size]
        target_window = tokens[i + 1:i + context_size + 1]
        print(f"  样本: x={input_window}, y={target_window}")
    
    samples_stride_1 = len(range(0, len(tokens) - context_size, stride))
    print(f"\n总样本数: {samples_stride_1}")
    
    # Stride = 4 (无重叠)
    print("\n" + "-" * 60)
    print("Stride = 4 (无重叠，较少训练样本):")
    print("-" * 60)
    
    stride = 4
    for i in range(0, len(tokens) - context_size, stride):
        input_window = tokens[i:i + context_size]
        target_window = tokens[i + 1:i + context_size + 1]
        print(f"  样本: x={input_window}, y={target_window}")
    
    samples_stride_4 = len(range(0, len(tokens) - context_size, stride))
    print(f"\n总样本数: {samples_stride_4}")
    
    print("\n" + "-" * 60)
    print("\nStride 的作用:")
    print(f"  Stride = 1: 样本多 ({samples_stride_1}), 重叠多, 可能过拟合")
    print(f"  Stride = 4: 样本少 ({samples_stride_4}), 无重叠, 更高效")
    print("\n  常用设置: stride = context_size (无重叠)")


def demo_batching():
    """
    演示批处理
    """
    print("\n" + "=" * 60)
    print("批处理 (Batching)")
    print("=" * 60)
    
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    context_size = 4
    batch_size = 2
    
    print(f"\nToken 序列: {tokens}")
    print(f"窗口大小: {context_size}")
    print(f"Batch size: {batch_size}")
    
    print("\n生成训练样本:")
    print("-" * 60)
    
    samples_x = []
    samples_y = []
    
    for i in range(0, len(tokens) - context_size, context_size):
        input_window = tokens[i:i + context_size]
        target_window = tokens[i + 1:i + context_size + 1]
        samples_x.append(input_window)
        samples_y.append(target_window)
        print(f"  样本 {len(samples_x)}: x={input_window}, y={target_window}")
    
    print(f"\n总样本数: {len(samples_x)}")
    
    # 转换为 tensor
    inputs = torch.tensor(samples_x)
    targets = torch.tensor(samples_y)
    
    print(f"\n转换为 Tensor:")
    print(f"  Inputs shape:  {inputs.shape} (样本数 × 窗口大小)")
    print(f"  Targets shape: {targets.shape}")
    
    print(f"\nInputs:\n{inputs}")
    print(f"\nTargets:\n{targets}")
    
    print("\n" + "-" * 60)
    print("\n这就是 DataLoader 输出的格式！")


def demo_complete_example():
    """
    完整示例：从文本到训练数据
    """
    print("\n" + "=" * 60)
    print("完整示例：从文本到训练数据")
    print("=" * 60)
    
    # 模拟 tokenization
    text = "The cat sat on the mat"
    tokens = text.split()
    
    print(f"\n原始文本: '{text}'")
    print(f"Tokens: {tokens}")
    
    # 模拟 token IDs
    vocab = {token: i for i, token in enumerate(set(tokens))}
    token_ids = [vocab[t] for t in tokens]
    
    print(f"\nVocabulary: {vocab}")
    print(f"Token IDs: {token_ids}")
    
    # 滑动窗口
    context_size = 4
    stride = 2
    
    print(f"\n窗口大小: {context_size}")
    print(f"步长: {stride}")
    
    print("\n生成训练样本:")
    print("-" * 60)
    
    for i in range(0, len(token_ids) - context_size, stride):
        input_ids = token_ids[i:i + context_size]
        target_ids = token_ids[i + 1:i + context_size + 1]
        
        input_tokens = tokens[i:i + context_size]
        target_tokens = tokens[i + 1:i + context_size + 1]
        
        print(f"\n样本 {i // stride + 1}:")
        print(f"  输入 IDs: {input_ids}")
        print(f"  输入文本: {input_tokens}")
        print(f"  目标 IDs: {target_ids}")
        print(f"  目标文本: {target_tokens}")


def demo_why_target_shifted():
    """
    解释为什么目标要偏移一位
    """
    print("\n" + "=" * 60)
    print("为什么目标要偏移一位？")
    print("=" * 60)
    
    print("""
模型的任务是预测"下一个 token"。

考虑这个序列: [A, B, C, D, E]

训练时，我们希望模型学会：
  位置 0: 看到 A → 预测 B
  位置 1: 看到 B → 预测 C
  位置 2: 看到 C → 预测 D
  位置 3: 看到 D → 预测 E

所以：
  输入: [A, B, C, D]
  目标: [B, C, D, E]
        ↑  ↑  ↑  ↑
        预测每个位置的下一个 token

这就是为什么目标 = 输入向右移动一位！
    """)
    
    print("\n图解:")
    print("-" * 60)
    
    tokens = ["The", "cat", "sat", "on", "the"]
    
    print("\n输入序列:")
    print(f"  位置 0: '{tokens[0]}' → 希望预测 '{tokens[1]}'")
    print(f"  位置 1: '{tokens[1]}' → 希望预测 '{tokens[2]}'")
    print(f"  位置 2: '{tokens[2]}' → 希望预测 '{tokens[3]}'")
    print(f"  位置 3: '{tokens[3]}' → 希望预测 '{tokens[4]}'")
    
    print(f"\n所以:")
    print(f"  输入 x = {tokens[:4]}")
    print(f"  目标 y = {tokens[1:5]}")
    print(f"            ↑ 目标是输入的下一个 token")


def main():
    print("\n" + "=" * 60)
    print("滑动窗口 (Sliding Window) 原理演示")
    print("=" * 60)
    
    demo_why_sliding_window()
    demo_basic_sliding_window()
    demo_sliding_window_visual()
    demo_stride_parameter()
    demo_batching()
    demo_complete_example()
    demo_why_target_shifted()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
滑动窗口的核心概念:

1. 目标
   - 让模型学习预测下一个 token
   - 输入 [A, B, C, D] → 预测 [B, C, D, E]

2. 窗口大小 (context_size / max_length)
   - 每个训练样本包含多少个 token
   - GPT-2: 1024, GPT-3: 2048, GPT-4: 8192+

3. 步长 (stride)
   - 窗口每次移动多少个 token
   - stride=1: 最大重叠，最多样本
   - stride=context_size: 无重叠，更高效

4. 为什么目标偏移一位？
   - 目标 = 输入的下一个 token
   - 这样模型学习"给定前面所有 token，预测下一个"

5. 批处理 (batching)
   - 把多个样本打包成一个 batch
   - 输入形状: (batch_size, context_size)
   - 目标形状: (batch_size, context_size)
    """)


if __name__ == "__main__":
    main()
