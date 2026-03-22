"""
Token ID 是如何确定的演示
"""

import tiktoken


def demo_token_id_is_fixed():
    """
    演示 Token ID 是固定的，不是随机的
    """
    print("=" * 60)
    print("Token ID 是固定的，不是随机的！")
    print("=" * 60)
    
    # GPT-2 的词表是固定的
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"\nGPT-2 词表大小: {tokenizer.n_vocab}")
    print("\n" + "-" * 60)
    
    # 多次编码同一个词，结果始终相同
    word = "hello"
    
    print(f"\n多次编码 '{word}'，看看 ID 是否相同:")
    for i in range(5):
        token_ids = tokenizer.encode(word)
        print(f"  第{i+1}次: {token_ids}")
    
    print(f"\n结论: '{word}' 的 Token ID 永远是 {tokenizer.encode(word)[0]}")
    
    # 一些常见 token 的固定 ID
    print("\n" + "-" * 60)
    print("\n一些常见 token 的固定 ID:")
    
    common_tokens = ["the", "a", "is", "hello", "world", ",", ".", "!", ""]
    
    for token in common_tokens:
        token_id = tokenizer.encode(token, allowed_special={"", "<|endoftext|>"})
        print(f"  '{token}' → ID {token_id}")


def demo_how_vocab_is_built():
    """
    演示词表是如何构建的
    """
    print("\n" + "=" * 60)
    print("词表是如何构建的？")
    print("=" * 60)
    
    print("""
词表构建过程（在训练 tokenizer 时完成）:

步骤1: 收集大量训练文本
    例如: Wikipedia, 书籍, 网页等 (几十 GB 的文本)

步骤2: 初始化字符集
    ['a', 'b', 'c', ..., 'A', 'B', ..., '0', '1', ..., ' ', ...]

步骤3: BPE 训练
    - 统计字符对频率
    - 合并最频繁的字符对
    - 重复直到达到目标词表大小 (50,257)

步骤4: 分配 ID
    - 按频率排序
    - 频率高的 token ID 小 (0, 1, 2, ...)
    - 频率低的 token ID 大

最终: 得到固定的词表文件
    - GPT-2 的词表: 50,257 个 token
    - 每个 token 对应唯一的 ID
    - 这个映射关系永久固定
    """)
    
    print("\n" + "-" * 60)
    print("\n为什么 ID 要固定？")
    print("-" * 60)
    
    print("""
1. 模型训练时使用这些 ID
   - 如果 ID 变了，模型就不知道 token 是什么了
   
2. 模型推理时使用相同的 ID
   - 必须和训练时一致，否则会出错
   
3. 权重对应关系
   - 模型权重是根据这些 ID 学习的
   - ID 15496 对应的权重，永远是 "hello" 的含义
   
类比：
    就像电话号码簿：
    - "张三" → 13812345678
    - 一旦分配，就固定了
    - 每次打 "张三" 的电话，都是这个号码
    """)


def demo_random_vs_fixed():
    """
    对比：随机 vs 固定
    """
    print("\n" + "=" * 60)
    print("对比：随机 ID vs 固定 ID")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n如果是随机的（错误理解）:")
    print("  第1次运行: 'hello' → ID 12345")
    print("  第2次运行: 'hello' → ID 67890  ❌ 每次都不同！")
    print("  → 模型会完全混乱")
    
    print("\n实际情况（正确的）:")
    print(f"  第1次运行: 'hello' → ID {tokenizer.encode('hello')[0]}")
    print(f"  第2次运行: 'hello' → ID {tokenizer.encode('hello')[0]}")
    print(f"  第3次运行: 'hello' → ID {tokenizer.encode('hello')[0]}")
    print("  ✓ 永远相同！")


def demo_different_tokenizers():
    """
    演示不同的 tokenizer 有不同的词表
    """
    print("\n" + "=" * 60)
    print("不同的模型使用不同的词表")
    print("=" * 60)
    
    tokenizers = {
        "GPT-2": tiktoken.get_encoding("gpt2"),
        "GPT-3/4 (cl100k_base)": tiktoken.get_encoding("cl100k_base"),
    }
    
    word = "hello"
    
    print(f"\n同一个词 '{word}' 在不同模型中的 Token ID:")
    print("-" * 60)
    
    for name, tokenizer in tokenizers.items():
        token_id = tokenizer.encode(word)
        vocab_size = tokenizer.n_vocab
        print(f"  {name}:")
        print(f"    Token ID: {token_id}")
        print(f"    词表大小: {vocab_size}")
    
    print("\n结论:")
    print("  - 不同模型有自己的词表")
    print("  - 同一个词在不同模型中可能有不同的 ID")
    print("  - 但在同一个模型内，ID 永远固定")


def demo_vocabulary_file():
    """
    演示词表文件
    """
    print("\n" + "=" * 60)
    print("词表文件长什么样？")
    print("=" * 60)
    
    print("""
GPT-2 的词表文件（简化版示例）:

{
    "!": 0,
    "\"": 1,
    "#": 2,
    "$": 3,
    ...
    "the": 1169,
    "hello": 15496,
    "world": 995,
    ...
    "Ġlearning": 40684,  # Ġ 表示空格
    ...
}

这个文件有 50,257 行，每行是一个 token 和它的 ID。

注意：
- Ġ 是 GPT-2 用来表示空格的特殊字符
- "Ġlearning" 表示 " learning"（前面有空格）
- "learning" 和 "Ġlearning" 是不同的 token

存储位置：
- GPT-2: vocab.bpe 和 encoder.json 文件
- tiktoken: 内置在库中，自动下载
    """)
    
    # 实际查看一些 token
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n" + "-" * 60)
    print("\n实际查看一些 token:")
    
    # 解码一些 ID
    for token_id in [0, 1, 100, 1000, 10000, 15496, 50256]:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"  ID {token_id:5d} → '{decoded}'")
        except:
            print(f"  ID {token_id:5d} → (无法解码)")


def main():
    print("\n" + "=" * 60)
    print("Token ID 是如何确定的？")
    print("=" * 60 + "\n")
    
    demo_token_id_is_fixed()
    demo_how_vocab_is_built()
    demo_random_vs_fixed()
    demo_different_tokenizers()
    demo_vocabulary_file()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
1. Token ID 是固定的，不是随机的
   - 在训练 tokenizer 时就确定了
   - 永远不会改变

2. 词表是如何构建的
   - 收集大量文本
   - 用 BPE 算法学习子词
   - 分配固定的 ID

3. 为什么必须固定
   - 模型权重是根据这些 ID 学习的
   - 如果 ID 变了，模型就废了

4. 不同模型有不同词表
   - GPT-2: 50,257 个 token
   - GPT-3/4: 100,000+ 个 token
   - 每个模型都有自己的词表文件
    """)


if __name__ == "__main__":
    main()
