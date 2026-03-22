"""
Tokenization 原理演示

这个文件帮助你理解 tokenization 是如何工作的
"""

import tiktoken


def demo_why_tokenization():
    """
    演示为什么需要 tokenization
    """
    print("=" * 60)
    print("为什么需要 Tokenization？")
    print("=" * 60)
    
    text = "Hello, world!"
    
    print(f"\n原始文本: {text}")
    print(f"类型: {type(text)}")
    print("\n问题：神经网络只能处理数字，不能直接处理字符串！")
    
    print("\n" + "-" * 60)
    print("解决方案：Tokenization")
    print("-" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text)
    
    print(f"\n文本: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"类型: {type(token_ids)}")
    print(f"\n✓ 现在是数字了！可以送入神经网络处理")
    
    # 解码回来
    decoded = tokenizer.decode(token_ids)
    print(f"\n解码回文本: {decoded}")


def demo_word_level_tokenization():
    """
    演示词级别 tokenization
    """
    print("\n" + "=" * 60)
    print("方法1: 词级别 Tokenization")
    print("=" * 60)
    
    text = "I love learning"
    
    # 简单的按空格分割
    tokens = text.split()
    
    print(f"\n文本: {text}")
    print(f"Tokens: {tokens}")
    print(f"词表大小: 需要为每个词分配一个ID")
    
    # 手动构建词表
    vocab = {word: idx for idx, word in enumerate(tokens)}
    print(f"词表: {vocab}")
    
    print("\n问题:")
    print("  1. 'learn', 'learning', 'learned' 是三个不同的 token")
    print("  2. 遇到新词 'unbelievable' 无法处理")
    print("  3. 词表会非常大（几十万）")


def demo_char_level_tokenization():
    """
    演示字符级别 tokenization
    """
    print("\n" + "=" * 60)
    print("方法2: 字符级别 Tokenization")
    print("=" * 60)
    
    text = "Hello"
    
    # 按字符分割
    tokens = list(text)
    
    print(f"\n文本: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token 数量: {len(tokens)}")
    
    print("\n优点:")
    print("  - 词表很小（几十个字符）")
    print("  - 永远不会遇到未知字符")
    
    print("\n缺点:")
    print("  - 序列太长（每个字符一个 token）")
    print("  - 丢失了词的语义信息")


def demo_subword_tokenization():
    """
    演示子词级别 tokenization (BPE)
    """
    print("\n" + "=" * 60)
    print("方法3: 子词级别 Tokenization (BPE)")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 常见词
    common_word = "learning"
    tokens_common = tokenizer.encode(common_word)
    
    print(f"\n常见词: '{common_word}'")
    print(f"Token IDs: {tokens_common}")
    print(f"解码每个 token:")
    for token_id in tokens_common:
        print(f"  {token_id} → '{tokenizer.decode([token_id])}'")
    
    print("\n" + "-" * 60)
    
    # 罕见词
    rare_word = "unbelievable"
    tokens_rare = tokenizer.encode(rare_word)
    
    print(f"\n罕见词: '{rare_word}'")
    print(f"Token IDs: {tokens_rare}")
    print(f"解码每个 token:")
    for token_id in tokens_rare:
        print(f"  {token_id} → '{tokenizer.decode([token_id])}'")
    
    print("\n观察:")
    print("  - 常见词 'learning' 可能是一个 token")
    print("  - 罕见词 'unbelievable' 被拆成多个子词")
    print("  - 这样既能处理新词，又能保留语义！")


def demo_bpe_algorithm():
    """
    演示 BPE 算法的工作原理
    """
    print("\n" + "=" * 60)
    print("BPE 算法演示")
    print("=" * 60)
    
    print("\n假设我们有语料: 'low lower lowest'")
    
    # 初始状态：按字符分割
    print("\n初始状态（字符级别）:")
    corpus = ["l", "o", "w", "l", "o", "w", "e", "r", "l", "o", "w", "e", "s", "t"]
    print(f"  {corpus}")
    
    # 统计相邻对
    print("\n统计相邻字符对的出现频率:")
    pairs = {}
    for i in range(len(corpus) - 1):
        pair = (corpus[i], corpus[i + 1])
        pairs[pair] = pairs.get(pair, 0) + 1
    
    for pair, count in sorted(pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair}: {count} 次")
    
    print("\n第1轮：合并最频繁的对 'l' + 'o' → 'lo'")
    print("  结果: ['lo', 'w', 'lo', 'w', 'er', 'lo', 'w', 'est']")
    
    print("\n第2轮：合并 'lo' + 'w' → 'low'")
    print("  结果: ['low', 'low', 'er', 'low', 'est']")
    
    print("\n第3轮：合并 'e' + 'r' → 'er'")
    print("  结果: ['low', 'low', 'er', 'low', 'est']")
    
    print("\n继续合并...直到达到目标词表大小")
    
    print("\n最终词表可能包含:")
    print("  ['low', 'er', 'est', 'lower', 'lowest', ...]")


def demo_gpt2_tokenizer_details():
    """
    深入理解 GPT-2 tokenizer
    """
    print("\n" + "=" * 60)
    print("GPT-2 Tokenizer 深入理解")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"\n词表大小: {tokenizer.n_vocab}")
    print(f"这意味着 GPT-2 可以识别 {tokenizer.n_vocab} 种不同的 token")
    
    # 不同类型文本的 tokenization
    examples = [
        "Hello",                      # 简单词
        "Hello, world!",              # 带标点
        "The quick brown fox",        # 多个词
        "machine learning",           # 两个常见词
        "GPT-4 is amazing",           # 带数字
        "https://example.com",        # URL
        "我会说中文",                   # 中文
        "🎉🎊🎁",                      # emoji
    ]
    
    print("\n不同文本的 tokenization:")
    print("-" * 60)
    
    for text in examples:
        tokens = tokenizer.encode(text)
        token_strs = [tokenizer.decode([t]) for t in tokens]
        
        print(f"\n文本: {text}")
        print(f"  Token 数量: {len(tokens)}")
        print(f"  Tokens: {token_strs}")
        print(f"  IDs: {tokens}")


def demo_tokenization_efficiency():
    """
    演示 tokenization 的效率
    """
    print("\n" + "=" * 60)
    print("Tokenization 效率对比")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text = "The quick brown fox jumps over the lazy dog"
    
    # 词级别
    word_tokens = text.split()
    word_count = len(word_tokens)
    
    # 字符级别
    char_tokens = list(text)
    char_count = len(char_tokens)
    
    # BPE
    bpe_tokens = tokenizer.encode(text)
    bpe_count = len(bpe_tokens)
    
    print(f"\n文本: {text}")
    print(f"\n词级别: {word_count} tokens")
    print(f"字符级别: {char_count} tokens")
    print(f"BPE: {bpe_count} tokens")
    
    print(f"\n效率对比:")
    print(f"  BPE 比词级别多: {bpe_count - word_count} tokens")
    print(f"  BPE 比字符级别少: {char_count - bpe_count} tokens")
    
    print(f"\n结论:")
    print(f"  - BPE 在效率和表达能力之间取得平衡")
    print(f"  - 比 词级别稍多，但能处理新词")
    print(f"  - 比 字符级别少很多，序列更短")


def main():
    print("\n" + "=" * 60)
    print("Tokenization 原理完整演示")
    print("=" * 60)
    
    demo_why_tokenization()
    demo_word_level_tokenization()
    demo_char_level_tokenization()
    demo_subword_tokenization()
    demo_bpe_algorithm()
    demo_gpt2_tokenizer_details()
    demo_tokenization_efficiency()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Tokenization 的作用:
  1. 把文本转换成数字（神经网络能处理）
  2. 平衡词表大小和序列长度
  3. 处理未知词（通过子词分解）

BPE 的优势:
  1. 词表大小适中（几万个）
  2. 能处理新词
  3. 保留部分语义信息
  4. 序列不会太长

GPT-2 使用的 BPE:
  - 词表大小: 50,257
  - 能处理英文、中文、emoji 等
  - 常见词可能是单个 token
  - 罕见词会被拆成多个子词
    """)


if __name__ == "__main__":
    main()
