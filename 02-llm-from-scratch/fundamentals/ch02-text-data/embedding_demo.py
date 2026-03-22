"""
Embedding 原理演示

解释 Token ID 如何转换为向量
"""

import torch
import torch.nn as nn


def demo_token_to_id():
    """
    演示 Token 到 ID 的映射
    """
    print("=" * 60)
    print("第1步: Token 到 ID 的映射（词表）")
    print("=" * 60)
    
    # 假设这是我们的词表
    vocab = {
        "<|endoftext|>": 0,
        "hello": 1,
        "world": 2,
        "the": 3,
        "is": 4,
        "a": 5,
        "test": 6,
        ",": 7,
        "!": 8,
    }
    
    print("\n词表（字典）:")
    for token, idx in vocab.items():
        print(f"  '{token}' → {idx}")
    
    print(f"\n词表大小: {len(vocab)}")
    
    # 使用词表
    text = "hello world"
    tokens = text.split()
    
    print(f"\n文本: '{text}'")
    print(f"Tokens: {tokens}")
    
    token_ids = [vocab[t] for t in tokens]
    print(f"Token IDs: {token_ids}")
    
    return vocab, token_ids


def demo_id_to_vector():
    """
    演示 ID 到向量的转换（Embedding）
    """
    print("\n" + "=" * 60)
    print("第2步: ID 到向量的转换（Embedding）")
    print("=" * 60)
    
    vocab_size = 10   # 词表大小
    embed_dim = 4     # 嵌入维度（实际中通常是 256, 512, 768 等）
    
    print(f"\n词表大小: {vocab_size}")
    print(f"嵌入维度: {embed_dim}")
    
    # 创建 Embedding 层
    torch.manual_seed(42)
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    print(f"\nEmbedding 层的本质是一个 {vocab_size} x {embed_dim} 的矩阵:")
    print(embedding.weight)
    
    print(f"\n理解这个矩阵:")
    print(f"  - 每一行对应一个 token")
    print(f"  - 第 0 行 → token ID 0 的向量")
    print(f"  - 第 1 行 → token ID 1 的向量")
    print(f"  - ...")
    
    # 查表操作
    token_ids = torch.tensor([1, 2])  # "hello", "world"
    
    print(f"\n输入 Token IDs: {token_ids.tolist()}")
    print(f"  ID 1 → 'hello'")
    print(f"  ID 2 → 'world'")
    
    vectors = embedding(token_ids)
    
    print(f"\n查表结果:")
    print(f"  形状: {vectors.shape} (2个token, 每个4维向量)")
    print(f"\n向量:")
    for i, vec in enumerate(vectors):
        print(f"  Token ID {token_ids[i]}: {vec.tolist()}")
    
    # 验证：embedding(1) 就是权重矩阵的第1行
    print(f"\n验证：Embedding 就是查表")
    print(f"  embedding(1) = 权重矩阵第1行:")
    print(f"  {embedding.weight[1].tolist()}")
    print(f"  {vectors[0].tolist()}")
    print(f"  相等? {torch.allclose(embedding.weight[1], vectors[0])}")


def demo_embedding_is_lookup():
    """
    深入理解 Embedding 就是查表
    """
    print("\n" + "=" * 60)
    print("深入理解: Embedding = 查表操作")
    print("=" * 60)
    
    vocab_size = 6
    embed_dim = 3
    
    torch.manual_seed(123)
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    print(f"\nEmbedding 权重矩阵 (6 x 3):")
    print(embedding.weight)
    
    # 手动查表
    print(f"\n手动查表 vs Embedding 层:")
    
    token_id = 2
    manual_lookup = embedding.weight[token_id]
    embedding_lookup = embedding(torch.tensor([token_id]))[0]
    
    print(f"\nToken ID {token_id}:")
    print(f"  手动查表 (权重矩阵第{token_id}行): {manual_lookup.tolist()}")
    print(f"  Embedding 层输出: {embedding_lookup.tolist()}")
    print(f"  相等? {torch.allclose(manual_lookup, embedding_lookup)}")
    
    print(f"\n结论:")
    print(f"  Embedding(token_id) = 权重矩阵[token_id]")
    print(f"  就是一个简单的查表操作！")


def demo_why_embedding():
    """
    为什么需要 Embedding？
    """
    print("\n" + "=" * 60)
    print("为什么需要 Embedding？")
    print("=" * 60)
    
    print("""
问题1: 为什么不直接用 Token ID？

  假设词表: {"hello": 1, "world": 2, "the": 3}
  
  ❌ 直接用 ID:
     "hello" → 1
     "world" → 2
     
     问题:
       - 1 和 2 有数学关系（2 > 1）
       - 但 "hello" 和 "world" 没有大小关系
       - Token ID 只是标识符，不应该有数学意义
  
  ✓ 用 Embedding:
     "hello" → [0.1, 0.3, -0.2, 0.5]
     "world" → [0.4, -0.1, 0.2, 0.3]
     
     优点:
       - 每个 token 是独立的向量
       - 向量之间可以计算相似度
       - 神经网络可以学习 token 之间的关系
    """)
    
    print("-" * 60)
    
    print("""
问题2: 为什么 Embedding 能学习语义？

  初始: 随机向量（没有意义）
  
  训练过程:
    "cat" 和 "dog" 经常出现在相似上下文
    → 它们的 Embedding 会逐渐接近
    → 最终学到语义关系
  
  示例（训练后）:
    "king"   - "man"   + "woman" ≈ "queen"
    [0.9, ...] - [0.5, ...] + [0.4, ...] ≈ [0.8, ...]
    
  这就是词向量的神奇之处！
    """)


def demo_one_hot_vs_embedding():
    """
    One-hot vs Embedding
    """
    print("\n" + "=" * 60)
    print("One-hot 编码 vs Embedding")
    print("=" * 60)
    
    vocab_size = 6
    
    print(f"词表大小: {vocab_size}")
    
    # One-hot 编码
    print(f"\nOne-hot 编码 (传统方法):")
    print(f"  Token ID 0 → [1, 0, 0, 0, 0, 0]")
    print(f"  Token ID 1 → [0, 1, 0, 0, 0, 0]")
    print(f"  Token ID 2 → [0, 0, 1, 0, 0, 0]")
    print(f"  ...")
    print(f"  问题:")
    print(f"    - 向量长度 = 词表大小（可能 50000+）")
    print(f"    - 大部分是 0，浪费空间")
    print(f"    - 所有词之间的距离相等（无法表示相似性）")
    
    # Embedding
    embed_dim = 3
    print(f"\nEmbedding (现代方法):")
    print(f"  Token ID 0 → [0.1, 0.2, 0.3]")
    print(f"  Token ID 1 → [0.4, 0.5, 0.6]")
    print(f"  Token ID 2 → [0.7, 0.8, 0.9]")
    print(f"  ...")
    print(f"  优点:")
    print(f"    - 向量长度固定（如 256, 512）")
    print(f"    - 稠密向量，计算高效")
    print(f"    - 可以学习词之间的相似性")
    
    print(f"\n数学上的等价性:")
    print(f"  Embedding(token_id) = OneHot(token_id) @ Embedding矩阵")
    print(f"  但 Embedding 层直接查表，更高效！")


def demo_complete_pipeline():
    """
    完整流程演示
    """
    print("\n" + "=" * 60)
    print("完整流程: 文本 → Token IDs → Embeddings")
    print("=" * 60)
    
    # 模拟词表
    vocab = {
        "hello": 0,
        "world": 1,
        "the": 2,
        "is": 3,
        "test": 4,
    }
    id_to_token = {v: k for k, v in vocab.items()}
    
    # 文本
    text = "hello world"
    
    print(f"\n原始文本: '{text}'")
    
    # Step 1: Tokenization
    tokens = text.split()
    print(f"\nStep 1 - Tokenization:")
    print(f"  文本 → Tokens: {tokens}")
    
    # Step 2: Token to ID
    token_ids = [vocab[t] for t in tokens]
    print(f"\nStep 2 - 查词表 (Token → ID):")
    for t, idx in zip(tokens, token_ids):
        print(f"  '{t}' → ID {idx}")
    print(f"  结果: {token_ids}")
    
    # Step 3: ID to Embedding
    vocab_size = len(vocab)
    embed_dim = 4
    
    torch.manual_seed(42)
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    token_ids_tensor = torch.tensor(token_ids)
    embeddings = embedding(token_ids_tensor)
    
    print(f"\nStep 3 - Embedding 查表 (ID → 向量):")
    print(f"  Embedding 矩阵形状: {embedding.weight.shape}")
    print(f"  输入 IDs: {token_ids}")
    print(f"  输出形状: {embeddings.shape}")
    print(f"\n  结果:")
    for i, (token, idx) in enumerate(zip(tokens, token_ids)):
        print(f"    '{token}' (ID {idx}) → {embeddings[i].tolist()}")
    
    print(f"\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"""
完整流程:
  文本: "hello world"
    ↓ Tokenization
  Tokens: ["hello", "world"]
    ↓ 查词表
  Token IDs: [0, 1]
    ↓ Embedding 查表
  Embeddings: [[向量0], [向量1]]
    形状: (2, 4)  # 2个token, 每个4维向量
    ↓
  送入神经网络...
    """)


def main():
    print("\n" + "=" * 60)
    print("Embedding 原理完整演示")
    print("=" * 60)
    
    demo_token_to_id()
    demo_id_to_vector()
    demo_embedding_is_lookup()
    demo_why_embedding()
    demo_one_hot_vs_embedding()
    demo_complete_pipeline()


if __name__ == "__main__":
    main()
