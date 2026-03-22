"""
Embedding Vector 初始化和学习演示

解释嵌入向量是如何从随机初始化到学习有意义的表示
"""

import torch
import torch.nn as nn
import numpy as np


def demo_embedding_initialization():
    """
    演示嵌入向量的初始化
    """
    print("=" * 60)
    print("嵌入向量的初始化")
    print("=" * 60)
    
    vocab_size = 10
    embed_dim = 4
    
    print(f"\n创建嵌入层:")
    print(f"  词表大小: {vocab_size}")
    print(f"  嵌入维度: {embed_dim}")
    
    # 创建嵌入层（随机初始化）
    torch.manual_seed(42)
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    print(f"\n初始权重（随机值）:")
    print(embedding.weight)
    
    print(f"\n关键点:")
    print(f"  1. 初始时，权重是随机值（无意义的）")
    print(f"  2. 训练过程中，这些权重会被更新")
    print(f"  3. 最终学到有意义的语义表示")
    
    # 不同的初始化方法
    print(f"\n常见初始化方法:")
    print(f"  - Xavier初始化: 适合tanh激活")
    print(f"  - He初始化: 适合ReLU激活")
    print(f"  - 随机正态分布: 最常用")
    print(f"  - 预训练向量: 迁移学习")


def demo_embedding_learning():
    """
    演示嵌入向量的学习过程
    """
    print("\n" + "=" * 60)
    print("嵌入向量的学习过程")
    print("=" * 60)
    
    # 模拟一个简单的训练场景
    vocab_size = 5
    embed_dim = 3
    
    torch.manual_seed(123)
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    # 初始权重
    initial_weight = embedding.weight[0].clone()
    
    print(f"\n初始状态:")
    print(f"  Token 0 的向量: {initial_weight.tolist()}")
    print(f"  (随机初始化，无意义)")
    
    # 模拟一次训练步骤
    print(f"\n模拟训练步骤:")
    
    # 输入
    input_ids = torch.tensor([0, 1, 2])
    
    # 前向传播
    embeddings = embedding(input_ids)
    
    print(f"  输入 IDs: {input_ids.tolist()}")
    print(f"  嵌入向量:")
    for i, emb in enumerate(embeddings):
        print(f"    ID {input_ids[i]}: {emb.tolist()}")
    
    # 模拟损失和反向传播
    print(f"\n反向传播过程:")
    
    # 创建一个简单的损失
    target = torch.randn_like(embeddings)
    loss = ((embeddings - target) ** 2).mean()
    
    print(f"  计算损失: {loss.item():.4f}")
    print(f"  反向传播...")
    
    # 反向传播
    loss.backward()
    
    # 查看梯度
    print(f"\n嵌入权重梯度:")
    print(f"  Token 0 的梯度: {embedding.weight.grad[0].tolist()}")
    
    # 更新权重
    print(f"\n权重更新:")
    print(f"  更新前: {initial_weight.tolist()}")
    
    # 手动更新一次（学习率=0.1）
    with torch.no_grad():
        embedding.weight -= 0.1 * embedding.weight.grad
    
    print(f"  更新后: {embedding.weight[0].tolist()}")
    
    print(f"\n观察:")
    print(f"  ✓ 梯度告诉我们如何调整向量")
    print(f"  ✓ 每次训练都会微调这些向量")
    print(f"  ✓ 经过大量训练，向量会学到语义")


def demo_semantic_learning():
    """
    演示语义学习的过程
    """
    print("\n" + "=" * 60)
    print("语义学习示例")
    print("=" * 60)
    
    print("""
假设我们有以下训练数据:
  - "cat sits on mat"
  - "dog sits on mat"
  - "cat runs in garden"
  - "dog runs in garden"

训练过程中，模型会学到:
  1. "cat" 和 "dog" 经常出现在相似上下文
  2. 它们的嵌入向量会逐渐接近
  3. 最终学到: "cat" ≈ "dog" (都是动物)

相似地:
  1. "sits" 和 "runs" 也出现在相似上下文
  2. 它们的向量也会接近
  3. 学到: "sits" ≈ "runs" (都是动作)

这就是为什么嵌入向量能捕捉语义！
    """)
    
    # 模拟训练前后的向量变化
    print("模拟训练前后的向量变化:")
    
    vocab_size = 5
    embed_dim = 3
    
    # 训练前的向量（随机）
    torch.manual_seed(42)
    before_cat = torch.randn(embed_dim)
    before_dog = torch.randn(embed_dim)
    
    print(f"\n训练前:")
    print(f"  'cat' 向量: {before_cat.tolist()}")
    print(f"  'dog' 向量: {before_dog.tolist()}")
    
    # 计算相似度（余弦相似度）
    cos_sim_before = torch.cosine_similarity(before_cat.unsqueeze(0), 
                                               before_dog.unsqueeze(0))
    print(f"  相似度: {cos_sim_before.item():.4f} (低)")
    
    # 训练后的向量（相似）
    after_cat = torch.tensor([0.5, 0.3, 0.8])
    after_dog = torch.tensor([0.6, 0.4, 0.7])
    
    print(f"\n训练后:")
    print(f"  'cat' 向量: {after_cat.tolist()}")
    print(f"  'dog' 向量: {after_dog.tolist()}")
    
    cos_sim_after = torch.cosine_similarity(after_cat.unsqueeze(0), 
                                              after_dog.unsqueeze(0))
    print(f"  相似度: {cos_sim_after.item():.4f} (高)")
    
    print(f"\n结论:")
    print(f"  ✓ 训练使相似词的向量接近")
    print(f"  ✓ 这是通过大量训练数据学到的")
    print(f"  ✓ 不需要人工标注语义关系")


def demo_different_embedding_models():
    """
    演示不同的嵌入模型
    """
    print("\n" + "=" * 60)
    print("不同的嵌入模型对比")
    print("=" * 60)
    
    models = {
        "Word2Vec (2013)": {
            "方法": "Skip-gram / CBOW",
            "训练数据": "Google News (100B tokens)",
            "特点": "预测上下文词",
            "维度": "300",
            "训练方式": "无监督",
        },
        "GloVe (2014)": {
            "方法": "矩阵分解",
            "训练数据": "Wikipedia + Gigaword",
            "特点": "全局词共现统计",
            "维度": "300",
            "训练方式": "无监督",
        },
        "BERT (2018)": {
            "方法": "Transformer",
            "训练数据": "Wikipedia + BookCorpus",
            "特点": "上下文相关",
            "维度": "768",
            "训练方式": "自监督（MLM + NSP）",
        },
        "GPT-2 (2019)": {
            "方法": "Transformer Decoder",
            "训练数据": "WebText (40GB)",
            "特点": "生成式预训练",
            "维度": "768-1600",
            "训练方式": "自监督（语言模型）",
        },
    }
    
    print("\n不同模型的特点:")
    print("-" * 60)
    
    for model_name, info in models.items():
        print(f"\n{model_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "-" * 60)
    print("\n关键区别:")
    print("""
1. 静态嵌入 (Word2Vec, GloVe)
   ✓ 每个词只有一个固定向量
   ✓ 无法处理多义词（如"bank"：银行 vs 河岸）
   
2. 上下文嵌入 (BERT, GPT)
   ✓ 同一个词在不同上下文中有不同向量
   ✓ 可以处理多义词
   ✓ 例子:
     "I went to the bank" → bank的向量偏向"银行"
     "I sat by the bank" → bank的向量偏向"河岸"
    """)


def demo_practical_example():
    """
    实际例子：嵌入向量的训练
    """
    print("\n" + "=" * 60)
    print("实际例子：语言模型训练")
    print("=" * 60)
    
    print("""
训练一个简单的语言模型:

输入: "The cat sat on"
目标: 预测 "mat"

训练步骤:
1. Tokenization: ["The", "cat", "sat", "on"] → [1, 2, 3, 4]
2. 查找嵌入: [向量1, 向量2, 向量3, 向量4]
3. 通过模型: 处理这些向量
4. 输出概率: P("mat"|上下文)
5. 计算损失: 如果预测错误
6. 反向传播: 更新嵌入向量
7. 重复数百万次

最终效果:
- "cat" 的向量学会与 "mat", "sit" 等词关联
- "dog" 的向量也会类似（都是动物）
- "run" 的向量与 "walk" 相似（都是动作）

这就是嵌入向量如何从随机变成有意义的！
    """)


def main():
    print("\n" + "=" * 60)
    print("嵌入向量初始化和学习完整演示")
    print("=" * 60 + "\n")
    
    demo_embedding_initialization()
    demo_embedding_learning()
    demo_semantic_learning()
    demo_different_embedding_models()
    demo_practical_example()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
嵌入向量是如何创建的？

1. 初始化
   - 随机初始化（通常是正态分布）
   - 或者使用预训练向量

2. 学习过程
   - 通过大量训练数据
   - 反向传播更新权重
   - 逐渐学到语义关系

3. 为什么有效
   - 相似上下文的词向量会接近
   - 模型自动发现语义关系
   - 不需要人工标注

4. 不同模型的区别
   - Word2Vec/GloVe: 静态嵌入
   - BERT/GPT: 上下文嵌入（更先进）

5. 关键洞察
   - Token ID是固定的标识符
   - 嵌入向量是可学习的表示
   - 训练过程就是学习这些向量的过程
    """)


if __name__ == "__main__":
    main()
