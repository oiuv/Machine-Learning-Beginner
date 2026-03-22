"""
第2章：文本数据处理 - 完整代码

运行此代码学习文本数据处理的各个步骤
"""

import re
import os
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================
# 2.2 文本分词 (Tokenization)
# ============================================

def simple_tokenize(text):
    """
    简单分词器: 使用正则表达式分割文本
    """
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed


def demo_tokenization():
    """
    演示基本分词
    """
    print("=" * 60)
    print("2.2 文本分词演示")
    print("=" * 60)
    
    text = "Hello, world. Is this-- a test?"
    tokens = simple_tokenize(text)
    
    print(f"原文: {text}")
    print(f"Tokens: {tokens}")
    print()


# ============================================
# 2.3 Token到ID的转换
# ============================================

class SimpleTokenizerV1:
    """
    简单分词器V1: 基础版本
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = simple_tokenize(text)
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """
    简单分词器V2: 支持未知词和特殊token
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = simple_tokenize(text)
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def demo_tokenizer():
    """
    演示Tokenizer的使用
    """
    print("=" * 60)
    print("2.3 Token到ID转换演示")
    print("=" * 60)
    
    with open("learning/ch02-text-data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    preprocessed = simple_tokenize(raw_text)
    all_words = sorted(set(preprocessed))
    
    all_tokens = all_words + ["", "<|unk|>"]
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    
    print(f"词表大小: {len(vocab)}")
    
    tokenizer = SimpleTokenizerV2(vocab)
    
    text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    print(f"原文: {text}")
    print(f"编码: {ids[:20]}...")
    print(f"解码: {decoded}")
    print()


# ============================================
# 2.5 Byte Pair Encoding (BPE)
# ============================================

def demo_bpe():
    """
    演示BPE tokenizer
    """
    print("=" * 60)
    print("2.5 BPE Tokenizer演示")
    print("=" * 60)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text = "Hello, do you like tea?  In the sunlit terraces of someunknownPlace."
    
    integers = tokenizer.encode(text, allowed_special={"", "<|unk|>"})
    strings = tokenizer.decode(integers)
    
    print(f"原文: {text}")
    print(f"编码后token数: {len(integers)}")
    print(f"编码: {integers}")
    print(f"解码: {strings}")
    
    unknown_word = "Akwirwierer"
    encoded_unknown = tokenizer.encode(unknown_word, allowed_special={"", "<|unk|>"})
    decoded_unknown = tokenizer.decode(encoded_unknown)
    
    print(f"\n未知词 '{unknown_word}' 的编码: {encoded_unknown}")
    print(f"解码后: {decoded_unknown}")
    print()


# ============================================
# 2.6 滑动窗口数据采样
# ============================================

class GPTDatasetV1(Dataset):
    """
    GPT数据集: 使用滑动窗口生成训练样本
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={"", "<|unk|>"})
        
        assert len(token_ids) > max_length, "文本太短"
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    """
    创建数据加载器
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def demo_dataloader():
    """
    演示DataLoader的使用
    """
    print("=" * 60)
    print("2.6 DataLoader演示")
    print("=" * 60)
    
    with open("learning/ch02-text-data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    
    print(f"Batch size: 8, Max length: 4, Stride: 4")
    print(f"\nInputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"\n第一个batch的输入:\n{inputs}")
    print(f"\n第一个batch的目标:\n{targets}")
    print()


# ============================================
# 2.7 Token Embedding
# ============================================

def demo_token_embedding():
    """
    演示Token Embedding
    """
    print("=" * 60)
    print("2.7 Token Embedding演示")
    print("=" * 60)
    
    vocab_size = 6
    output_dim = 3
    
    torch.manual_seed(123)
    embedding_layer = nn.Embedding(vocab_size, output_dim)
    
    print(f"Embedding层权重矩阵 (vocab_size={vocab_size}, output_dim={output_dim}):")
    print(embedding_layer.weight)
    
    input_ids = torch.tensor([2, 3, 5, 1])
    embeddings = embedding_layer(input_ids)
    
    print(f"\n输入IDs: {input_ids}")
    print(f"Embedding结果 shape: {embeddings.shape}")
    print(f"Embedding结果:\n{embeddings}")
    print()


# ============================================
# 2.8 Positional Embedding
# ============================================

def demo_positional_embedding():
    """
    演示Positional Embedding
    """
    print("=" * 60)
    print("2.8 Positional Embedding演示")
    print("=" * 60)
    
    vocab_size = 50257
    output_dim = 256
    max_length = 4
    
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(max_length, output_dim)
    
    with open("learning/ch02-text-data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(raw_text, allowed_special={"", "<|unk|>"})
    
    torch.manual_seed(123)
    
    context_length = max_length
    token_ids_sample = token_ids[:context_length]
    
    token_embeddings = token_embedding_layer(torch.tensor(token_ids_sample))
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    
    input_embeddings = token_embeddings + pos_embeddings
    
    print(f"词表大小: {vocab_size}")
    print(f"嵌入维度: {output_dim}")
    print(f"最大长度: {max_length}")
    print(f"\nToken IDs (前{context_length}个): {token_ids_sample}")
    print(f"\nToken Embeddings shape: {token_embeddings.shape}")
    print(f"Positional Embeddings shape: {pos_embeddings.shape}")
    print(f"最终 Input Embeddings shape: {input_embeddings.shape}")
    
    print(f"\n第一个token的embedding前5维:")
    print(f"  Token embedding: {token_embeddings[0, :5]}")
    print(f"  Positional embedding: {pos_embeddings[0, :5]}")
    print(f"  Combined: {input_embeddings[0, :5]}")
    print()


# ============================================
# 完整流程演示
# ============================================

def demo_complete_pipeline():
    """
    演示完整的文本处理流程
    """
    print("=" * 60)
    print("完整流程演示")
    print("=" * 60)
    
    text = "Hello, world! This is a test."
    print(f"原始文本: {text}")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text, allowed_special={"", "<|unk|>"})
    print(f"\n1. Tokenization -> Token IDs: {token_ids}")
    
    vocab_size = 50257
    output_dim = 256
    max_length = len(token_ids)
    
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(max_length, output_dim)
    
    token_embeddings = token_embedding_layer(torch.tensor(token_ids))
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    
    print(f"\n2. Token Embedding shape: {token_embeddings.shape}")
    print(f"3. Positional Embedding shape: {pos_embeddings.shape}")
    
    input_embeddings = token_embeddings + pos_embeddings
    print(f"\n4. 最终 Input Embeddings shape: {input_embeddings.shape}")
    
    print(f"\n✓ 文本处理完成! 可以送入Transformer进行处理")
    print()


# ============================================
# 主函数
# ============================================

def main():
    print("\n" + "=" * 60)
    print("第2章：文本数据处理 - 学习代码")
    print("=" * 60 + "\n")
    
    demo_tokenization()
    demo_tokenizer()
    demo_bpe()
    demo_dataloader()
    demo_token_embedding()
    demo_positional_embedding()
    demo_complete_pipeline()
    
    print("=" * 60)
    print("第2章学习完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
