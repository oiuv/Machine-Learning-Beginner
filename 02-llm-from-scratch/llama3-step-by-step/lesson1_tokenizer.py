"""
阶段 1: 文本分词 (Tokenization)
从零开始理解文本如何变成数字
"""

import torch

# ============================================
# 第一部分：最简单的字符级分词
# ============================================

print("=" * 50)
print("第一部分：字符级分词（最简单）")
print("=" * 50)

# 定义我们的小词汇表（只包含几个字符）
# 每个字符对应一个唯一的数字
vocab = {
    '<PAD>': 0,   # 填充符，用于对齐
    '<START>': 1, # 开始符
    '<END>': 2,   # 结束符
    '我': 3,
    '爱': 4,
    '学': 5,
    '习': 6,
    'A': 7,
    'I': 8,
    ' ': 9,       # 空格
}

# 创建反向映射：数字 → 字符
id_to_char = {v: k for k, v in vocab.items()}

print("\n【词汇表】字符 → 数字:")
for char, idx in vocab.items():
    print(f"  '{char}' → {idx}")

# 编码函数：文本 → 数字列表
def encode(text):
    """把文本转换成数字列表"""
    tokens = [vocab['<START>']]  # 先加开始符
    for char in text:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(vocab['<PAD>'])  # 未知字符用PAD代替
    tokens.append(vocab['<END>'])  # 加结束符
    return tokens

# 解码函数：数字列表 → 文本
def decode(tokens):
    """把数字列表转换回文本"""
    chars = []
    for token in tokens:
        char = id_to_char.get(token, '?')
        if char not in ['<PAD>', '<START>', '<END>']:  # 跳过特殊符号
            chars.append(char)
    return ''.join(chars)

# 测试编码和解码
text = "我爱 AI"
print(f"\n【测试】原始文本: '{text}'")

tokens = encode(text)
print(f"【测试】编码结果: {tokens}")
print(f"        解释: ", end="")
for t in tokens:
    print(f"{id_to_char[t]}({t})", end=" ")
print()

decoded_text = decode(tokens)
print(f"【测试】解码结果: '{decoded_text}'")

# 转换成 PyTorch 张量（这是后续步骤需要的格式）
tokens_tensor = torch.tensor(tokens)
print(f"\n【张量】tokens 变成 PyTorch 张量: {tokens_tensor}")
print(f"        形状: {tokens_tensor.shape}")
print(f"        数据类型: {tokens_tensor.dtype}")

# ============================================
# 第二部分：单词级分词（更接近真实模型）
# ============================================

print("\n" + "=" * 50)
print("第二部分：单词级分词（更接近真实）")
print("=" * 50)

# 定义单词级词汇表
word_vocab = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '我': 3,
    '喜欢': 4,
    '学习': 5,
    '人工智能': 6,
    '非常': 7,
    '有趣': 8,
}

word_to_id = word_vocab
id_to_word = {v: k for k, v in word_vocab.items()}

print("\n【词汇表】单词 → 数字:")
for word, idx in word_vocab.items():
    print(f"  '{word}' → {idx}")

# 简单的分词：按空格和常见词分割
def simple_tokenize(text):
    """简单的分词函数"""
    # 这里用简单规则，真实模型用更复杂的算法
    words = []
    i = 0
    while i < len(text):
        matched = False
        # 尝试匹配最长的词
        for length in range(min(5, len(text) - i), 0, -1):
            substr = text[i:i+length]
            if substr in word_vocab:
                words.append(substr)
                i += length
                matched = True
                break
        if not matched:
            i += 1
    return words

def encode_words(text):
    """单词级编码"""
    words = simple_tokenize(text)
    tokens = [word_to_id['<START>']]
    for word in words:
        tokens.append(word_to_id.get(word, word_to_id['<PAD>']))
    tokens.append(word_to_id['<END>'])
    return tokens, words

def decode_words(tokens):
    """单词级解码"""
    words = []
    for token in tokens:
        word = id_to_word.get(token, '?')
        if word not in ['<PAD>', '<START>', '<END>']:
            words.append(word)
    return ' '.join(words)

# 测试
text2 = "我喜欢学习人工智能"
print(f"\n【测试】原始文本: '{text2}'")

tokens2, words = encode_words(text2)
print(f"【测试】分词结果: {words}")
print(f"【测试】编码结果: {tokens2}")
print(f"        解释: ", end="")
for t in tokens2:
    print(f"{id_to_word[t]}({t})", end=" ")
print()

decoded2 = decode_words(tokens2)
print(f"【测试】解码结果: '{decoded2}'")

# ============================================
# 第三部分：理解 BPE 分词（Llama3 使用）
# ============================================

print("\n" + "=" * 50)
print("第三部分：BPE 分词（Llama3 实际使用）")
print("=" * 50)

print("""
【概念】BPE (Byte Pair Encoding) 是什么？

BPE 是一种"子词"分词方法，介于字符级和单词级之间：
- 常见单词保持完整（如 "the", "and"）
- 罕见单词拆成子词（如 "unhappiness" → "un" + "happiness"）

优点：
1. 词汇表大小可控（不会无限增长）
2. 能处理任何新词（通过子词组合）
3. 平衡了表达能力和效率

【例子】
"unhappiness" 可能被分成：
- ["un", "happiness"]  或
- ["un", "happy", "ness"]

Llama3 的词汇表大小：128,256
""")

# 模拟简单的 BPE 分词
bpe_vocab = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    'the': 3,
    'answer': 4,
    'to': 5,
    'life': 6,
    'un': 7,
    'happiness': 8,
    'happy': 9,
    'ness': 10,
    'is': 11,
    '42': 12,
}

bpe_id_to_token = {v: k for k, v in bpe_vocab.items()}

def bpe_tokenize(text):
    """模拟 BPE 分词"""
    # 简化版：按空格分词，然后查表
    words = text.lower().split()
    tokens = []
    for word in words:
        if word in bpe_vocab:
            tokens.append(bpe_vocab[word])
        else:
            # 尝试拆分子词（简化处理）
            if word.startswith('un') and 'un' in bpe_vocab:
                tokens.append(bpe_vocab['un'])
                rest = word[2:]
                if rest in bpe_vocab:
                    tokens.append(bpe_vocab[rest])
                else:
                    tokens.append(bpe_vocab['<PAD>'])
            else:
                tokens.append(bpe_vocab['<PAD>'])
    return tokens

# 测试 BPE
text3 = "the answer to life is 42"
print(f"\n【测试】原始文本: '{text3}'")
bpe_tokens = bpe_tokenize(text3)
print(f"【测试】BPE 编码: {bpe_tokens}")
print(f"        解释: ", end="")
for t in bpe_tokens:
    print(f"'{bpe_id_to_token[t]}'({t})", end=" ")
print()

text4 = "unhappiness"
print(f"\n【测试】原始文本: '{text4}'")
bpe_tokens2 = bpe_tokenize(text4)
print(f"【测试】BPE 编码: {bpe_tokens2}")
print(f"        解释: ", end="")
for t in bpe_tokens2:
    print(f"'{bpe_id_to_token[t]}'({t})", end=" ")
print("\n        注意：'unhappiness' 被拆成了 'un' + 'happiness'")

# ============================================
# 第四部分：总结和可视化
# ============================================

print("\n" + "=" * 50)
print("第四部分：总结")
print("=" * 50)

print("""
【本阶段重点】

1. Tokenization 是文本 → 数字的转换过程
2. 有三种粒度：
   - 字符级：简单但序列长
   - 单词级：直观但词汇表大
   - 子词级 (BPE)：平衡，现代 LLM 都用这个

3. 特殊 Token：
   - <START>: 序列开始
   - <END>: 序列结束
   - <PAD>: 填充对齐

4. 最终输出是 PyTorch 张量，形状为 [序列长度]

【下一步】
这些数字 (tokens) 接下来要变成向量 (embeddings)
""")

# 可视化：展示文本到数字的转换流程
print("\n【流程图】文本 → Token → 张量")
print("-" * 40)
print("'我爱 AI'")
print("    ↓")
print("分词: ['<START>', '我', '爱', ' ', 'A', 'I', '<END>']")
print("    ↓")
print("编码: [1, 3, 4, 9, 7, 8, 2]")
print("    ↓")
print(f"张量: tensor([1, 3, 4, 9, 7, 8, 2])")
print(f"形状: torch.Size([7])")
print("-" * 40)
