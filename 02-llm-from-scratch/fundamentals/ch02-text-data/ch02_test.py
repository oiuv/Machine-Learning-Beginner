"""
Chapter 2 Test Script

Run this script to verify Chapter 2 code
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ch02_code import (
    simple_tokenize,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    GPTDatasetV1,
    create_dataloader_v1,
)


def test_tokenization():
    """Test basic tokenization"""
    print("Test 1: Basic tokenization...")
    
    text = "Hello, world. Is this-- a test?"
    tokens = simple_tokenize(text)
    
    expected = ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
    assert tokens == expected, f"Tokenization failed: {tokens} != {expected}"
    
    print("  [OK] Basic tokenization correct")
    return True


def test_tokenizer_v1():
    """Test SimpleTokenizerV1"""
    print("\nTest 2: SimpleTokenizerV1...")
    
    vocab = {'hello': 0, 'world': 1, ',': 2, '!': 3}
    tokenizer = SimpleTokenizerV1(vocab)
    
    text = "hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    assert ids == [0, 2, 1, 3], f"Encoding failed: {ids}"
    assert decoded == "hello, world!", f"Decoding failed: {decoded}"
    
    print("  [OK] Encode/Decode correct")
    return True


def test_tokenizer_v2():
    """Test SimpleTokenizerV2 (handles unknown words)"""
    print("\nTest 3: SimpleTokenizerV2 (unknown word handling)...")
    
    vocab = {'hello': 0, 'world': 1, ',': 2, '!': 3, '<|unk|>': 4}
    tokenizer = SimpleTokenizerV2(vocab)
    
    text = "hello, unknown!"
    ids = tokenizer.encode(text)
    
    assert 4 in ids, f"Unknown word not handled correctly: {ids}"
    
    print("  [OK] Unknown word handling correct")
    return True


def test_bpe():
    """Test BPE tokenizer"""
    print("\nTest 4: BPE Tokenizer...")
    
    import tiktoken
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    assert decoded == text, f"BPE encode/decode mismatch: {decoded} != {text}"
    
    print(f"  Text: {text}")
    print(f"  Token count: {len(ids)}")
    print("  [OK] BPE tokenizer correct")
    return True


def test_dataloader():
    """Test DataLoader"""
    print("\nTest 5: DataLoader...")
    
    import torch
    
    txt = "This is a sample text for testing the dataloader functionality."
    
    dataloader = create_dataloader_v1(
        txt, batch_size=2, max_length=4, stride=2, shuffle=False
    )
    
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    
    assert inputs.shape == (2, 4), f"Input shape error: {inputs.shape}"
    assert targets.shape == (2, 4), f"Target shape error: {targets.shape}"
    
    print(f"  Batch size: 2, Max length: 4")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    print("  [OK] DataLoader correct")
    return True


def test_embeddings():
    """Test Token and Positional Embeddings"""
    print("\nTest 6: Embeddings...")
    
    import torch
    import torch.nn as nn
    
    vocab_size = 100
    output_dim = 64
    seq_length = 10
    
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(seq_length, output_dim)
    
    token_ids = torch.randint(0, vocab_size, (seq_length,))
    token_embeddings = token_embedding_layer(token_ids)
    pos_embeddings = pos_embedding_layer(torch.arange(seq_length))
    
    input_embeddings = token_embeddings + pos_embeddings
    
    assert input_embeddings.shape == (seq_length, output_dim), \
        f"Embedding shape error: {input_embeddings.shape}"
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {output_dim}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Final shape: {input_embeddings.shape}")
    print("  [OK] Embeddings correct")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Chapter 2 Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_tokenization,
        test_tokenizer_v1,
        test_tokenizer_v2,
        test_bpe,
        test_dataloader,
        test_embeddings,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
