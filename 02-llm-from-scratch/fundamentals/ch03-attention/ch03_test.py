"""
Chapter 3 Test Script

Run this script to verify Chapter 3 code
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ch03_code import (
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttention,
)


def test_self_attention_v2():
    """Test self-attention with trainable weights"""
    print("Test 1: Self-Attention v2...")
    
    torch.manual_seed(123)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
    ])
    
    d_in, d_out = 3, 2
    sa = SelfAttention_v2(d_in, d_out)
    
    output = sa(inputs)
    
    assert output.shape == (3, 2), f"Output shape error: {output.shape}"
    
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {output.shape}")
    print("  [OK] Self-Attention v2 correct")
    return True


def test_causal_attention():
    """Test causal attention"""
    print("\nTest 2: Causal Attention...")
    
    torch.manual_seed(123)
    batch = torch.randn(2, 6, 3)  # batch_size=2, seq_len=6, d_in=3
    
    d_in, d_out = 3, 2
    context_length = batch.shape[1]
    
    ca = CausalAttention(d_in, d_out, context_length)
    output = ca(batch)
    
    assert output.shape == batch.shape[:2] + (d_out,), \
        f"Output shape error: {output.shape}"
    
    print(f"  Input shape: {batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Causal mask shape: {ca.mask.shape}")
    print("  [OK] Causal Attention correct")
    return True


def test_multihead_attention():
    """Test multi-head attention"""
    print("\nTest 3: Multi-Head Attention...")
    
    torch.manual_seed(123)
    batch = torch.randn(2, 6, 3)  # batch_size=2, seq_len=6, d_in=3
    
    d_in, d_out = 3, 4
    context_length = batch.shape[1]
    num_heads = 2
    
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)
    output = mha(batch)
    
    assert output.shape == batch.shape[:2] + (d_out,), \
        f"Output shape error: {output.shape}"
    
    print(f"  Input shape: {batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {mha.head_dim}")
    print("  [OK] Multi-Head Attention correct")
    return True


def test_attention_masking():
    """Test that causal masking works correctly"""
    print("\nTest 4: Attention Masking...")
    
    torch.manual_seed(123)
    batch = torch.randn(1, 4, 3)
    
    d_in, d_out = 3, 2
    context_length = batch.shape[1]
    
    ca = CausalAttention(d_in, d_out, context_length)
    
    # Get attention scores before masking
    queries = ca.W_query(batch)
    keys = ca.W_key(batch)
    attn_scores = queries @ keys.transpose(1, 2)
    
    # Check that mask is upper triangular
    mask = ca.mask[:context_length, :context_length]
    expected_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
    
    assert torch.equal(mask, expected_mask), "Mask is not upper triangular"
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask (True = masked):")
    print(f"  {mask.int()}")
    print("  [OK] Attention masking correct")
    return True


def test_gradient_flow():
    """Test that gradients flow through attention"""
    print("\nTest 5: Gradient Flow...")
    
    torch.manual_seed(123)
    batch = torch.randn(2, 4, 3, requires_grad=True)
    
    d_in, d_out = 3, 2
    context_length = batch.shape[1]
    
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    output = mha(batch)
    
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert batch.grad is not None, "No gradient for input"
    assert mha.W_query.weight.grad is not None, "No gradient for W_query"
    
    print(f"  Input gradient shape: {batch.grad.shape}")
    print(f"  W_query gradient shape: {mha.W_query.weight.grad.shape}")
    print("  [OK] Gradient flow correct")
    return True


def test_deterministic_output():
    """Test that same input produces same output"""
    print("\nTest 6: Deterministic Output...")
    
    torch.manual_seed(123)
    batch = torch.randn(1, 4, 3)
    
    d_in, d_out = 3, 2
    context_length = batch.shape[1]
    
    torch.manual_seed(456)
    ca1 = CausalAttention(d_in, d_out, context_length)
    output1 = ca1(batch.clone())
    
    torch.manual_seed(456)
    ca2 = CausalAttention(d_in, d_out, context_length)
    output2 = ca2(batch.clone())
    
    assert torch.allclose(output1, output2), "Outputs are not deterministic"
    
    print("  [OK] Deterministic output correct")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Chapter 3 Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_self_attention_v2,
        test_causal_attention,
        test_multihead_attention,
        test_attention_masking,
        test_gradient_flow,
        test_deterministic_output,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
