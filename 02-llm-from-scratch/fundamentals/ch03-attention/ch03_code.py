"""
Chapter 3: Attention Mechanisms - Complete Code

Run this code to learn attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================
# 3.2 Simple Self-Attention (No Trainable Weights)
# ============================================

def simple_self_attention():
    """
    Simple self-attention without trainable weights
    """
    print("=" * 60)
    print("3.2 Simple Self-Attention (No Trainable Weights)")
    print("=" * 60)
    
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ])
    
    print(f"Input shape: {inputs.shape}")
    
    query = inputs[1]  # 2nd input token as query
    
    # Step 1: Compute attention scores
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    
    print(f"\nStep 1: Attention scores for query 2:")
    print(f"  {attn_scores_2}")
    
    # Step 2: Normalize with softmax
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print(f"\nStep 2: Attention weights (normalized):")
    print(f"  {attn_weights_2}")
    print(f"  Sum: {attn_weights_2.sum()}")
    
    # Step 3: Compute context vector
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    
    print(f"\nStep 3: Context vector for position 2:")
    print(f"  {context_vec_2}")
    
    # More efficient: compute all context vectors at once
    print(f"\n\nEfficient computation (matrix multiplication):")
    
    # All attention scores
    attn_scores = inputs @ inputs.T
    print(f"Attention scores matrix shape: {attn_scores.shape}")
    
    # All attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(f"Attention weights matrix shape: {attn_weights.shape}")
    
    # All context vectors
    all_context_vecs = attn_weights @ inputs
    print(f"All context vectors shape: {all_context_vecs.shape}")
    print(f"\nFirst 3 context vectors:")
    print(all_context_vecs[:3])
    
    print()


# ============================================
# 3.3 Self-Attention with Trainable Weights
# ============================================

class SelfAttention_v1(nn.Module):
    """
    Self-attention mechanism with trainable weights
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        # Project to Query, Key, Value
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        
        # Compute attention scores
        attn_scores = queries @ keys.T
        
        # Scale by sqrt(d_k)
        d_k = keys.shape[-1]
        attn_scores = attn_scores / math.sqrt(d_k)
        
        # Normalize with softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Self-attention using nn.Linear layers
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_scores = attn_scores / math.sqrt(d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        context_vec = attn_weights @ values
        return context_vec


def demo_self_attention_with_weights():
    """
    Demonstrate self-attention with trainable weights
    """
    print("=" * 60)
    print("3.3 Self-Attention with Trainable Weights")
    print("=" * 60)
    
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])
    
    torch.manual_seed(123)
    d_in, d_out = 3, 2
    
    sa_v1 = SelfAttention_v1(d_in, d_out)
    context_v1 = sa_v1(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {context_v1.shape}")
    print(f"\nContext vectors (first 3):")
    print(context_v1[:3])
    
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    context_v2 = sa_v2(inputs)
    
    print(f"\nUsing nn.Linear (first 3):")
    print(context_v2[:3])
    
    print()


# ============================================
# 3.4 Causal Attention
# ============================================

class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models
    """
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (not a parameter)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        d_k = keys.shape[-1]
        attn_scores = attn_scores / math.sqrt(d_k)
        
        # Apply causal mask
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec


def demo_causal_attention():
    """
    Demonstrate causal attention
    """
    print("=" * 60)
    print("3.4 Causal Attention")
    print("=" * 60)
    
    batch = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]).unsqueeze(0)  # Add batch dimension
    
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    
    ca = CausalAttention(d_in, d_out, context_length)
    context_vecs = ca(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {context_vecs.shape}")
    print(f"\nCausal mask:")
    print(ca.mask.int())
    print(f"\nContext vectors:")
    print(context_vecs)
    
    print()


# ============================================
# 3.5 Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Split into multiple heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        d_k = self.head_dim
        attn_scores = attn_scores / math.sqrt(d_k)
        
        # Apply causal mask
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # Concatenate heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # Final projection
        context_vec = self.out_proj(context_vec)
        
        return context_vec


def demo_multihead_attention():
    """
    Demonstrate multi-head attention
    """
    print("=" * 60)
    print("3.5 Multi-Head Attention")
    print("=" * 60)
    
    batch = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]).unsqueeze(0)
    
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out = 3, 4
    num_heads = 2
    
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)
    context_vecs = mha(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {context_vecs.shape}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {mha.head_dim}")
    print(f"\nContext vectors:")
    print(context_vecs)
    
    print()


# ============================================
# Summary and Comparison
# ============================================

def demo_summary():
    """
    Summary of all attention mechanisms
    """
    print("=" * 60)
    print("Summary: Attention Mechanisms Comparison")
    print("=" * 60)
    
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])
    
    batch = inputs.unsqueeze(0)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    
    print("\n1. Simple Self-Attention (no weights):")
    attn_scores = inputs @ inputs.T
    attn_weights = torch.softmax(attn_scores, dim=-1)
    context_simple = attn_weights @ inputs
    print(f"   Output shape: {context_simple.shape}")
    print(f"   Parameters: 0 (no trainable weights)")
    
    print("\n2. Self-Attention with Weights:")
    torch.manual_seed(123)
    sa = SelfAttention_v2(d_in, d_out)
    context_sa = sa(inputs)
    print(f"   Output shape: {context_sa.shape}")
    print(f"   Parameters: {sum(p.numel() for p in sa.parameters())}")
    
    print("\n3. Causal Attention:")
    torch.manual_seed(123)
    ca = CausalAttention(d_in, d_out, context_length)
    context_ca = ca(batch)
    print(f"   Output shape: {context_ca.shape}")
    print(f"   Parameters: {sum(p.numel() for p in ca.parameters())}")
    print(f"   Feature: Masks future positions")
    
    print("\n4. Multi-Head Attention:")
    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_mha = mha(batch)
    print(f"   Output shape: {context_mha.shape}")
    print(f"   Parameters: {sum(p.numel() for p in mha.parameters())}")
    print(f"   Feature: Multiple attention heads")
    
    print()


# ============================================
# Main
# ============================================

def main():
    print("\n" + "=" * 60)
    print("Chapter 3: Attention Mechanisms - Learning Code")
    print("=" * 60 + "\n")
    
    simple_self_attention()
    demo_self_attention_with_weights()
    demo_causal_attention()
    demo_multihead_attention()
    demo_summary()
    
    print("=" * 60)
    print("Chapter 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
