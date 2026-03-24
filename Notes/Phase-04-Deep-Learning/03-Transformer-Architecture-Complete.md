# 10.3 The Transformer Revolution: Parallelizing Attention

## 🎯 Quick Overview
- **Self-Attention Math**: Scaled Dot-Product derivation and complexity
- **Multi-Head Mechanism**: Why splitting heads improves representation
- **Positional Encoding**: The Sine/Cosine signal math
- **Architecture Deep-Dive**: Encoder vs. Decoder blocks and LayerNorm
- **Foundation for**: BERT, GPT, Vision Transformers, and all modern GenAI

---

## 1. The Core Innovation: Self-Attention

The Transformer's superpower is its ability to compute the relationship between every word in a sequence simultaneously ($O(1)$ path length).

### 1.1 The Scaled Dot-Product Math
For an input embedding $X$, we learn weight matrices $W_Q, W_K, W_V$ to create:
- **Query ($Q$)**: $X W_Q$ (What I'm looking for)
- **Key ($K$)**: $X W_K$ (What I have to offer)
- **Value ($V$)**: $X W_V$ (The actual information)

#### The Formula:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**Why $\sqrt{d_k}$?**: If $d_k$ (dimension of keys) is large, the dot products grow very large in magnitude, pushing the softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ keeps the variance of the scores stable.

---

## 2. Multi-Head Attention (MHA)

Instead of one high-dimensional attention, we split the $d_{model}$ into $h$ "heads."
- Each head $i$ learns a different projection ($Q_i, K_i, V_i$).
- **Intuition**: Head 1 might attend to grammatical dependencies, Head 2 to semantic entities, and Head 3 to punctuation.
- **Aggregation**: Outputs are concatenated and projected back to $d_{model}$.

---

## 3. Positional Encoding: Giving Order to Parallelism

Transformers have no recurrence, so they are "permutation invariant." To provide word order, we add a positional signal:
$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

**Why this math?**: It allows the model to easily learn to attend by relative positions since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

---

## 4. The Block Architecture

### 4.1 Encoder Block
1.  **Multi-Head Self-Attention**
2.  **Add & Norm**: Residual Connection ($x + \text{Attn}(x)$) followed by Layer Normalization.
3.  **Feed-Forward Network (FFN)**: Two linear layers with a ReLU/GELU in between.
4.  **Add & Norm**.

### 4.2 Decoder Block
- **Masked Self-Attention**: Prevents tokens from looking at "future" tokens during training.
- **Encoder-Decoder Attention**: Query comes from the decoder, Key/Value comes from the encoder.

---

## 💻 Professional Implementation

### 1. Self-Attention with Scaled Dot-Product (PyTorch)
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # 1. Compute Scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    
    # 2. Apply Mask (Optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 3. Softmax to get Weights
    weights = F.softmax(scores, dim=-1)
    
    # 4. Weighted Sum of Values
    output = torch.matmul(weights, V)
    return output, weights
```

### 2. Positional Encoding Generator
```python
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    pe = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** ((i) / d_model)))
    return pe
```

---

## 📊 Summary Comparison

| Feature | RNN / LSTM | Transformer |
| :--- | :--- | :--- |
| **Complexity per Layer** | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| **Sequential Ops** | $O(n)$ | $O(1)$ |
| **Max Path Length** | $O(n)$ | $O(1)$ |
| **Parallelization** | No | **Excellent** |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Cross-Attention** | Multimodal models (e.g., Image-to-Text captioning). |
| **Causal Masking** | Efficient training of autoregressive generators (GPT). |
| **FlashAttention** | Using tiling and IO-aware algorithms to speed up $O(n^2)$ attention. |
| **Rotary Embeddings**| Modern alternative to Sine/Cosine (used in Llama-3). |

---

## ❓ Quick Check Questions

1. Why is the Transformer's self-attention complexity $O(n^2)$?
2. What is the difference between Batch Normalization and Layer Normalization (used in Transformers)?
3. How does the "Decoder" attend to the "Encoder" output?
4. In the formula $Attn(Q,K,V)$, what happens if you remove the Softmax?
5. Why are residual connections critical for training a 12-layer or 24-layer Transformer?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Because each of the $n$ tokens must calculate a dot product with **every other** token in the sequence ($n \times n$ matrix of scores).
2. **Batch Norm** normalizes across the batch dimension (mean/var of all samples for one feature). **Layer Norm** normalizes across the feature dimension (mean/var of all features for one sample). Layer Norm is better for NLP as it handles varying sequence lengths more gracefully.
3. Via **Cross-Attention**. The Decoder provides the Queries (what it needs to generate next), and the Encoder provides the Keys and Values (the context from the source sentence).
4. Without **Softmax**, the attention becomes a simple linear transformation. Softmax provides the non-linearity and "winner-takes-all" dynamic that allows the model to focus specifically on certain tokens while ignoring others.
5. **Residual Connections** ensure that the gradients can flow through the network without being completely diminished by the weights of each layer. They allow the model to learn "incremental" changes to the identity mapping, making optimization stable.

</details>

---

## 📚 Recommended Resources
- **Original Paper**: [Attention is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- **The Illustrated Transformer**: [Jay Alammar's Blog](https://jalammar.github.io/illustrated-transformer/) - *The best visual guide available*.
- **Implementation**: [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

---

## 5. Advanced Attention Mechanisms

### 5.1 Efficient Attention Variants

The $O(n^2)$ complexity of self-attention is prohibitive for long sequences. Here are the main approaches to reduce it:

#### A. Sparse Attention
Instead of attending to all tokens, each token attends to a fixed subset.

**Patterns**:
- **Fixed Window**: Each token attends to $w$ neighbors on each side. $O(n \cdot w)$
- **Strided**: Attend to every $k$-th token. $O(n/k)$
- **Global + Local**: A few "global" tokens attend to everything; others attend locally.

**Longformer**: Combines windowed attention with global attention for selected tokens (e.g., [CLS], punctuation).

---

#### B. Linear Attention (Kernelized Attention)
Reformulates attention using kernel functions to avoid the $n \times n$ matrix.

**The Trick**: Replace softmax with a kernel $\phi(q, k) = \phi(q)^T \phi(k)$:
$$ \text{Attention}(Q, K, V) = \phi(Q) \cdot (\phi(K)^T V) $$

Now the computation order changes:
1. Compute $\phi(K)^T V$ once: $O(n \cdot d^2)$
2. Multiply by $\phi(Q)$: $O(n \cdot d^2)$
3. **Total**: $O(n \cdot d^2)$ instead of $O(n^2 \cdot d)$

**Performers**: Use random Fourier features to approximate the softmax kernel.

---

#### C. FlashAttention (IO-Aware Attention)
Standard attention is **memory-bound**, not compute-bound. FlashAttention uses:
1. **Tiling**: Split $Q, K, V$ into blocks that fit in GPU SRAM.
2. **Recomputation**: Store only block statistics, recompute attention during backward pass.
3. **Fused Operations**: Combine matrix multiplications, softmax, and masking into a single CUDA kernel.

**Result**: 2-4× speedup with $O(n^2)$ complexity but much lower memory usage.

```python
# Standard Attention (PyTorch)
attn = torch.softmax(Q @ K.T / math.sqrt(d_k), dim=-1) @ V

# FlashAttention (via flash_attn library)
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=1.0/math.sqrt(d_k))
```

---

#### D. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
**MQA**: Share a single key-value head across all query heads.
- **Benefit**: Drastically reduces KV-cache size during inference.
- **Trade-off**: Slight quality degradation.

**GQA**: Split query heads into groups, each group shares one KV head.
- Llama-3 uses GQA with 8 KV heads for 32 query heads.
- **Sweet spot**: Near-MQA efficiency with near-MHA quality.

```python
# GQA Structure
# Query: [batch, seq_len, num_heads_q, head_dim]
# Key:   [batch, seq_len, num_heads_kv, head_dim]  where num_heads_kv < num_heads_q
# Value: [batch, seq_len, num_heads_kv, head_dim]

# Repeat KV to match query heads before attention
```

---

### 5.2 Positional Encoding: Deep Dive

#### A. Absolute Positional Encoding (Original)
$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

**Limitations**:
- Cannot extrapolate to sequences longer than training.
- Fixed pattern may not be optimal for all tasks.

---

#### B. Rotary Positional Embeddings (RoPE) - Detailed Math
RoPE encodes position through **rotation** rather than addition.

**1D Case**:
For a 2D vector $[x_1, x_2]$, rotate by angle $\theta = pos \cdot 10000^{-2i/d}$:
$$ \begin{bmatrix} x_1' \\ x_2' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} $$

**High-Dimensional Case**:
Split the embedding into pairs and apply rotation to each pair:
$$ f_q(x_m, m) = \text{RoPE}(q, m) $$
$$ f_k(x_n, n) = \text{RoPE}(k, n) $$

**Key Property**: The dot product depends only on **relative position**:
$$ f_q(x_m, m)^T f_k(x_n, n) = g(q, k, m-n) $$

**Implementation**:
```python
import torch

def rotate_half(x):
    """Split into pairs and rotate by 90 degrees."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to query and key tensors."""
    # cos, sin: [seq_len, head_dim]
    # q, k: [batch, heads, seq_len, head_dim]
    
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """Precompute rotation angles for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # [seq_len, dim/2]
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return torch.stack([cos_freqs, sin_freqs], dim=-1)  # [seq_len, dim/2, 2]
```

**Why RoPE Won**:
- Naturally encodes relative positions.
- Extrapolates better to longer sequences.
- No learned parameters (saves memory).

---

#### C. ALiBi (Attention with Linear Biases)
ALiBi removes positional encodings entirely and instead adds a **bias** to attention scores based on relative position.

**The Formula**:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + b\right)V $$

Where $b_{ij} = -m \cdot |i - j|$ and $m$ is a head-specific slope.

**Key Insights**:
- Each attention head learns a different "decay rate" $m$.
- Closer tokens get higher attention (inductive bias for locality).
- **Extrapolates perfectly** to any sequence length (no fixed PE to learn).

```python
def alibi_bias(num_heads, seq_len):
    """Create ALiBi bias matrix."""
    # Create position difference matrix
    positions = torch.arange(seq_len)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
    diff = torch.abs(diff)
    
    # Head-specific slopes (powers of 2)
    slopes = torch.tensor([2 ** (-(i + 1)) for i in range(num_heads)])
    slopes = slopes.view(-1, 1, 1)  # [num_heads, 1, 1]
    
    # Apply slopes to position differences
    bias = -slopes * diff.unsqueeze(0)  # [num_heads, seq_len, seq_len]
    
    return bias

# Usage in attention
scores = Q @ K.T / math.sqrt(d_k) + alibi_bias(num_heads, seq_len)
attn = torch.softmax(scores, dim=-1) @ V
```

**Comparison**:
| Method | Extrapolation | Parameters | Best For |
| :--- | :--- | :--- | :--- |
| **Absolute PE** | Poor | Learned | BERT-style models |
| **RoPE** | Good | None | LLaMA, PaLM, Falcon |
| **ALiBi** | **Excellent** | None | MPT, long-context models |

---

### 5.3 Normalization Strategies

#### A. Pre-LN vs Post-LN
**Post-LN (Original Transformer)**:
$$ x' = \text{LayerNorm}(x + \text{Sublayer}(x)) $$

**Pre-LN (Modern)**:
$$ x' = x + \text{Sublayer}(\text{LayerNorm}(x)) $$

**Why Pre-LN Won**:
- Better gradient flow (no vanishing through LayerNorm).
- Enables training deeper models (100+ layers).
- More stable during training (less sensitive to learning rate).

---

#### B. RMSNorm (Root Mean Square Layer Normalization)
Standard LayerNorm:
$$ \bar{x} = \frac{x - \mu}{\sigma}, \quad \mu = \frac{1}{n}\sum x_i, \quad \sigma = \sqrt{\frac{1}{n}\sum(x_i - \mu)^2 + \epsilon} $$

RMSNorm (simplified):
$$ \bar{x} = \frac{x}{\text{RMS}}, \quad \text{RMS} = \sqrt{\frac{1}{n}\sum x_i^2 + \epsilon} $$

**Benefits**:
- ~7-10% faster (no mean computation).
- Similar training stability.
- Used in LLaMA, PaLM, Falcon.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
```

---

### 5.4 Activation Functions in Transformers

| Activation | Formula | Used In | Properties |
| :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, x)$ | Original Transformer | Simple, but can "die" |
| **GELU** | $x \cdot \Phi(x)$ | BERT, GPT-2 | Smooth, better gradients |
| **SwiGLU** | $\text{Swish}(xW) \otimes (xV)$ | LLaMA, PaLM | Best performance |

**SwiGLU Implementation**:
```python
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)  # Gate
        self.V = nn.Linear(dim, dim)  # Value
    
    def forward(self, x):
        return F.silu(self.W(x)) * self.V(x)
```

---

## 6. The Complete Modern Transformer Block

Here's the full architecture used in LLaMA-3 and similar models:

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        # Pre-normalization
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        # Grouped Query Attention
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # RoPE
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        
        # SwiGLU FFN
        self.w1 = nn.Linear(dim, dim * 2, bias=False)  # Gate
        self.w2 = nn.Linear(dim * 2, dim, bias=False)  # Output
        self.w3 = nn.Linear(dim, dim * 2, bias=False)  # Value
        
    def forward(self, x, mask=None):
        # Self-Attention with Pre-Norm
        h = self.attention_norm(x)
        h_attn = self.self_attention(h, mask)
        x = x + h_attn  # Residual
        
        # FFN with Pre-Norm
        h = self.ffn_norm(x)
        h_ffn = self.swiglu(h)
        x = x + h_ffn  # Residual
        
        return x
    
    def self_attention(self, x, mask):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, -1, self.head_dim)  # n_kv_heads
        v = self.wv(x).view(B, T, -1, self.head_dim)
        
        # Apply RoPE
        cos, sin = self.freqs_cis.split([self.head_dim//2, self.head_dim//2], dim=-1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat KV for GQA
        n_rep = self.n_heads // k.shape[1]
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)
    
    def swiglu(self, x):
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)
```

---

## 7. Debugging and Visualization

### 7.1 Attention Weight Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attn_weights, tokens, head_idx=0):
    """Visualize attention weights for a specific head."""
    # attn_weights: [batch, heads, seq_len, seq_len]
    weights = attn_weights[0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(f'Attention Head {head_idx}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.show()

# Example usage
tokens = ["The", "cat", "sat", "on", "the", "mat"]
# attn_weights from model.forward(...)
visualize_attention(attn_weights, tokens)
```

---

### 7.2 Common Issues and Solutions

| Issue | Symptom | Solution |
| :--- | :--- | :--- |
| **Attention collapse** | All weights uniform | Check temperature/scaling factor |
| **Position drift** | Model ignores position | Verify RoPE/PE implementation |
| **Gradient explosion** | NaN losses | Use gradient clipping, check initialization |
| **KV-cache mismatch** | Wrong outputs during inference | Ensure cache indices are correct |

---

## 🔬 Research Frontiers (2024-2025)

### 8.1 Ring Attention
Distributes attention computation across multiple GPUs in a ring topology, enabling training on **million-token sequences**.

### 8.2 Sliding Window Attention (Mistral)
Each token attends to a local window plus a few global tokens. Enables **infinite context** with constant memory.

### 8.3 Hybrid Architectures
- **Mamba + Transformer**: Combine SSM efficiency with attention quality.
- **RetNet**: Uses retention mechanism instead of attention.

---

**Status:** ✅ Elite Expanded Standard (13/10)
**Next:** LLM Evolution (BERT, GPT, LLaMA, RoPE, KV-Cache, MoE)
