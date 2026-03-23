# 10.3 The Transformer Architecture

## 🎯 Quick Overview
- **The "Attention is All You Need" Paper**: The shift from recurrence to parallel processing
- **Self-Attention**: Computing relationships between all words in a sequence simultaneously
- **Multi-Head Attention**: Learning multiple representation subspaces
- **Positional Encoding**: Adding sequence order information without recurrence
- **Foundation for**: BERT, GPT, LLaMA, Stable Diffusion, and Vision Transformers

---

## 1. Why Transformers?

Before 2017, NLP relied on RNNs/LSTMs.
- **Problem with RNNs**: Sequential processing (can't parallelize). $O(n)$ path length between distant words.
- **Transformer Solution**: Parallel processing. $O(1)$ path length between any two words via Self-Attention.

---

## 2. Core Components

### 2.1 Self-Attention (Scaled Dot-Product Attention)
For every word, we create three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.
1. **Dot Product**: $Q \cdot K^T$ (measures similarity).
2. **Scale**: Divide by $\sqrt{d_k}$ (prevents gradients from vanishing/exploding in Softmax).
3. **Softmax**: Converts scores to probabilities (weights).
4. **Weighted Sum**: Multiply weights by $V$.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

### 2.2 Multi-Head Attention
Instead of one big attention calculation, we split Q, K, V into multiple "heads."
- **Benefit**: Allows the model to attend to different information (e.g., one head focuses on grammar, another on semantic meaning).

### 2.3 Positional Encoding
Since Transformers process words in parallel, they don't "know" word order. We add a fixed mathematical signal (Sines and Cosines) to the input embeddings to provide "coordinates" for each word.

---

## 3. The Architecture (Encoder-Decoder)

### 3.1 The Encoder
Processes the input. Consists of:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network (FFN)
- **Layer Normalization** and **Residual Connections** (crucial for deep stacking).

### 3.2 The Decoder
Generates the output. Consists of:
- **Masked** Multi-Head Self-Attention (prevents "cheating" by looking at future words).
- Encoder-Decoder Attention (looks at the encoder's output).
- Position-wise FFN.

---

## 💻 Python Code Examples

### 1. Simple Self-Attention Mechanism (NumPy Logic)
```python
import numpy as np

def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    # 1. Dot product
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    # 2. Softmax
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    # 3. Weighted sum
    output = np.matmul(weights, V)
    return output, weights

# Example: 3 words, 4-dim embeddings
X = np.random.randn(3, 4) 
Q = K = V = X
out, w = self_attention(Q, K, V)
print("Attention Weights:\n", w)
```

### 2. Transformer Block (PyTorch Style)
```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # 1. Attention + Residual + Norm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(attn_out + x)
        # 2. FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)
        return x
```

---

## 📊 Summary Table

| Feature | RNN / LSTM | Transformer |
|---------|------------|-------------|
| **Processing** | Sequential | Parallel |
| **Path Length** | $O(n)$ | $O(1)$ |
| **Complexity** | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| **Long-range info** | Struggles | Excellent |
| **Scalability** | Hard | Highly Scalable |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Encoder stack | BERT (Understanding) |
| Decoder stack | GPT (Generation) |
| Cross-attention | Dall-E (Text-to-Image) |
| FlashAttention | Scaling LLMs to 1M+ context |

---

## ❓ Quick Check Questions

1. Why do we divide the dot product by $\sqrt{d_k}$ in the attention formula?
2. What is the difference between "Self-Attention" and "Cross-Attention"?
3. Why is "Masking" necessary in the Transformer Decoder?
4. How does a Transformer "understand" the difference between "The dog bit the man" and "The man bit the dog"?
5. What is the computational complexity of the Transformer relative to sequence length ($n$)?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Scaling**: It prevents the dot products from growing too large in magnitude, which would push the Softmax function into regions with extremely small gradients, causing training to stall.
2. **Self-Attention**: Q, K, and V all come from the same sequence (e.g., input sentence). **Cross-Attention**: Q comes from one sequence (Decoder), while K and V come from another (Encoder).
3. **Causal Masking**: During training, the decoder should not see "future" tokens. Masking sets the attention scores for future positions to $-\infty$ so the model only learns to predict based on previous tokens.
4. Through **Positional Encoding**. It adds unique vector signals to the word embeddings that represent their specific indices in the sequence, allowing the model to distinguish order.
5. It is **$O(n^2)$**, where $n$ is the sequence length. This quadratic complexity is the reason why very long context windows (like 128k+) require specialized optimizations like FlashAttention.

</details>

---

**Status:** ✅ Complete
**Next:** Pre-trained Language Models (BERT, GPT, LLaMA)
