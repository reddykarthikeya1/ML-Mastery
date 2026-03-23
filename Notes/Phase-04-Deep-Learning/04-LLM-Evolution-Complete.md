# 10.4 LLM Evolution: From BERT to Llama-3 & Optimization

## 🎯 Quick Overview
- **Encoder-only (BERT)**: Bidirectional context and pre-training objectives (MLM, NSP)
- **Decoder-only (GPT)**: Scaling laws and the emergence of In-Context Learning
- **Modern Architectures**: RoPE (Rotary Embeddings), SwiGLU, and RMSNorm
- **Inference Optimization**: KV-Cache, Grouped-Query Attention (GQA), and PagedAttention
- **Foundation for**: Training, deploying, and serving high-performance models

---

## 1. The Pre-training Paradigms

### 1.1 Encoder-only (Understanding) - BERT
BERT learns by looking at the entire sentence at once.
- **MLM (Masked LM)**: Predicting hidden tokens ($15\%$).
- **NSP (Next Sentence Prediction)**: Learning discourse-level coherence.
- **Best for**: NER, Sentiment Analysis, and Semantic Search.

### 1.2 Decoder-only (Generative) - GPT
GPT models are autoregressive ($P(x_t | x_{<t})$). 
- **Scaling Laws**: Performance scales predictably with Compute, Data, and Parameters (Kaplan et al., Chinchilla).
- **Emergent Abilities**: Scale leads to unexpected skills like logical reasoning and few-shot learning.

---

## 2. The Modern "Llama-style" Architecture

Modern SOTA models (Llama-2/3, Mistral) have refined the original Transformer for better stability and efficiency.

### 2.1 Rotary Positional Embeddings (RoPE)
Instead of adding a fixed vector (Sine/Cosine), RoPE applies a **rotation** to the Query and Key vectors.
- **Why?**: It naturally encodes the **relative distance** between tokens. As the distance increases, the dot product between rotated vectors decays, mimicking how human attention works.

### 2.2 SwiGLU Activation
Replaces standard ReLU/GELU in the FFN. 
- **Math**: $\text{SwiGLU}(x, W, V, b, c) = \text{Swish}_{1}(xW + b) \otimes (xV + c)$
- **Benefit**: More stable training and better performance for the same parameter count.

### 2.3 Grouped-Query Attention (GQA)
A middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
- Multiple query heads share a single pair of Key/Value heads.
- **Benefit**: Significantly reduces the memory footprint of the **KV-Cache** during inference.

---

## 3. Inference Optimization: The Engine Room

Serving LLMs is expensive. We use specialized techniques to speed up text generation.

### 3.1 The KV-Cache
During autoregressive generation, we don't need to recompute the Keys and Values for previous tokens at every step. We store them in a "cache."
- **Problem**: KV-Cache grows linearly with sequence length, consuming massive VRAM.

### 3.2 PagedAttention (vLLM)
Inspired by virtual memory in OS. It partitions the KV-Cache into non-contiguous memory blocks (pages).
- **Benefit**: Near-zero memory waste and allows for high-throughput serving of many requests simultaneously.

---

## 💻 Professional Implementation

### 1. Rotary Embedding Logic (Simplified)
```python
import torch

def apply_rotary_emb(x, cos, sin):
    # x shape: [batch, heads, seq_len, head_dim]
    # Split head_dim into pairs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # Rotation matrix application: [x1, x2] * [cos, -sin; sin, cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    return torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
```

### 2. Inference with KV-Cache (Conceptual PyTorch)
```python
def generate_step(token_id, past_key_values=None):
    # 1. Feed only the new token
    output = model(token_id, use_cache=True, past_key_values=past_key_values)
    
    # 2. Get the new token and updated cache
    logits = output.logits
    new_cache = output.past_key_values
    
    # 3. Sample next token
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    
    return next_token, new_cache
```

---

## 📊 Summary Comparison

| Feature | Original Transformer | Llama-3 / Modern |
| :--- | :--- | :--- |
| **Normalization** | LayerNorm (Post-Attn) | RMSNorm (Pre-Attn) |
| **Embeddings** | Absolute (Sine/Cos) | **RoPE** (Rotary) |
| **Activation** | ReLU | **SwiGLU** |
| **Attention** | MHA | **GQA** |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Model Quantization**| Running a 70B model on 24GB VRAM using 4-bit (AWQ/GPTQ). |
| **Speculative Decoding**| Using a tiny "draft" model to speed up a large "target" model. |
| **Context Compression**| Summarizing KV-caches to handle million-token prompts. |
| **MoE (Mixture of Experts)**| Scaling models to trillions of parameters while keeping inference cost low (Mixtral). |

---

## ❓ Quick Check Questions

1. Why is **RMSNorm** preferred over standard LayerNorm in modern LLMs?
2. Explain the "quadratic bottleneck" of the KV-Cache.
3. How does **SwiGLU** differ from a standard gated linear unit?
4. What is the difference between "Kaplan" and "Chinchilla" scaling laws?
5. Why does **Grouped-Query Attention** specifically help with inference throughput?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **RMSNorm** (Root Mean Square Layer Normalization) is computationally simpler as it doesn't calculate the mean (only the variance). This provides similar stability benefits to LayerNorm but with lower overhead.
2. The KV-Cache stores vectors for every previous token. As the sequence grows, the memory required to store these vectors increases linearly. However, because attention is $O(n^2)$, the compute cost to process this cache increases quadratically.
3. **SwiGLU** uses the Swish activation function ($\sigma(x) \cdot x$) inside the gate. It provides a smoother non-linearity and better gradient flow than ReLU-based gates.
4. **Kaplan laws** suggested that increasing model size is the most important factor. **Chinchilla laws** (Hoffmann et al.) proved that for every doubling of model size, the training tokens must also double. Most models were "undertrained" before this realization.
5. **GQA** reduces the number of Keys and Values stored in the KV-Cache (by sharing them across query heads). Smaller caches mean more requests can fit in GPU memory at once, increasing the "batch size" during serving.

</details>

---

## 📚 Recommended Resources
- **Paper**: [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- **Paper**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Deep Dive**: [Mastering KV-Cache (FlashAttention Team)](https://triton-lang.org/main/index.html).

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** Prompt Engineering (ToT, Self-Consistency, DSPy)
