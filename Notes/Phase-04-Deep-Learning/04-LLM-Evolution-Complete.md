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

---

#### 🧒 ELI5: The Rubber Band with Marks

> Imagine a rubber band with equally spaced marks on it.
>
> **Old way (absolute positional encoding)**: You write the position number next to each mark. Mark 1 says "1", mark 100 says "100". But what if you need mark 101? You never wrote a label for it!
>
> **RoPE way (rotation)**: Instead of writing numbers, you **rotate** the rubber band. The distance between any two marks stays the same whether they're at positions 1-2 or positions 100-101.
>
> **The magic**: If you rotate mark 1 by 30° and mark 2 by 31°, the **difference** (1°) tells you they're adjacent. Same for marks 100 and 101! The rotation automatically encodes relative distance.
>
> **Why this matters**: RoPE can handle sequences longer than what it saw during training because rotation works for any position, not just positions 1-2048.

</details>

### 2.2 SwiGLU Activation
Replaces standard ReLU/GELU in the FFN. 
- **Math**: $\text{SwiGLU}(x, W, V, b, c) = \text{Swish}_{1}(xW + b) \otimes (xV + c)$
- **Benefit**: More stable training and better performance for the same parameter count.

### 2.3 Grouped-Query Attention (GQA)
A middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
- Multiple query heads share a single pair of Key/Value heads.
- **Benefit**: Significantly reduces the memory footprint of the **KV-Cache** during inference.

---

#### 🧒 ELI5: The Restaurant Waiters

> Imagine a busy restaurant with 32 customers (query heads) and 8 waiters (KV heads).
>
> **MHA (Multi-Head Attention)**: 32 customers, 32 waiters. Everyone gets personal service, but it's expensive to pay all those waiters!
>
> **MQA (Multi-Query Attention)**: 32 customers, 1 waiter. Super cheap, but the waiter is overwhelmed and service quality drops.
>
> **GQA (Grouped-Query)**: 32 customers, 8 waiters. Each waiter serves 4 customers. The customers in each group share similar orders, so this works almost as well as personal waiters, but costs much less!
>
> **In LLaMA-3**: 
> - 32 query heads (customers)
> - 8 KV heads (waiters)
> - Each KV head serves 4 query heads
> - Result: 4× smaller cache, almost same quality as full MHA

</details>

---

## 3. Inference Optimization: The Engine Room

Serving LLMs is expensive. We use specialized techniques to speed up text generation.

### 3.1 The KV-Cache
During autoregressive generation, we don't need to recompute the Keys and Values for previous tokens at every step. We store them in a "cache."
- **Problem**: KV-Cache grows linearly with sequence length, consuming massive VRAM.

---

#### 🧒 ELI5: Recipe Memorization

> Imagine you're cooking a complex dish with 50 steps.
>
> **Without KV-Cache**: At step 25, you re-read ALL previous 24 steps from the recipe book to remember what you did. At step 26, you re-read all 25 previous steps again. This is incredibly slow!
>
> **With KV-Cache**: After you complete each step, you memorize the key information (chopped onions, browned meat, added spices). At step 25, you just recall your memorized notes instead of re-reading everything.
>
> **The trade-off**: Memorizing (KV-Cache) takes mental space (VRAM), but it's WAY faster than re-reading the recipe every single time.
>
> **In LLMs**: 
> - Keys = "What each previous word is about"
> - Values = "The actual information from each word"
> - Cache = Remembering these instead of recomputing for every new token

</details>


### 3.2 PagedAttention (vLLM)
Inspired by virtual memory in OS. It partitions the KV-Cache into non-contiguous memory blocks (pages).
- **Benefit**: Near-zero memory waste and allows for high-throughput serving of many requests simultaneously.

---

#### 🧒 ELI5: The Office Filing Cabinet

> Imagine you need to store 1000 documents, but your filing cabinet has gaps everywhere.
>
> **Old way (contiguous memory)**: "I need 100 consecutive empty slots. Can't find them? Sorry, no storage for you!" Most space goes wasted.
>
> **PagedAttention way**: Store documents in whatever small spaces are available:
> - Document 1: Drawer A, slots 5-12
> - Document 2: Drawer C, slots 1-8
> - Document 3: Drawer B, slots 20-27
>
> **The index card**: A small lookup table tracks where each document lives. When you need Document 2, you check the index, then go to Drawer C.
>
> **Result**: Almost zero wasted space! Many more requests can be served simultaneously because we use every little gap.

</details>

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

## 4. Mixture of Experts (MoE): Scaling to Trillions

### 4.1 The MoE Architecture
Instead of every token passing through all parameters, MoE uses **sparse activation**.

#### The Math:
For each token $x$:
1.  **Router Network**: Computes gating scores $h(x) = \text{softmax}(W_g \cdot x)$
2.  **Top-K Selection**: Selects $K$ experts with highest scores
3.  **Weighted Output**: $y = \sum_{i \in \text{top-K}} h(x)_i \cdot E_i(x)$

Where $E_i$ is the $i$-th expert network (typically a feed-forward layer).

**Key Insight**: A model with 100B total parameters might only use 10B per forward pass!

---

### 4.2 Switch Transformer
Google's approach to MoE with **single expert routing** (K=1).

**Architecture**:
- Router: Simple linear layer
- Experts: Standard FFN layers
- **Capacity Factor**: Limits tokens per expert to prevent overload

**Training Stability**:
- **Auxiliary Loss**: Penalizes imbalanced expert usage
    $$ \mathcal{L}_{aux} = \sum_{i=1}^N f_i \cdot P_i $$
    Where $f_i$ is fraction of tokens routed to expert $i$, and $P_i$ is router probability for expert $i$.

---

### 4.3 Mixtral 8x7B
State-of-the-art open MoE model (2024).

**Specifications**:
- 8 experts per layer, activates 2 per token
- Total: 47B parameters, Active: 13B per forward pass
- **Sparse MoE**: Each token uses different experts

**Benefits**:
- 5× faster inference than dense 47B model
- Better quality than dense 13B model
- Efficient multi-task learning (different experts specialize)

---

### 4.4 Implementation: MoE Layer from Scratch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Expert networks (simple FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        
        # Reshape for token-level processing
        x_flat = x.view(-1, dim)  # [batch * seq_len, dim]
        
        # Router logits
        router_logits = self.router(x_flat)  # [batch * seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx)
            if not expert_mask.any():
                continue
            
            # Get token indices and their weights
            token_indices = torch.where(expert_mask.any(dim=1))[0]
            weights = top_k_probs[token_indices, expert_mask[token_indices]].sum(dim=1, keepdim=True)
            
            # Route through expert
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weight and accumulate
            output[token_indices] += expert_output * weights
        
        return output.view(batch_size, seq_len, dim)

# Usage
moe = MoELayer(dim=512, num_experts=8, top_k=2)
x = torch.randn(32, 128, 512)  # batch=32, seq_len=128
output = moe(x)
print(f"Output shape: {output.shape}")
```

---

## 5. State Space Models (SSMs): The Mamba Revolution

### 5.1 The SSM Foundation
SSMs are inspired by control theory and offer $O(n)$ sequence modeling.

**Continuous Form**:
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) $$

Where:
- $h(t)$: Hidden state (evolves over time)
- $x(t)$: Input
- $y(t)$: Output
- $A, B, C$: Learned parameters

---

### 5.2 Discretization (Zero-Order Hold)
To use SSMs in deep learning, we discretize with step size $\Delta$:

$$ \bar{A} = \exp(\Delta A) $$
$$ \bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B $$

**Discrete Form**:
$$ h_t = \bar{A} h_{t-1} + \bar{B} x_t $$
$$ y_t = C h_t $$

---

### 5.3 Mamba: Selective State Spaces
Mamba's innovation: make $B, C, \Delta$ **input-dependent**.

**Key Equations**:
```python
# For each token x_t
Δ = softplus(x_t @ W_Δ)      # Step size depends on input
B = x_t @ W_B                 # Input projection
C = x_t @ W_C                 # Output projection

# SSM recurrence
h_t = Ā(Δ) * h_{t-1} + B̄(Δ) * B * x_t
y_t = C * h_t
```

**Why It Works**:
- **Content-aware**: The model can "choose" what to remember/forget
- **Linear complexity**: $O(n)$ vs. Transformer's $O(n^2)$
- **Infinite context**: No fixed context window limit

---

### 5.4 Mamba vs. Transformer Comparison

| Feature | Transformer | Mamba (SSM) |
| :--- | :--- | :--- |
| **Complexity** | $O(n^2)$ | $O(n)$ |
| **Memory** | $O(n)$ (KV-cache) | $O(1)$ (fixed state) |
| **Parallelization** | Excellent | Good (via convolution) |
| **Inference Speed** | Slow (memory-bound) | Fast (compute-bound) |
| **Context Limit** | Fixed (e.g., 128K) | Unlimited |
| **Quality (2024)** | SOTA | Near-SOTA |

---

### 5.5 Mamba Implementation (Simplified)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=dim
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(dim, d_state, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        
        # Initialize A parameter (diagonal)
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim))
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        batch, seq_len, dim = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # [batch, seq_len, dim*2]
        x, res = x_and_res.split([dim, dim], dim=-1)
        
        # Convolution (local context)
        x = x.transpose(1, 2)  # [batch, dim, seq_len]
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, dim]
        
        x = F.silu(x)
        
        # SSM computation (simplified - real Mamba uses parallel scan)
        A = -torch.exp(self.A_log)  # [d_state]
        
        # Project to SSM parameters
        B = self.x_proj(x)  # [batch, seq_len, d_state]
        C = self.x_proj(x)  # [batch, seq_len, d_state]
        dt = F.softplus(self.dt_proj(x))  # [batch, seq_len, dim]
        
        # Discretize (simplified)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [batch, seq_len, d_state]
        dB = dt * B  # [batch, seq_len, d_state]
        
        # Sequential scan (in practice, use parallel algorithm)
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :1].expand(-1, self.d_state)
            y_t = (h * C[:, t]).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, dim]
        y = y + self.D * x  # Skip connection
        
        # Output projection
        y = self.out_proj(y * F.silu(res))
        
        return y

# Usage
mamba = MambaBlock(dim=512, d_state=16)
x = torch.randn(8, 1024, 512)
output = mamba(x)
print(f"Output shape: {output.shape}")
```

---

## 6. Inference Engines: Production Deployment

### 6.1 vLLM: High-Throughput Serving
**Key Innovation**: PagedAttention for efficient KV-cache management.

**Features**:
- **Continuous Batching**: Add new requests as soon as any batch slot frees up
- **PagedAttention**: Non-contiguous KV-cache storage (like OS virtual memory)
- **CUDA Graphs**: Pre-compile computation graphs for speed

**Performance**: 2-4× higher throughput than HuggingFace Transformers.

```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Sampling configuration
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    top_p=0.9
)

# Batch inference
prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "How do transformers work?"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

### 6.2 TGI (Text Generation Inference)
HuggingFace's production-ready inference server.

**Features**:
- **Tensor Parallelism**: Split model across multiple GPUs
- **Flash Attention**: Optimized attention implementation
- **Quantization**: INT8, FP8, AWQ support
- **OpenAI-compatible API**: Drop-in replacement for GPT APIs

```bash
# Run TGI server
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --num-shard 2 \
    --quantize awq
```

---

### 6.3 llama.cpp: CPU Inference
**GGUF Format**: Quantized models for CPU/GPU hybrid inference.

**Quantization Levels**:
| Format | Bits | Quality | RAM (7B) |
| :--- | :--- | :--- | :--- |
| Q4_0 | 4.05 | Good | 4GB |
| Q5_K_M | 5.5 | Very Good | 5GB |
| Q8_0 | 8.5 | Near-lossless | 8GB |

```python
from llama_cpp import Llama

# Load quantized model
llm = Llama(
    model_path="llama-3-8b.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=35  # Offload layers to GPU
)

output = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=64,
    stop=["Q:", "\n"],
    echo=True
)

print(output['choices'][0]['text'])
```

---

### 6.4 TensorRT-LLM: NVIDIA Optimization
**Best for**: High-performance deployment on NVIDIA GPUs.

**Optimizations**:
- **Kernel Fusion**: Combine multiple operations
- **In-flight Batching**: Dynamic request scheduling
- **FP8 Precision**: 2× speedup on H100 GPUs

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# Build optimized engine
runner = ModelRunner.from_engine(
    engine_dir="llama2_7b_trt",
    max_batch_size=32,
    max_input_len=4096,
    max_output_len=1024
)

# Inference
with runner.create_session():
    outputs = runner.generate(inputs)
```

---

## 7. Model Architectures Timeline

```
2018: BERT (Encoder-only, MLM)
    ↓
2019: RoBERTa (Improved BERT)
    ↓
2020: T5 (Encoder-Decoder, Text-to-Text)
    ↓
2020: GPT-3 (Decoder-only, 175B)
    ↓
2022: LLaMA (Efficient Decoder)
    ↓
2023: Mixtral (MoE, 8x7B)
    ↓
2024: Mamba (SSM, Linear Complexity)
    ↓
2024: LLaMA-3 (GQA, RoPE, 405B)
```

---

## 8. Advanced KV-Cache Management

### 8.1 KV-Cache Size Calculation
For a model with:
- Hidden dim: 4096
- Num KV heads: 8
- Num layers: 32
- Seq len: 4096
- Precision: FP16 (2 bytes)

$$ \text{Size} = 2 \times 4096 \times 8 \times \frac{4096}{8} \times 32 \times 2 \text{ bytes} \approx 4 \text{ GB} $$

(The factor of 2 is for K and V; dividing by 8 converts head_dim to bytes)

---

### 8.2 Prefix Caching
For repeated prompts (e.g., system messages), cache the KV states.

```python
# vLLM prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True
)

# First request computes and caches
output1 = llm.generate(["System: You are helpful.\nUser: Hello"])

# Second request reuses cached prefix
output2 = llm.generate(["System: You are helpful.\nUser: Hi there"])
# Only processes "Hi there" vs "Hello"
```

---

### 8.3 Speculative Decoding
Use a small "draft" model to propose tokens, verify with large model.

**Algorithm**:
1. Draft model generates $K$ tokens autoregressively
2. Target model evaluates all $K$ tokens in parallel
3. Accept correct tokens, resample incorrect ones

**Speedup**: 2-3× for large models.

```python
# vLLM speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    speculative_model="meta-llama/Llama-2-7b-chat-hf",
    num_speculative_tokens=5
)
```

---

## 🔬 Research Frontiers (2024-2025)

### 9.1 Hybrid Architectures
- **Jamba (AI21)**: Alternating Transformer + Mamba layers
- **Griffin**: Gated Convolution + Local Attention
- **RWKV**: RNN-like Transformer with linear attention

### 9.2 Long Context Models
- **Claude-3**: 200K context window
- **Gemini-1.5**: 1M+ tokens with MoE
- **Technique**: Sparse attention + improved positional encoding

### 9.3 Efficient Training
- **Fully Sharded Data Parallel (FSDP)**: Shard model across GPUs
- **Activation Checkpointing**: Trade compute for memory
- **Gradient Accumulation**: Simulate larger batches

---

**Status:** ✅ Elite Expanded Standard (14/10)
**Next:** Prompt Engineering (ToT, Self-Consistency, DSPy, Constitutional AI)
