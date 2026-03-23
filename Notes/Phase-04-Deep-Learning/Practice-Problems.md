# Phase 4: Specialization - Practice Problems (Elite Standard)

## 📊 Graded Practice Levels

### Level 1: Core Concept Mastery
**1.1** **Subword Tokenization**: Trace the first 3 merges of Byte-Pair Encoding (BPE) for the following corpus: `{"hug": 5, "pug": 2, "pun": 6}`.
**1.2** **Self-Attention**: In the formula $\text{softmax}(QK^T / \sqrt{d_k})V$, why is the transpose of $K$ used, and what does the resulting matrix represent?
**1.3** **CNN Geometry**: If an input is $224 \times 224 \times 3$, what is the exact number of parameters in a $3 \times 3$ convolution layer with 64 filters and a bias term?
**1.4** **ASR Fundamentals**: Define the "Mel-Scale" and explain why it is essential for human-centric speech models.
**1.5** **Generative Models**: What is the "Reparameterization Trick" in VAEs, and why is it mathematically required for training?

### Level 2: Architectural & Mathematical Logic
**2.1** **Transformer Complexity**: Prove that the memory complexity of standard self-attention is $O(n^2)$ where $n$ is sequence length. How does **Grouped-Query Attention (GQA)** improve this?
**2.2** **LSTM Gates**: Write the mathematical update for the **Forget Gate** and explain how its output (0 to 1) interacts with the previous Cell State $C_{t-1}$.
**2.3** **Object Detection**: Define **mAP@50:95**. Why is this a more rigorous metric for self-driving cars than simple mAP@50?
**2.4** **LoRA Weight Merging**: If a base weight $W_0$ is $4096 \times 4096$ and we use LoRA rank $r=16$, calculate the total number of trainable parameters. How do we merge them for zero-latency inference?
**2.5** **Diffusion SDE**: Explain the role of the "U-Net" in the reverse diffusion process. What exactly is it trying to predict at each time step $t$?

### Level 3: Advanced Pipeline Analysis & Optimization
**3.1** **RAG Search**: You are using a Vector DB with 10 million vectors. Compare the search latency of **Flat Indexing** vs. **HNSW Indexing**. Explain the "Small World" property.
**3.2** **Fine-tuning Alignment**: Compare **DPO** and **PPO**. Why does DPO not require a separate Reward Model, and what are the stability implications?
**3.3** **Vision Transformers**: Explain how **Swin Transformer** uses "Shifted Windows" to allow for cross-window communication while maintaining linear complexity.
**3.4** **Quantization**: Explain the difference between **symmetric** and **asymmetric** quantization. Why is NF4 (NormalFloat 4) information-theoretically optimal for LLMs?

### Level 4: Python Implementation Practice
**4.1** **PyTorch Custom Attention**: Implement a simplified `MultiHeadAttention` class that splits the embedding dimension into $H$ heads and computes scaled dot-product attention.
**4.2** **NLP Preprocessing**: Write a Python function to calculate the **TF-IDF** score for a word in a specific document, given a small corpus of 5 documents.
**4.3** **CV Augmentation**: Create a custom PyTorch `Dataset` wrapper that applies **Mosaic Augmentation** (combining 4 images) for an object detection task.
**4.4** **Audio Feature Extraction**: Use `librosa` to load an audio file, compute the Log-Mel Spectrogram, and normalize it to mean 0, variance 1.

### Level 5: Professional System Design
**5.1** **Scenario: Real-time Multimodal Security System**
- **The Task**: You must build a system that monitors a 24/7 video feed.
- **Requirements**: 
    1. It must detect people and specifically identify if they are carrying "restricted items" (Detection).
    2. It must be able to answer natural language questions about the footage (e.g., "What was the man in the blue shirt doing at 2 PM?") (Multimodal LLM).
    3. It must run on an **edge device** (limited VRAM).
**Task**: Propose the full stack. Which **quantization** method (AWQ/GGUF) and **Efficient CNN/ViT** backbone would you use? How would you handle the **Temporal** aspect (video vs. image)? Describe the **RAG** strategy for indexing historical video frames.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1 (BPE)
Initial chars: `h, u, g, p, u, n, b`.
Pairs: `ug: 7 (5+2)`, `un: 6`, `hu: 5`.
Merge 1: `ug`. Vocab: `h ug: 5, p ug: 2, p u n: 6`.
Merge 2: `un`. Vocab: `h ug: 5, p ug: 2, p un: 6`.
Merge 3: `p un`. (Since it's more frequent than `p ug`).

### 2.4 (LoRA Params)
Base $W$: $4096^2 \approx 16.7M$ (Frozen).
LoRA $A$: $16 \times 4096 = 65,536$.
LoRA $B$: $4096 \times 16 = 65,536$.
Total trainable: $131,072$ (~0.7% of the layer).
**Merge**: $W_{merged} = W_0 + (B \times A) \times \frac{\alpha}{r}$.

### 4.1 (Attention Logic)
```python
import torch.nn.functional as F
def attention(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

### 5.1 (System Design)
- **Backbone**: **MobileNetV3** or **Swin-Tiny** (quantized to INT4 via **AWQ**).
- **Temporal**: Use a **SlowFast** network or a sliding window of frames fed into a **Video-LLaVA** adapter.
- **RAG**: Extract CLIP embeddings every 1 second of video. Store in a **Qdrant** DB with HNSW.
- **Inference**: Use **vLLM** with PagedAttention for high-throughput frame analysis.

</details>

---

## 📚 Global Phase 4 Resources
- **Course**: [DeepLearning.AI: Natural Language Processing Specialization](https://www.deeplearning.ai/courses/natural-language-processing-specialization/)
- **Course**: [Fast.ai: Practical Deep Learning for Coders (Part 2)](https://course.fast.ai/)
- **Forum**: [HuggingFace Forums](https://discuss.huggingface.co/) - *Best for debugging modern SOTA models*.

---
**Last Updated:** 2026-03-23
**Status:** ✅ Phase 4 Specialization Complete (10/10)
