# 10.7 Advanced Fine-tuning: PEFT, Alignment, and Quantization

## 🎯 Quick Overview
- **PEFT (Parameter-Efficient Fine-tuning)**: Why training <1% of parameters works
- **LoRA & QLoRA**: Mathematical derivation of low-rank updates and 4-bit quantization
- **Alignment Math**: DPO (Direct Preference Optimization) vs. RLHF (PPO) objective functions
- **Advanced Quantization**: GGUF, EXL2, and the bits-per-weight frontier
- **Foundation for**: Specializing LLMs for niche domains while keeping hardware costs low

---

## 1. LoRA: Low-Rank Adaptation Math

LoRA assumes that during task adaptation, the change in weights $\Delta W$ has a low "intrinsic rank."

### 1.1 The Derivation
For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, we freeze $W_0$ and learn $\Delta W = BA$, where:
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ (the Rank)

#### Forward Pass:
$$ h = W_0x + \Delta Wx = W_0x + BAx $$

**Why it's brilliant**: At inference time, we can **merge** the weights ($W_{new} = W_0 + BA$) so there is **zero latency overhead** compared to the base model.

---

## 2. Quantization Theory: Precision vs. Performance

Quantization reduces the precision of weights (e.g., from FP32 to 4-bit) to save VRAM.

### 2.1 NF4 (NormalFloat 4-bit)
Introduced with **QLoRA**, NF4 is an information-theoretically optimal data type for normally distributed weights.
1.  **Quantization**: Maps weights to 16 discrete levels.
2.  **Double Quantization**: Quantizes the quantization constants themselves to save additional memory.
3.  **Paged Optimizers**: Prevents OOM (Out-of-Memory) errors by managing optimizer states in CPU RAM when needed.

---

## 3. The Alignment Frontier: DPO vs. RLHF

Making an LLM follow human values (Helpful, Honest, Harmless).

### 3.1 RLHF (PPO)
Uses a **Reward Model** ($R_\phi$) to score outputs.
- **Objective**: Maximize $\mathbb{E}_{x, y \sim \pi_\theta} [R_\phi(x, y)] - \beta \text{KL}(\pi_\theta \| \pi_{ref})$
- **Complexity**: Requires 4 models in memory (Base, Reference, Reward, Value).

### 3.2 DPO (Direct Preference Optimization)
Directly optimizes the LLM on preference pairs $(x, y_w, y_l)$ where $y_w$ is preferred over $y_l$.
- **The DPO Loss**:
  $$ \mathcal{L}_{DPO}(\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right] $$
- **Why it's winning**: Mathematically equivalent to RLHF but significantly more stable and $2\times$ faster to train.

---

## 💻 Professional Implementation

### 1. Merging LoRA Weights (Conceptual)
```python
import torch

def merge_lora(base_weight, lora_A, lora_B, alpha, r):
    # scaling = lora_alpha / r
    scaling = alpha / r
    
    # Calculate Delta W
    delta_w = torch.matmul(lora_B, lora_A) * scaling
    
    # Merge
    merged_weight = base_weight + delta_w
    return merged_weight
```

### 2. Loading a Model in 4-bit (bitsandbytes)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "llama-3-8b", 
    quantization_config=quant_config
)
```

---

## 📊 Summary Comparison

| Metric | Full Fine-tuning | LoRA | QLoRA | DPO |
| :--- | :--- | :--- | :--- | :--- |
| **Trainable Params** | 100% | < 1% | < 1% | 100% (or with LoRA)|
| **VRAM Requirement** | Massive | Moderate | **Low** | Moderate |
| **Inference Overhead**| Zero | Zero (if merged)| Zero (Quantized) | Zero |
| **Task Performance** | **Highest** | High | High | Alignment only |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Domain-SFT** | Training a Llama model on legal case law to understand specific terminology. |
| **Adapters-Switching**| Swapping small LoRA files at runtime to handle different users (e.g., "Coding Expert" vs. "Creative Writer"). |
| **KTO (Kahneman-Tversky)**| A simpler alignment method that only requires "thumbs up/down" data instead of pairs. |
| **Unsloth** | A library that uses custom OpenAI Triton kernels to make LoRA training $2\times$ faster. |

---

## ❓ Quick Check Questions

1. In LoRA, why is the weight update $\Delta W = BA$ more efficient than $W$?
2. What is the difference between "Post-Training Quantization" (PTQ) and "Quantization-Aware Training" (QAT)?
3. Why does DPO eliminate the need for a separate Reward Model?
4. What is "Catastrophic Forgetting," and how does PEFT mitigate it?
5. Explain "Double Quantization" in the context of QLoRA.

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. If $W$ is $4096 \times 4096$ (16.7M params), and we use rank $r=8$, then $A$ is $8 \times 4096$ and $B$ is $4096 \times 8$. Total params = $2 \times (8 \times 4096) = 65,536$. This is a **$250\times$ reduction** in trainable parameters.
2. **PTQ** quantizes a model after it has been trained (fast, slight accuracy drop). **QAT** simulates quantization during training so the model learns to be robust to the precision loss (slow, higher accuracy).
3. The DPO derivation shows that human preferences can be expressed as a function of the **optimal policy** itself. By rearranging the RLHF math, we can optimize the model directly using the log-likelihood of the preferred answer relative to the reference model.
4. **Catastrophic Forgetting** is when a model loses its general knowledge while learning a specific task. Because PEFT (like LoRA) freezes the original weights, the base knowledge remains untouched, and the model only learns "add-on" behaviors.
5. QLoRA quantizes the weights to 4-bit. It then identifies the "Quantization Constants" (used to scale the 4-bit values back to floats) and quantizes **those constants** to 8-bit, saving an additional ~0.3 bits per parameter.

</details>

---

## 📚 Recommended Resources
- **Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- **Library**: [Unsloth AI](https://github.com/unslothai/unsloth) - *The fastest way to fine-tune LLMs*.

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** CNN Architectures (Receptive Fields, Depthwise Conv, EfficientNet)
