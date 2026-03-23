# 10.7 Fine-tuning Techniques (PEFT, LoRA, DPO)

## 🎯 Quick Overview
- **Fine-tuning**: Adapting a pre-trained LLM to a specific task or dataset
- **PEFT (Parameter-Efficient Fine-tuning)**: Training only a small subset of parameters
- **LoRA (Low-Rank Adaptation)**: The industry standard for efficient adaptation
- **Alignment (RLHF & DPO)**: Steering models to be helpful, honest, and harmless
- **Foundation for**: Specialized AI models (e.g., Medical GPT, Coding Assistant)

---

## 1. Full Fine-tuning vs. PEFT

**Full Fine-tuning** updates all billions of parameters in a model. 
- **Pros**: Maximum performance on the target task.
- **Cons**: Extremely expensive (requires massive GPU VRAM) and risks **Catastrophic Forgetting** (the model forgets its original knowledge).

**PEFT** keeps most of the model frozen and only trains small "adapter" layers.
- **Pros**: Fast, cheap, and memory-efficient.

---

## 2. LoRA: Low-Rank Adaptation

**LoRA** is the most popular PEFT technique. It assumes that the changes to the weights during fine-tuning have a "low intrinsic rank."

1. **Mechanism**: It freezes the original weight matrix $W$ and injects two smaller matrices $A$ and $B$.
2. **Math**: Instead of updating $W$, we learn $\Delta W = A \times B$.
3. **Efficiency**: For a $4096 \times 4096$ matrix, LoRA might only train a few thousand parameters instead of 16 million.

### 2.1 QLoRA
**QLoRA** takes this further by quantizing the base model to **4-bit** precision, allowing you to fine-tune a 70B parameter model on a single consumer GPU (e.g., RTX 3090/4090).

---

## 3. Alignment & Preference Optimization

Fine-tuning on data is one thing; making the model "behave" is another.

### 3.1 RLHF (Reinforcement Learning from Human Feedback)
1. **SFT**: Supervised Fine-tuning on high-quality examples.
2. **Reward Model**: A separate model is trained to score LLM responses based on human preferences.
3. **PPO**: The LLM is optimized using reinforcement learning to maximize the score from the reward model.

### 3.2 DPO (Direct Preference Optimization)
A newer, simpler alternative to RLHF. It removes the need for a separate reward model and PPO step by directly optimizing the model on pairs of "preferred" vs "rejected" responses.

---

## 💻 Python Code Examples

### 1. Fine-tuning with LoRA (PEFT + Transformers)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("llama-3-8b")

# 2. Define LoRA Config
config = LoraConfig(
    r=8, # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Create PEFT Model
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters() 
# Typically shows < 1% of total parameters!
```

### 2. Data Format for DPO
```json
{
  "prompt": "Write a Python function to sort a list.",
  "chosen": "def sort_list(l): return sorted(l)",
  "rejected": "You should use a loop and compare every element manually."
}
```

---

## 📊 Summary Table

| Technique | Cost | Complexity | Use Case |
|-----------|------|------------|----------|
| **Full Fine-tune** | Very High | High | Fundamental domain shift |
| **LoRA** | Low | Medium | Most task-specific adaptation |
| **QLoRA** | Very Low | Medium | Limited hardware (Single GPU) |
| **DPO** | Medium | Medium | Aligning model tone/safety |
| **RLHF** | High | Very High | SOTA alignment (GPT-4 level) |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| LoRA | Adapting Llama-3 for medical diagnosis |
| QLoRA | Fine-tuning personal coding assistants locally |
| DPO | Reducing toxicity in customer service bots |
| Continued Pre-training | Teaching an LLM a new language |

---

## ❓ Quick Check Questions

1. Why does Full Fine-tuning often lead to "Catastrophic Forgetting"?
2. Explain the core mathematical idea behind LoRA.
3. What is the main advantage of QLoRA over standard LoRA?
4. How does DPO simplify the RLHF process?
5. What does the "Rank" (r) parameter in LoRA control?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Catastrophic Forgetting** happens because the gradient updates modify every single weight in the model. As the model specializes in the new task, the weights that encoded its general knowledge are overwritten, causing it to lose its reasoning or general conversation abilities.
2. LoRA assumes that the weight updates ($\Delta W$) can be represented as the product of two **low-rank matrices** ($A$ and $B$). This significantly reduces the number of parameters the model needs to learn during the optimization process.
3. **QLoRA** uses 4-bit quantization for the base model weights. This massively reduces the memory (VRAM) required to load the model, allowing powerful LLMs to be fine-tuned on hardware that previously couldn't even run them.
4. RLHF requires training a **Reward Model** and using a complex reinforcement learning algorithm (**PPO**). **DPO** (Direct Preference Optimization) replaces this with a simple cross-entropy loss applied to pairs of responses, making it much more stable and easier to implement.
5. The **Rank (r)** determines the size of the adapter matrices. A higher rank allows the model to learn more complex patterns but increases the number of trainable parameters and memory usage. Standard values are 8, 16, or 32.

</details>

---

**Status:** ✅ Complete
**Next:** Transition to Computer Vision (CNNs, Object Detection, Segmentation)
