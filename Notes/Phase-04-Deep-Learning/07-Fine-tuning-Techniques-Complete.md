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

## 4. Advanced PEFT Methods

### 4.1 AdaLoRA (Adaptive LoRA)
Standard LoRA uses fixed rank for all layers. **AdaLoRA** allocates rank adaptively based on importance.

**The Math**:
Instead of fixed $\Delta W = BA$, AdaLoRA uses SVD parameterization:
$$ \Delta W = P \Lambda Q $$
Where:
- $P, Q$: Orthogonal matrices (learned)
- $\Lambda$: Diagonal matrix of singular values (learned, can be pruned)

**Key Innovation**: Importance scores guide rank allocation:
$$ I(\Delta W) = \|\Delta W\|_F^2 = \sum_i \lambda_i^2 $$

Layers with higher importance get more rank budget.

```python
class AdaLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, initial_rank=8, target_rank=4):
        super().__init__()
        self.rank = initial_rank
        self.target_rank = target_rank
        
        # SVD parameterization
        self.P = nn.Parameter(torch.randn(out_dim, initial_rank))
        self.Q = nn.Parameter(torch.randn(initial_rank, in_dim))
        self Lambda = nn.Parameter(torch.ones(initial_rank))  # Singular values
        
        # Importance scores for pruning
        self.importance = torch.zeros(initial_rank)
    
    def forward(self, x):
        # Apply low-rank update
        delta = self.P @ torch.diag(self.Lambda) @ self.Q
        return F.linear(x, delta)
    
    def prune(self, budget_ratio=0.5):
        """Prune less important singular values."""
        # Calculate importance based on gradient magnitude
        with torch.no_grad():
            importance = torch.abs(self.Lambda * self.P.norm(dim=0) * self.Q.norm(dim=1))
            
            # Keep top-k singular values
            k = int(self.rank * budget_ratio)
            top_k_indices = torch.topk(importance, k).indices
            
            # Mask out pruned values
            mask = torch.zeros_like(self.Lambda)
            mask[top_k_indices] = 1
            self.Lambda = self.Lambda * mask
```

---

### 4.2 DoRA (Weight-Decomposed LoRA)
DoRA decomposes pre-trained weights into magnitude and direction for finer updates.

**The Decomposition**:
$$ W = m \cdot \frac{W}{\|W\|_2} $$
Where:
- $m$: Magnitude (scalar, learnable)
- $\frac{W}{\|W\|_2}$: Direction (unit vector)

**DoRA Update**:
$$ W' = (m + \Delta m) \cdot \frac{W + \Delta W}{\|W + \Delta W\|_2} $$

**Benefits**:
- More stable training than LoRA
- Better performance on vision tasks
- Comparable inference speed to LoRA

```python
class DoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        
        # Base weight (frozen)
        self.register_buffer('base_weight', torch.randn(out_dim, in_dim))
        
        # LoRA for directional update
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim))
        self.lora_B = nn.Parameter(torch.randn(out_dim, rank))
        
        # Magnitude (learnable)
        self.magnitude = nn.Parameter(torch.ones(out_dim, 1))
        
        # Scaling
        self.alpha = 1.0 / rank
    
    def forward(self, x):
        # Directional update
        delta_W = self.lora_B @ self.lora_A * self.alpha
        updated_W = self.base_weight + delta_W
        
        # Normalize direction
        norm_W = updated_W / updated_W.norm(dim=1, keepdim=True)
        
        # Apply magnitude
        final_W = self.magnitude * norm_W
        
        return F.linear(x, final_W)
```

---

### 4.3 PiSSA (Principal Singular Values Adaptation)
PiSSA initializes LoRA with the **principal singular vectors** of the pre-trained weights.

**The Algorithm**:
1.  Compute SVD of pre-trained weight: $W = U \Sigma V^T$
2.  Initialize LoRA with top-k singular vectors:
    - $A = \sqrt{\Sigma_k} V_k^T$
    - $B = U_k \sqrt{\Sigma_k}$
3.  Residual weight: $W_{res} = W - BA$

**Why It Works**:
- Captures most important directions from the start
- Faster convergence than random LoRA initialization
- Better final performance on most tasks

```python
def initialize_pissa(W, rank=8):
    """Initialize PiSSA adapters from pre-trained weight."""
    # Compute SVD
    U, S, Vt = torch.svd(W.float())
    
    # Take top-k singular vectors
    U_k = U[:, :rank]
    S_k = S[:rank]
    V_k = Vt[:rank, :]
    
    # Initialize A and B
    sqrt_S_k = torch.sqrt(torch.diag(S_k))
    A = sqrt_S_k @ V_k
    B = U_k @ sqrt_S_k
    
    # Residual weight
    W_residual = W - B @ A
    
    return A, B, W_residual
```

---

### 4.4 LoRA+ (LoRA Plus)
Different learning rates for A and B matrices in LoRA.

**Key Insight**: In $\Delta W = BA$, the matrices play asymmetric roles:
- $A$ projects input to low-rank space
- $B$ projects from low-rank to output

**LoRA+ Update**:
$$ A \leftarrow A - \eta_A \nabla_A \mathcal{L} $$
$$ B \leftarrow B - \eta_B \nabla_B \mathcal{L} $$

Where $\eta_B = \lambda \cdot \eta_A$ and $\lambda \approx 16$ (empirically).

```python
class LoRAPlusOptimizer:
    def __init__(self, lora_params, base_lr=1e-4, loraplus_ratio=16):
        self.base_lr = base_lr
        self.loraplus_ratio = loraplus_ratio
        
        # Separate A and B parameters
        params_A = [p for n, p in lora_params if 'lora_A' in n]
        params_B = [p for n, p in lora_params if 'lora_B' in n]
        
        # Different learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': params_A, 'lr': base_lr},
            {'params': params_B, 'lr': base_lr * loraplus_ratio}
        ])
```

---

## 5. Multi-Task Fine-Tuning

### 5.1 Instruction Tuning
Train on diverse tasks formatted as instructions.

**Dataset Structure**:
```json
[
    {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"},
    {"instruction": "Summarize", "input": "Long text...", "output": "Brief summary"},
    {"instruction": "Answer the question", "input": "What is 2+2?", "output": "4"}
]
```

**Training Objective**:
$$ \mathcal{L} = -\sum_{i} \log P(\text{output}_i | \text{instruction}_i, \text{input}_i) $$

---

### 5.2 FLAN (Fine-tuned LAnguage Net)
Google's instruction tuning framework.

**Key Components**:
- **Task Clustering**: Group similar tasks for better generalization
- **Chain-of-Thought Finetuning**: Include reasoning steps in training data
- **Multitask Unification**: All tasks as text-to-text

**FLAN-T5 Results**:
- 1.8T tokens of instruction data
- Outperforms models 10× larger on zero-shot tasks

---

### 5.3 Implementation: Multi-Task Fine-Tuning

```python
from datasets import load_dataset, concatenate_datasets

class MultiTaskFinetuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def load_instruction_datasets(self):
        """Load multiple instruction-tuning datasets."""
        datasets = {}
        
        # Load various datasets
        datasets['alpaca'] = load_dataset('tatsu-lab/alpaca')
        datasets['dolly'] = load_dataset('databricks/databricks-dolly-15k')
        datasets['gsm8k'] = load_dataset('openai/gsm8k', 'main')
        datasets['human_eval'] = load_dataset('openai/openai_humaneval')
        
        return datasets
    
    def format_instruction(self, instruction, input_text, output):
        """Format as instruction-following example."""
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Output:
{output}"""
    
    def prepare_dataset(self, datasets, max_length=512):
        """Combine and tokenize datasets."""
        def tokenize(example):
            formatted = self.format_instruction(
                example.get('instruction', ''),
                example.get('input', ''),
                example.get('output', '')
            )
            return self.tokenizer(
                formatted,
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
        
        # Combine all datasets
        combined = concatenate_datasets([d['train'] for d in datasets.values()])
        
        # Tokenize
        tokenized = combined.map(tokenize, batched=True)
        
        return tokenized
    
    def train(self, train_dataset, epochs=3, batch_size=8):
        """Fine-tune on multi-task data."""
        training_args = TrainingArguments(
            output_dir="./finetuned_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        trainer.train()
```

---

## 6. Continual Learning: Avoiding Catastrophic Forgetting

### 6.1 The Forgetting Problem
When fine-tuning on task B, the model forgets task A.

**Causes**:
- Weight updates for B conflict with A's optimal weights
- No mechanism to preserve previously learned knowledge

---

### 6.2 Elastic Weight Consolidation (EWC)
Add regularization to prevent important weights from changing.

**The Loss**:
$$ \mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2 $$

Where:
- $\mathcal{L}_B$: Loss for new task B
- $F_i$: Fisher information for parameter $i$ (importance)
- $\theta_{A,i}^*$: Optimal parameter for task A
- $\lambda$: Regularization strength

```python
class EWCFinetuning:
    def __init__(self, model, fisher_estimation_samples=100):
        self.model = model
        self.fisher_samples = fisher_estimation_samples
    
    def compute_fisher(self, train_loader, device):
        """Estimate Fisher information matrix (diagonal approximation)."""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        self.model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= self.fisher_samples:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Compute gradient
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    # Fisher = E[(gradient)^2]
                    fisher[n] += (p.grad ** 2) / self.fisher_samples
        
        return fisher
    
    def ewc_loss(self, fisher, old_params, lambda_ewc=1000):
        """Compute EWC regularization term."""
        loss = 0
        for n, p in self.model.named_parameters():
            if n in fisher:
                loss += (lambda_ewc / 2) * (fisher[n] * (p - old_params[n]) ** 2).sum()
        return loss
    
    def train_with_ewc(self, new_train_loader, old_train_loader, device):
        """Train on new task with EWC regularization."""
        # Compute Fisher on old task
        fisher = self.compute_fisher(old_train_loader, device)
        
        # Store old parameters
        old_params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(3):
            for inputs, targets in new_train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = self.model(inputs)
                base_loss = F.cross_entropy(outputs, targets)
                
                # Add EWC regularization
                ewc_reg = self.ewc_loss(fisher, old_params)
                total_loss = base_loss + ewc_reg
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
```

---

### 6.3 LoRA for Continual Learning
Use separate LoRA adapters for each task.

**Strategy**:
1.  Train LoRA-A on Task A, save adapter
2.  Train LoRA-B on Task B, save adapter
3.  At inference, load appropriate adapter or merge them

```python
class ContinualLoRA:
    def __init__(self, base_model):
        self.base_model = base_model
        self.task_adapters = {}
    
    def train_task(self, task_name, train_data, rank=8):
        """Train a new LoRA adapter for a task."""
        # Add LoRA adapters
        model = get_peft_model(
            self.base_model,
            LoraConfig(task_type="CAUSAL_LM", r=rank)
        )
        
        # Train
        trainer = Trainer(model=model, train_dataset=train_data, ...)
        trainer.train()
        
        # Save adapter
        self.task_adapters[task_name] = model.peft_config
        model.disable_adapter()
    
    def load_adapter(self, task_name):
        """Load adapter for specific task."""
        if task_name in self.task_adapters:
            self.base_model.load_adapter(self.task_adapters[task_name])
    
    def merge_adapters(self, task_names, weights=None):
        """Merge multiple task adapters."""
        if weights is None:
            weights = [1.0 / len(task_names)] * len(task_names)
        
        merged_state = {}
        for name, adapter in zip(task_names, weights):
            adapter_state = self.task_adapters[name]
            for key in adapter_state.state_dict():
                if key not in merged_state:
                    merged_state[key] = adapter * adapter_state.state_dict()[key]
                else:
                    merged_state[key] += adapter * adapter_state.state_dict()[key]
        
        return merged_state
```

---

## 7. Domain-Adaptive Fine-Tuning

### 7.1 Continued Pre-training
Further pre-train on domain-specific corpus before task fine-tuning.

**Process**:
1.  Collect domain corpus (e.g., medical papers, legal documents)
2.  Continue language modeling objective on domain text
3.  Fine-tune on downstream task

**Example: BioBERT**
- BERT further trained on PubMed abstracts
- Significant improvements on biomedical NLP tasks

---

### 7.2 Domain-Adaptive LoRA (DALoRA)
Combine domain adaptation with task adaptation.

```python
class DomainAdaptiveLoRA:
    def __init__(self, base_model, domain_name, task_name):
        self.base_model = base_model
        
        # Domain adapter (trained on domain corpus)
        self.domain_adapter = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
        
        # Task adapter (trained on task data)
        self.task_adapter = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
    
    def train_domain(self, domain_corpus):
        """Train on domain corpus with language modeling objective."""
        model = get_peft_model(self.base_model, self.domain_adapter)
        
        # Train with MLM objective
        trainer = Trainer(model=model, train_dataset=domain_corpus, ...)
        trainer.train()
        
        # Save and unload
        model.save_adapter(f"domain_{self.domain_name}")
        model.disable_adapter()
    
    def train_task(self, task_data):
        """Load domain adapter and train task adapter."""
        # Load domain adapter
        self.base_model.load_adapter(f"domain_{self.domain_name}")
        
        # Add task adapter
        model = get_peft_model(self.base_model, self.task_adapter)
        
        # Train on task
        trainer = Trainer(model=model, train_dataset=task_data, ...)
        trainer.train()
```

---

## 8. Fine-Tuning Best Practices

### 8.1 Hyperparameter Guidelines

| Parameter | LoRA | Full Fine-tuning | QLoRA |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | 1e-4 to 2e-4 | 1e-5 to 5e-5 | 2e-4 to 4e-4 |
| **Batch Size** | 8-32 | 4-16 | 8-32 |
| **LoRA Rank** | 8-64 | N/A | 16-128 |
| **LoRA Alpha** | 16-32 | N/A | 32-64 |
| **Epochs** | 3-10 | 3-5 | 5-15 |
| **Warmup** | 5-10% | 5-10% | 10-15% |

---

### 8.2 Common Issues and Solutions

| Issue | Symptom | Solution |
| :--- | :--- | :--- |
| **Overfitting** | Train loss ↓, Val loss ↑ | Reduce epochs, add dropout, lower rank |
| **Underfitting** | Both losses high | Increase rank, train longer, higher LR |
| **Catastrophic forgetting** | Base capabilities degraded | Use LoRA, EWC, or replay buffer |
| **GPU OOM** | CUDA out of memory | Use QLoRA, gradient checkpointing, smaller batch |
| **Unstable training** | Loss spikes, NaN | Lower LR, gradient clipping, warmup |

---

### 8.3 Evaluation Checklist

Before deploying fine-tuned model:

1.  **Task Performance**: Evaluate on held-out test set
2.  **Base Capabilities**: Check if general knowledge is preserved
3.  **Robustness**: Test on out-of-distribution examples
4.  **Bias**: Evaluate for unintended biases
5.  **Calibration**: Check if confidence scores are meaningful

---

## 🔬 Research Frontiers (2024-2025)

### 9.1 Efficient Fine-Tuning
- **LoRA-Pro**: Dynamic rank allocation during training
- **Meta-LoRA**: Learn to initialize LoRA for fast adaptation
- **Federated LoRA**: Distributed fine-tuning with privacy

### 9.2 Alignment Techniques
- **ORPO (Odds Ratio Preference Optimization)**: Combine SFT and DPO in one stage
- **SimPO (Simple Preference Optimization)**: Remove reference model requirement
- **KTO (Kahneman-Tversky Optimization)**: Optimize from binary feedback only

### 9.3 Specialized Adaptation
- **Task Arithmetic**: Combine task vectors for zero-shot multitask
- - **TIES-Merging**: Merge multiple fine-tuned models without interference
- **Model Soups**: Average weights of multiple fine-tuned models

---

**Status:** ✅ Elite Expanded Standard (14/10)
**Next:** CNN Architectures (Receptive Fields, Depthwise Conv, EfficientNet, Modern CNNs)
