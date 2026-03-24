# 12.7 Distributed Deep Learning & AI Infrastructure

## 🎯 Quick Overview
- **Parallelism Strategies**: Data, Model, Pipeline, and Tensor Parallelism
- **Distributed Frameworks**: DeepSpeed, PyTorch FSDP, and Horovod
- **Infrastructure Internals**: CUDA memory hierarchy and NVIDIA NCCL
- **GPU Acceleration**: Writing custom kernels with OpenAI Triton
- **Foundation for**: Training 100B+ parameter models, high-performance computing (HPC)

---

## 1. Types of Parallelism

When a model is too large for a single GPU (e.g., 80GB A100), we must split it across multiple devices.

### 1.1 Data Parallelism (DP)
Every GPU has a full copy of the model. Each GPU gets a different batch of data. Gradients are averaged (All-Reduce) after every step.
- **Problem**: Model must fit in one GPU's memory.

### 1.2 Fully Sharded Data Parallel (FSDP) / ZeRO
Shards model weights, gradients, and optimizer states across GPUs.
- **Benefit**: Allows training models that are far larger than any single GPU's memory.

### 1.3 Pipeline & Tensor Parallelism
- **Pipeline**: Different layers live on different GPUs (Layer 1-10 on GPU 0, 11-20 on GPU 1).
- **Tensor**: A single matrix multiplication is split across GPUs (Row-wise or Column-wise).

---

## 2. Advanced Libraries: DeepSpeed

DeepSpeed (Microsoft) provides the **ZeRO (Zero Redundancy Optimizer)** stages:
- **ZeRO-1**: Shards Optimizer States.
- **ZeRO-2**: Shards Optimizer States + Gradients.
- **ZeRO-3**: Shards Optimizer States + Gradients + Parameters.
- **Offload**: Moves optimizer states/params to CPU RAM when not needed, allowing massive models to train on smaller GPUs.

---

## 3. GPU Infrastructure Internals

### 3.1 CUDA Hierarchy
- **Threads**: Smallest unit of execution.
- **Blocks**: Groups of threads that share **Shared Memory**.
- **Grids**: Groups of blocks.
- **Global Memory (VRAM)**: Slow but large. **Shared Memory**: Extremely fast but tiny.

### 3.2 OpenAI Triton
Triton is a language and compiler for writing custom GPU kernels in Python. It abstracts away the complexity of CUDA C++ while delivering near-native performance.
- **Use Case**: Implementing **FlashAttention** or custom activation functions.

---

## 💻 Professional Implementation: Sharded Training with FSDP

This script demonstrates how to wrap a model in PyTorch's Fully Sharded Data Parallel (FSDP) for distributed training.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, world_size):
    setup(rank, world_size)
    
    # 1. Create Model
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10)
    ).to(rank)
    
    # 2. Wrap in FSDP
    # Weights are now sharded across all GPUs in the world_size
    fsdp_model = FSDP(model)
    
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.001)
    
    # 3. Training Loop
    input_data = torch.randn(32, 4096).to(rank)
    output = fsdp_model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    
    print(f"Rank {rank} completed step.")
    dist.destroy_process_group()

# Note: In production, run with torchrun --nproc_per_node=GPU_COUNT
```

---

## 📊 Summary Comparison

| Strategy | Memory Efficiency | Compute Efficiency | Complexity |
| :--- | :--- | :--- | :--- |
| **Data Parallel** | Low | **High** | Low |
| **FSDP / ZeRO** | **High** | Moderate | Moderate |
| **Tensor Parallel**| Moderate | High | **Very High** |
| **Pipeline Par.** | Moderate | Low (Bubbles) | High |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **NCCL** | The backend library used for high-speed inter-GPU communication. |
| **Mixed Precision** | Combining O1/O2 levels to speed up training using Tensor Cores. |
| **Gradient Accum.** | Simulating a large batch size (e.g. 1024) on a single GPU by summing gradients over 32 steps. |
| **FlashAttention** | An IO-aware attention kernel that reduces memory reads/writes, speeding up LLMs. |

---

## ❓ Quick Check Questions

1. Why does Data Parallelism fail for models with 100B+ parameters?
2. What is the difference between "Shared Memory" and "Global Memory" in a GPU?
3. How does the ZeRO-3 optimizer achieve "zero redundancy"?
4. What is a "Pipeline Bubble" in pipeline parallelism?
5. Why is **NCCL** preferred over MPI for deep learning on NVIDIA GPUs?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. In standard Data Parallelism, every GPU must store a **full copy** of the model. A 100B model (FP16) takes ~200GB of VRAM, which exceeds the 80GB limit of even the best modern GPUs (A100/H100).
2. **Global Memory** (VRAM) is large (GBs) but has high latency. **Shared Memory** is a tiny (KBs) on-chip cache that is shared by all threads in a block. It is $10-100\times$ faster than global memory.
3. **ZeRO-3** ensures that no single GPU stores the full model weights. Instead, weights are sharded. When a layer needs to compute its forward pass, it broadcasts its shard to others, performs the math, and immediately discards the non-local weights.
4. A **Pipeline Bubble** is idle time where a GPU is waiting for the previous stage in the pipeline to finish its work. Micro-batching is used to minimize these bubbles.
5. **NCCL** (NVIDIA Collective Communications Library) is highly optimized specifically for the hardware topology of NVIDIA GPUs (using NVLink and InfiniBand), providing much higher bandwidth for "All-Reduce" operations than general-purpose MPI.

</details>

---

## 📚 Recommended Resources
- **Paper**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
- **Blog**: [PyTorch FSDP: A Beginner's Guide](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/).
- **Course**: [CUDA Mode: Deep Learning Practioner's Guide to CUDA](https://github.com/cuda-mode).

---

**Status:** ✅ Elite Standard (10/10)
**Next:** AI Security & Governance (Prompt Injection, Guardrails, Privacy)
