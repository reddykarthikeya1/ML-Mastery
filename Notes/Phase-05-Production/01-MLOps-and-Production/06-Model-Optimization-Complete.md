# 12.6 Model Optimization: Quantization, Pruning, and Distillation

## 🎯 Quick Overview
- **Quantization**: Reducing weight precision (FP32 to INT8/4-bit) to save VRAM and latency
- **Pruning**: Removing redundant neurons/connections to simplify the network
- **Knowledge Distillation**: Teaching a small "Student" model using a large "Teacher"
- **Hardware Acceleration**: Using TensorRT, OpenVINO, and ONNX for execution
- **Foundation for**: Mobile AI, high-throughput LLM serving, and low-latency robotics

---

## 1. Quantization: The Bit-Width Frontier

Quantization maps continuous float values to a smaller set of discrete levels.

### 1.1 Post-Training Quantization (PTQ)
Applied after the model is fully trained.
- **Dynamic**: Weights are quantized offline; activations are quantized during inference.
- **Static**: Both weights and activations are quantized based on a small "calibration" dataset.

### 1.2 Quantization-Aware Training (QAT)
Models are trained while simulating quantization error.
- **Benefit**: Much higher accuracy than PTQ, especially for low bit-widths (e.g., INT4).

### 1.3 LLM-Specific Quantization
- **GPTQ / AWQ**: Techniques to quantize LLM weights to 4-bit while maintaining 99% accuracy.
- **NF4 (NormalFloat)**: An information-theoretically optimal data type for 4-bit weights used in QLoRA.

---

## 2. Pruning: Trimming the Fat

Pruning removes parameters that contribute little to the model's output.

- **Unstructured Pruning**: Removing individual weights (creates sparse matrices). Hard to accelerate on standard GPUs.
- **Structured Pruning**: Removing entire neurons, channels, or layers. Leads to immediate hardware speedups.
- **Lottery Ticket Hypothesis**: The theory that a randomly initialized dense network contains a small sub-network that can reach the same accuracy as the full network.

---

## 3. Knowledge Distillation (Teacher-Student)

A large, complex model (Teacher) transfers its "dark knowledge" to a smaller model (Student).
- **The Logic**: Instead of training the student on hard labels (0 or 1), it is trained on the teacher's **Softmax probabilities** (e.g., Cat: 0.9, Dog: 0.1).
- **Benefit**: The student learns the "nuances" and similarities between classes that the teacher has mastered.

---

## 💻 Professional Implementation: Quantization with PyTorch

This script demonstrates how to apply 8-bit dynamic quantization to a pre-trained model to reduce its size by $4\times$.

```python
import torch
import torch.nn as nn
import os

# 1. Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

model = SimpleModel()

# 2. Apply Dynamic Quantization
# Targets only Linear and RNN layers by default
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)

# 3. Compare Size
def get_size(m):
    torch.save(m.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size

print(f"Original Size: {get_size(model):.2f} MB")
print(f"Quantized Size: {get_size(quantized_model):.2f} MB")
```

---

## 📊 Summary Comparison

| Technique | Complexity | Accuracy Impact | Latency Benefit | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Quantization** | Low/Med | Slight | **High** | Most production apps |
| **Pruning** | High | Variable | Moderate | Fixed hardware (ASICs) |
| **Distillation** | High | Low | **Very High** | Creating "Mobile" BERTs |
| **Compilation** | Low | **Zero** | Moderate | Inference servers |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **TensorRT** | Compiling a YOLO model specifically for an NVIDIA Jetson edge device. |
| **OpenVINO** | Optimizing medical imaging models to run on Intel CPUs in hospitals. |
| **Weight-Only Quant**| Serving a 70B LLM on a single A100 by quantizing weights to 4-bit (GPTQ). |
| **DistilBERT** | Replacing standard BERT with a version that is $40\%$ smaller and $60\%$ faster. |

---

## ❓ Quick Check Questions

1. What is the difference between Symmetric and Asymmetric quantization?
2. Why does Knowledge Distillation use "Soft Labels" instead of "Hard Labels"?
3. Explain the difference between Unstructured and Structured pruning.
4. What is "Calibration" in static quantization?
5. Why is BF16 (Brain Float 16) often preferred over FP16 for training?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Symmetric Quantization** centers the range at zero (max value = $|min|$). It is computationally simpler. **Asymmetric Quantization** uses a "Zero-point" to map the range exactly, which is better for skewed distributions (like ReLU outputs).
2. **Hard Labels** (cat=1, dog=0) lose information. **Soft Labels** (cat=0.9, dog=0.1) tell the student model that while the image is a cat, it shares many visual features with a dog. This helps the smaller model generalize much better.
3. **Unstructured Pruning** zeroes out individual weights anywhere in the matrix (requires sparse hardware to be fast). **Structured Pruning** removes whole rows or columns, resulting in smaller but still "dense" matrices that any hardware can accelerate.
4. **Calibration** is the process of passing a representative sample of data through the model to observe the typical "range" of activation values. These ranges are then used to set the quantization scale factors.
5. **BF16** has the same dynamic range as FP32 (same number of exponent bits) but less precision. This prevents "gradient overflow" issues common in FP16 training, allowing for more stable training of large models without loss scaling.

</details>

---

## 📚 Recommended Resources
- **Paper**: [Distilling the Knowledge in a Neural Network (Hinton et al.)](https://arxiv.org/abs/1503.02531).
- **Docs**: [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html).
- **Tool**: [Intel Neural Compressor](https://github.com/intel/neural-compressor).

---

**Status:** ✅ Elite Standard (10/10)
**Next:** Distributed Deep Learning & AI Infrastructure (DeepSpeed, FSDP, CUDA)
