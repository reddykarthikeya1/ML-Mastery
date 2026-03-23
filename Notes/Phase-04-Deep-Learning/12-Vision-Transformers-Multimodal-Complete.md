# 11.5 Advanced Vision Transformers & Multimodal AI

## 🎯 Quick Overview
- **ViT Math**: Linear projection of patches and Position Embedding interpolation
- **Swin Transformer**: Shifted Window math and Hierarchical complexity reduction
- **Multimodal Learning**: CLIP's Contrastive Loss and LLaVA's Projection Layer
- **Emergent Vision**: DINO (Self-distillation) and MAE (Masked Autoencoders)
- **Foundation for**: Multimodal LLMs (GPT-4V), Image search, and Zero-shot classification

---

## 1. Vision Transformers (ViT) Deep Dive

ViT treats an image as a sequence of tokens, just like a sentence.

### 1.1 Patch Embedding Math
1.  **Image**: $x \in \mathbb{R}^{H \times W \times C}$.
2.  **Patches**: Split into $N = (HW) / P^2$ patches of size $(P, P, C)$.
3.  **Projection**: Flatten patches and project to $D$ dimensions using a trainable linear layer $E$:
    $$ z_0 = [x_{class}; x_p^1 E; x_p^2 E; \dots; x_p^N E] + E_{pos} $$
    - $x_{class}$ is a special "learnable" token used for classification.
    - $E_{pos}$ provides the 1D spatial information.

---

## 2. Multimodal AI: Aligning Vision and Text

### 2.1 CLIP Contrastive Loss
CLIP learns by maximizing the similarity between correct (Image, Text) pairs and minimizing it for all other pairs in a batch.
- **The Batch**: $N$ pairs of (image, text).
- **The Matrix**: $N \times N$ similarity scores.
- **Loss**: Cross-entropy across the rows (finding the right text for an image) and columns (finding the right image for a text).

### 2.2 LLaVA: The Visual Connector
LLaVA bridges a Vision Encoder (CLIP) and a Language Model (LLaMA).
- **The Projection Layer**: A simple MLP or linear layer that maps the $1024$-dim CLIP features into the $4096$-dim space of the LLM.
- **Visual Instruction Tuning**: The model is trained on data like:
    - *User*: "What is the man in the photo holding?"
    - *Assistant*: "He is holding a red umbrella."

---

## 3. Self-Supervised Vision

How to learn without labels?

### 3.1 Masked Autoencoders (MAE)
1.  **Masking**: Randomly hide $75\%$ of image patches.
2.  **Encoder**: Only processes the *unmasked* patches (very efficient).
3.  **Decoder**: Tries to reconstruct the original image from the sparse features.
- **Result**: Learns extremely robust high-level visual features.

### 3.2 DINO (Self-Distillation)
Uses a **Student** and **Teacher** network. Both see different "views" (crops) of the same image. The student tries to predict the teacher's output.
- **Emergent Property**: DINO naturally learns to segment objects without ever being shown a segmentation mask!

---

## 💻 Professional Implementation

### 1. Patch Partitioning Logic (NumPy)
```python
import numpy as np

def patchify(image, patch_size):
    # image: (C, H, W)
    C, H, W = image.shape
    # Reshape into patches
    patches = image.reshape(C, H//patch_size, patch_size, W//patch_size, patch_size)
    # Transpose to (N, P*P*C)
    patches = patches.transpose(1, 3, 2, 4, 0).reshape(-1, patch_size * patch_size * C)
    return patches

# Example: 224x224 image, 16x16 patches
img = np.random.randn(3, 224, 224)
p = patchify(img, 16)
print(f"Number of patches: {p.shape[0]}") # 196
```

### 2. LLaVA-style Linear Projection (PyTorch)
```python
import torch.nn as nn

class VisionLanguageConnector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        # Project CLIP features to LLM embedding space
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, visual_features):
        return self.projector(visual_features)
```

---

## 📊 Summary Comparison

| Model | Architecture | Logic | Best For |
| :--- | :--- | :--- | :--- |
| **ResNet** | CNN | Local Convolutions | General Vision |
| **ViT** | Transformer | Global Attention | Large Pre-training |
| **CLIP** | Dual-Encoder | Contrastive Alignment| Search / Zero-shot |
| **LLaVA** | Vision + LLM | Generative Alignment| Chatting with Images|

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Zero-shot Seg.** | Using CLIP features to find "anomalies" in production lines without training data. |
| **Multi-modal RAG**| Searching a vector DB for images using text queries (and vice versa). |
| **Self-Attention Viz**| Visualizing what the model "looks at" when classifying an image. |
| **Video-LLM** | Extending LLaVA to handle a sequence of frames for video understanding. |

---

## ❓ Quick Check Questions

1. Why does ViT need a `[CLS]` (class) token?
2. Explain the "Quadratic Bottleneck" of ViT and how Swin Transformer solves it.
3. In CLIP, why is the similarity matrix $N \times N$ instead of $N \times 1$?
4. What is the "Projection Layer" in a Multimodal LLM actually doing?
5. How does a Masked Autoencoder (MAE) learn visual features?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Since Transformers are designed for sequence-to-sequence, there is no single output for an image. The **`[CLS]` token** is a learnable vector added to the start of the sequence. Its final state is used as the aggregate representation of the entire image for classification.
2. ViT uses global self-attention (every patch looks at every other patch), which is **$O(N^2)$**. Swin Transformer computes attention in **local windows**, reducing complexity to **$O(N)$**. It uses "shifted windows" to allow information to flow between local regions in deeper layers.
3. Because each image in a batch must be compared against **every text string** in that same batch. The diagonal contains the correct pairs, while all other cells are "negatives" that the model learns to push apart.
4. It acts as a **translator**. The vision model and the language model "speak different languages" (different vector dimensionalities). The projection layer transforms the visual vectors into the language model's space so the LLM can treat them as just another set of tokens.
5. MAE learns by **reconstruction**. By masking $75\%$ of the pixels and forcing the model to predict them, the model is forced to learn a deep internal understanding of geometry, textures, and object structures to fill in the blanks realistically.

</details>

---

## 📚 Recommended Resources
- **Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
- **Paper**: [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- **Repo**: [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA).

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** Audio & Speech Processing (FFT math, Mel-scale, CTC Loss)
