# 11.5 Vision Transformers & Multimodal AI (ViT, CLIP, LLaVA)

## 🎯 Quick Overview
- **Vision Transformers (ViT)**: Applying the Transformer architecture to images (no more convolutions!)
- **Shifted Windows (Swin)**: Solving the quadratic complexity of ViT for high-res images
- **Multimodal AI**: Models that understand both text and images simultaneously
- **CLIP**: Learning visual concepts from natural language supervision
- **LLaVA**: The emergence of "Multimodal LLMs" (Visual Instruction Tuning)
- **Foundation for**: Modern search engines, AI image captioning, and multimodal assistants (GPT-4V)

---

## 1. Vision Transformers (ViT)

In 2020, researchers showed that a pure Transformer can outperform CNNs on images if trained on enough data.

### 1.1 How it works
1. **Patch Partitioning**: The image is split into a grid of fixed-size patches (e.g., $16 \times 16$ pixels).
2. **Linear Projection**: Each patch is flattened and projected into a vector embedding.
3. **Position Embeddings**: Since Transformers are permutation-invariant, 1D position embeddings are added to the patch embeddings.
4. **Transformer Encoder**: The patches are treated exactly like "words" in a sentence.

- **Limitation**: Standard ViT has **quadratic complexity** ($O(n^2)$) relative to the number of patches, making high-resolution images very expensive to process.

---

## 2. Swin Transformer (The Improved ViT)

Introduced **Hierarchical** feature maps and **Shifted Windows**.
- It computes self-attention only within local windows, which reduces complexity to **linear** ($O(n)$) relative to image size.
- It is the standard "backbone" for modern vision tasks like detection and segmentation.

---

## 3. Multimodal AI: Connecting Vision and Language

### 3.1 CLIP (Contrastive Language-Image Pre-training)
CLIP is trained on 400M image-text pairs from the internet.
- **The Goal**: It learns to put an image and its corresponding caption close together in a shared vector space.
- **Why it's revolutionary**: It enables **Zero-shot** image classification. You can give it a list of labels like ["cat", "dog", "ufo"] and it will pick the right one without ever being specifically trained on those labels.

### 3.2 LLaVA (Large Language-and-Vision Assistant)
LLaVA connects a visual encoder (CLIP) with a language model (LLaMA) using a simple linear layer.
- **Visual Instruction Tuning**: It allows you to "talk" to an image (e.g., "Describe what's happening in this photo").

---

## 💻 Python Code Examples

### 1. Zero-Shot Classification with CLIP (HuggingFace)
```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], 
    images=image, 
    return_tensors="pt", 
    padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # Image-text similarity scores
probs = logits_per_image.softmax(dim=1) # Get probabilities
print(probs)
```

---

## 📊 Summary Table

| Architecture | Mechanism | Best For | Complexity |
|--------------|-----------|----------|------------|
| **ResNet** | Convolutions | General Image Tasks | Moderate |
| **ViT** | Global Attention | Large-scale Pre-training | High |
| **Swin** | Windowed Attention| High-res Detection/Seg | Moderate |
| **CLIP** | Contrastive | Search, Zero-shot, Alignment | Very High |
| **LLaVA** | Vision + LLM | Chatting with images | Massive |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| ViT | Foundation for modern ImageNet SOTA |
| CLIP | Semantic image search (Google Photos style) |
| LLaVA | Automated medical report generation from X-rays |
| Multi-modal | Content moderation (identifying text in images) |

---

## ❓ Quick Check Questions

1. Why do we need to split images into "patches" for Vision Transformers?
2. What is the fundamental difference between a CNN's "Local Receptive Field" and a ViT's "Global Self-Attention"?
3. How does the Swin Transformer achieve linear complexity?
4. Explain the concept of a "Shared Embedding Space" in CLIP.
5. In LLaVA, how are the visual features connected to the language model?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Transformers were designed for sequences of tokens (words). An image has too many pixels to treat each one as a token (e.g., $224 \times 224 = 50,176$ tokens). **Patching** reduces the sequence length to a manageable number (e.g., $14 \times 14 = 196$ patches).
2. A **CNN** processes information locally (only nearby pixels interact in early layers). A **ViT** uses self-attention, allowing every patch to interact with every other patch in the image immediately, capturing long-range dependencies from the very first layer.
3. Swin Transformer computes attention within **non-overlapping local windows**. To allow information flow between windows, it "shifts" the window boundaries in successive layers, combining local efficiency with global connectivity.
4. CLIP trains an image encoder and a text encoder simultaneously to produce vectors. It ensures that an image of a "sunset" and the text "a beautiful sunset" result in vectors that are very close (high cosine similarity) in the **same vector space**.
5. LLaVA uses a **Vision Encoder** (like CLIP) to extract features from the image. These features are then passed through a **projection layer** (adapter) that translates them into the same format/dimensionality as the LLM's word embeddings, allowing the LLM to "see" the image features as if they were text tokens.

</details>

---

**Status:** ✅ Complete
**Next:** Audio & Speech Processing (Whisper, Wav2Vec)
