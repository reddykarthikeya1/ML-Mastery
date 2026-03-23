# Phase 4: Specialization - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic NLP & CV Concept Recall
**1.1** Explain the difference between **Stemming** and **Lemmatization**.
**1.2** What is the fundamental difference between **Semantic Segmentation** and **Instance Segmentation**?
**1.3** Define **TF-IDF** and explain why it is used in information retrieval.
**1.4** What are the two core components of a **GAN**?
**1.5** What is the "Zero-shot" capability in models like **CLIP**?

### Level 2: Intermediate Architectural Logic
**2.1** In the **Attention** mechanism, why do we use "Query," "Key," and "Value" vectors? Describe their relationship.
**2.2** How does a **Residual (Skip) Connection** solve the vanishing gradient problem in deep CNNs like ResNet?
**2.3** Compare **One-Stage** (YOLO) and **Two-Stage** (Faster R-CNN) object detectors. What is the trade-off?
**2.4** What is the purpose of the **Forget Gate** in an LSTM cell?
**2.5** Explain **Compound Scaling** in EfficientNet.

### Level 3: Advanced Pipeline Analysis
**3.1** **RAG Optimization:** Your RAG system is retrieving irrelevant documents. Describe three techniques (e.g., Re-ranking, Hybrid Search) you would use to improve retrieval quality.
**3.2** **Fine-tuning Math:** How does **LoRA** (Low-Rank Adaptation) reduce the number of trainable parameters during fine-tuning? (Hint: $W = A \times B$).
**3.3** **Vision Transformers:** Why does a standard **Vision Transformer (ViT)** have quadratic complexity, and how does the **Swin Transformer** fix this?
**3.4** **Diffusion Physics:** Explain the difference between the **Forward Diffusion** process and the **Reverse Diffusion** process.

### Level 4: Python Implementation Practice
**4.1** **OpenAI/HuggingFace:** Write a Python script to load a pre-trained **Whisper** model and transcribe a 30-second audio file.
**4.2** **Computer Vision:** Write a PyTorch-style data augmentation pipeline that includes Resizing, Random Flipping, and Color Jittering.
**4.3** **NLP Preprocessing:** Implement a simple "Self-Attention" calculation using NumPy for a 3-token sequence with 4-dimensional embeddings.

### Level 5: Real-world System Design
**5.1** **Scenario:** You are building an "AI Assistant for Radiologists."
- **The Task:** Automatically detect and segment potential tumors in high-resolution CT scans.
- **The Constraint:** The system must be able to cite its medical "reasoning" by looking up similar historical cases from a massive medical database.
**Task:** Propose a hybrid architecture. Which CV model would you use for detection/segmentation? How would you integrate a **RAG** architecture to provide the "reasoning" from medical journals? Which evaluation metrics are most critical here?

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.2
Semantic segmentation classifies every pixel into a category (e.g., all cars are blue). Instance segmentation identifies and separates every individual object (e.g., Car 1 is blue, Car 2 is red).

### 2.1
- **Query (Q)**: What I am looking for.
- **Key (K)**: What I have to offer.
- **Value (V)**: The actual information.
The model computes a score (dot product) between Q and K to determine "how much attention" to pay to each V.

### 3.2
LoRA freezes the original weight matrix $W$ ($d \times d$). It learns two low-rank matrices $A$ ($d \times r$) and $B$ ($r \times d$), where $r << d$. The number of parameters drops from $d^2$ to $2 \times d \times r$.

### 4.1
```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

### 5.1
- **Vision:** Use a **Mask R-CNN** or **Swin Transformer** for detection/segmentation.
- **Reasoning:** Convert segmented tumor features into a vector and use a **Multimodal RAG** system to query a Vector DB of medical cases.
- **Metrics:** Prioritize **Recall** (Sensitivity) and **Dice Coefficient** (for mask accuracy).

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Phase 4 Specialization Complete!
