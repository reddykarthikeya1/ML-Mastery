# 10.4 Large Language Model Evolution (BERT to LLaMA)

## 🎯 Quick Overview
- **BERT**: Encoder-only models for deep text understanding (Bidirectional)
- **GPT Family**: Decoder-only models for text generation (Autoregressive)
- **T5**: Encoder-Decoder models for unified text-to-text tasks
- **LLaMA & Open Source**: The democratization of high-performance LLMs
- **Foundation for**: Modern AI Assistants, RAG systems, and specialized fine-tuning

---

## 1. BERT: The Understanding King (Encoder-Only)

**BERT** (Bidirectional Encoder Representations from Transformers) changed NLP in 2018 by looking at words in both directions simultaneously.

### 1.1 Pre-training Objectives
1. **Masked Language Modeling (MLM)**: Hide 15% of tokens and predict them. (e.g., "The [MASK] sat on the mat").
2. **Next Sentence Prediction (NSP)**: Predict if sentence B follows sentence A.

### 1.2 Variants
- **RoBERTa**: BERT with more data and removed NSP (Better performance).
- **DistilBERT**: Smaller, faster version using Knowledge Distillation.

---

## 2. GPT: The Generation Giant (Decoder-Only)

**GPT** (Generative Pre-trained Transformer) models are **Autoregressive**, meaning they predict the next token based *only* on the tokens that came before.

### 2.1 The Evolution
- **GPT-1**: Demonstrated that unsupervised pre-training works.
- **GPT-2**: Showed **Zero-shot** capabilities (learning tasks without specific training).
- **GPT-3**: Massive scale (175B parameters). Introduced **In-Context Learning** (Few-shot prompting).
- **GPT-4**: Multimodal capabilities and superior reasoning.

---

## 3. T5: The Unified Framework (Encoder-Decoder)

**T5** (Text-to-Text Transfer Transformer) treats every NLP task as a "text-to-text" problem.
- **Example**: To summarize, the input is `"summarize: [Text]"` and the output is the summary text.

---

## 4. The Modern Open Source Era (LLaMA & Beyond)

Meta's **LLaMA** (Large Language Model Meta AI) proved that smaller, well-trained models can outperform larger ones.
- **Key Innovations**: RMSNorm (normalization), SwiGLU activation, and Rotary Positional Embeddings (RoPE).
- **Impact**: Led to the explosion of open-source models like Mistral, Mixtral (MoE), and Falcon.

---

## 💻 Python Code Examples

### 1. Feature Extraction with BERT (HuggingFace)
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Natural Language Processing is amazing!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Last hidden state: (batch, seq_len, hidden_dim)
embeddings = outputs.last_hidden_state
print(embeddings.shape)
```

### 2. Text Generation with GPT-2
```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
prompt = "The future of AI is"
result = generator(prompt, max_length=30, num_return_sequences=1)

print(result[0]['generated_text'])
```

---

## 📊 Summary Table

| Model | Architecture | Direction | Primary Use Case |
|-------|--------------|-----------|------------------|
| **BERT** | Encoder-only | Bidirectional | NER, Classification, Q&A |
| **GPT** | Decoder-only | Unidirectional | Creative Writing, Chat, Coding |
| **T5** | Encoder-Decoder | Unified | Translation, Summarization |
| **LLaMA** | Decoder-only | Unidirectional | General purpose Open-Source |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| BERT Fine-tuning | Sentiment analysis for stock markets |
| GPT Prompting | Automated customer support bots |
| T5 | Multi-lingual translation engines |
| LLaMA (Quantized) | Running powerful LLMs on a laptop (Ollama) |

---

## ❓ Quick Check Questions

1. Why is BERT called "Bidirectional" while GPT is "Unidirectional"?
2. What is the difference between "Zero-shot" and "Few-shot" learning in the context of GPT-3?
3. How does T5 simplify the approach to different NLP tasks?
4. What is "Masked Language Modeling," and which architecture uses it?
5. Why did the LLaMA model release trigger a "Cambrian Explosion" in the AI community?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **BERT** sees the entire sentence at once, allowing every word to attend to both its left and right neighbors. **GPT** only looks at previous words to predict the next one, strictly following the flow of time/text.
2. **Zero-shot**: Providing a task description but no examples. **Few-shot**: Providing a task description followed by 2-5 examples of the task within the prompt to guide the model.
3. It converts every task—whether it's translation, sentiment analysis, or regression—into a simple **text-to-text** string mapping, meaning one single model can be used for any task without changing the output layer.
4. **MLM** is a pre-training objective where some words in a sentence are replaced with a `[MASK]` token, and the model must guess the original word. It is used by **Encoder-only** architectures like BERT.
5. LLaMA provided weights for a high-performance model that was small enough to run on consumer hardware. This allowed researchers and developers worldwide to build specialized models (like Alpaca or Vicuna) without needing a supercomputer.

</details>

---

**Status:** ✅ Complete
**Next:** LLMs and Prompt Engineering (The Art of Interaction)
