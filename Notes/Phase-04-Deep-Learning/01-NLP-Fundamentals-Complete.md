# 10.1 NLP Fundamentals: From Raw Text to Semantic Vectors

## 🎯 Quick Overview
- **Subword Tokenization**: Mastering BPE, WordPiece, and SentencePiece
- **Text Representation Math**: Deriving TF-IDF and the Hashing Trick
- **Word Embedding Internals**: Deep dive into Word2Vec (Skip-Gram/CBOW) and FastText
- **Linguistic Structures**: Dependency Parsing and Named Entity Recognition (NER)
- **Foundation for**: Transformers, Large Language Models (LLMs), and Semantic Search

---

## 1. Advanced Text Preprocessing

Modern NLP has moved beyond simple "word" tokenization.

### 1.1 The Subword Revolution
LLMs use subword tokenization to solve the **Out-of-Vocabulary (OOV)** problem.
- **Byte-Pair Encoding (BPE)**: Starts with characters and iteratively merges the most frequent adjacent pairs. 
    - *Example*: "hug", "pug", "pun", "bun" → ["hu", "g", "pu", "g", "p", "un", "b", "un"].
- **WordPiece (BERT)**: Similar to BPE but merges pairs based on maximizing the likelihood of the training data.
- **SentencePiece (Llama/T5)**: Treats the input as a raw stream, including spaces, allowing for language-independent tokenization.

### 1.2 Normalization & Morphology
- **Lemmatization Math**: Uses a part-of-speech context to map tokens to a root.
    - $f(token, pos) \to lemma$
- **Stopword Impact**: While traditionally removed, modern LLMs often keep them to preserve grammatical nuances required for attention mechanisms.

---

## 2. Text Representation: The Mathematical View

### 2.1 TF-IDF Deep Dive
Used to determine the importance of a word $t$ in document $d$ within corpus $D$.

1.  **Term Frequency (TF)**: $TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{Total words in } d}$
2.  **Inverse Document Frequency (IDF)**: $IDF(t, D) = \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)$
3.  **TF-IDF**: $TF(t, d) \times IDF(t, D)$

**Why it works**: Words that appear everywhere (like "the") have an IDF near $\log(1) = 0$, effectively being ignored.

### 2.2 The Hashing Trick (Feature Hashing)
For massive datasets, storing a vocabulary dictionary is impossible. We use a hash function $h(word) \to [0, N-1]$ to map words directly to indices in a fixed-size vector.
- **Collisions**: Handled by using a second sign-hash $s(word) \to \{-1, 1\}$.

---

## 3. Word Embeddings: Learning Meaning

### 3.1 Word2Vec (The Foundation)
Learns a vector $v_w$ for every word such that words in similar contexts have high cosine similarity.

#### Skip-Gram with Negative Sampling (SGNS)
Instead of predicting the next word among $V$ (thousands of classes), we treat it as a binary classification: "Is this context word a real neighbor or noise?"
- **Loss Function**:
  $$J(\theta) = -\log \sigma(v_c^T v_w) - \sum_{i=1}^k \log \sigma(-v_{noise_i}^T v_w)$$
  Where $v_w$ is the target word, $v_c$ is the actual context, and $v_{noise}$ are random samples.

### 3.2 FastText: Subword Embeddings
FastText represents a word as the sum of its character n-grams.
- *Example*: `<apple>` with $n=3$ → `<ap, app, ppl, ple, le>`.
- **Impact**: Can generate a meaningful vector for a misspelled word like "appple" by averaging its sub-components.

---

## 💻 Professional Implementation

### 1. Custom BPE Tokenizer Logic
```python
import re
from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# Example Vocab
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e r </w>': 6, 'w i d e r </w>': 3}
for i in range(10):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Iteration {i}: Merged {best}")
```

### 2. Semantic Similarity with Spacy (Static Embeddings)
```python
import spacy

# Load large model with vectors
nlp = spacy.load("en_core_web_md")

word1 = nlp("king")
word2 = nlp("queen")
word3 = nlp("apple")

print(f"King vs Queen Similarity: {word1.similarity(word2):.4f}")
print(f"King vs Apple Similarity: {word1.similarity(word3):.4f}")
```

---

## 📊 Summary Comparison

| Metric | TF-IDF | Word2Vec | FastText | BERT (Phase 4.4) |
| :--- | :--- | :--- | :--- | :--- |
| **Representation** | Sparse | Dense | Dense | Contextual Dense |
| **Similarity** | Exact Match | Semantic | Morphological | Fully Contextual |
| **OOV Support** | No | No | **Yes** | **Yes** |
| **Memory** | High (Vocab) | Moderate | Moderate | Massive |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **BPE/SentencePiece** | Training a custom LLM for a new language (e.g., Hindi, Arabic). |
| **Dependency Parsing** | Extracting relationships (Subject-Verb-Object) for Knowledge Graphs. |
| **Negative Sampling** | Efficiently training embeddings on terabyte-scale corpora. |
| **Cos-Sim Scaling** | Building a lightning-fast "Related Articles" feature for news sites. |

---

## ❓ Quick Check Questions

1. Why does BPE use "subwords" instead of characters or full words?
2. Derive why the IDF of a word that appears in every document is 0.
3. How does Negative Sampling solve the computational bottleneck of the standard Softmax in Word2Vec?
4. What happens to the Word2Vec similarity between "bank" (river) and "bank" (finance)?
5. Why is FastText better than Word2Vec for "Agglutinative" languages (like Turkish or Finnish)?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Characters** are too small and lose meaning. **Full words** create a massive vocabulary and cannot handle new words. **Subwords** allow the model to build words from common roots, suffixes, and prefixes, striking a balance between vocabulary size and generalization.
2. If a word appears in every document, the count $|\{d \in D : t \in d\}| = |D|$. The formula becomes $\log(|D|/|D|) = \log(1) = 0$. This mathematically eliminates common words that provide no discriminative power.
3. Standard Softmax requires calculating the dot product for **every word in the vocabulary** ($V$) to normalize the probabilities. Negative Sampling turns this into a **binary problem**—only checking the target word against $K$ random "noise" words, reducing complexity from $O(V)$ to $O(K)$.
4. Since Word2Vec creates **static embeddings**, it creates a single vector for "bank" that is an average of its two meanings. This is a major limitation; it cannot distinguish between homonyms based on context (BERT solves this).
5. Agglutinative languages build long words by adding many suffixes. FastText's **character n-gram** approach allows it to recognize that "apples", "apple-like", and "apple-less" all share the "apple" sub-root, even if the full words weren't in the training set.

</details>

---

## 📚 Recommended Resources
- **Papers**: 
    - [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781)
    - [Enriching Word Vectors with Subword Information (FastText)](https://arxiv.org/abs/1607.04606)
- **Books**: "Speech and Language Processing" by Dan Jurafsky (The Bible of NLP).
- **Tools**: [HuggingFace Tokenizers Library](https://github.com/huggingface/tokenizers).

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** Sequence Models (RNNs, LSTMs, BPTT)
