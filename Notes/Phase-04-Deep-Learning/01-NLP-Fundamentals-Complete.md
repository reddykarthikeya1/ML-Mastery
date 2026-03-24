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

## 4. Advanced Topics: Beyond the Basics

### 4.1 Attention Mechanisms (The Bridge to Transformers)
Before Transformers, **Attention** was introduced to solve the Seq2Seq bottleneck.

#### The Bahdanau Attention (Additive):
$$ \alpha_{ij} = \frac{\exp(\text{score}(h_i, s_j))}{\sum_{k} \exp(\text{score}(h_i, s_k))} $$
Where $h_i$ is the encoder state and $s_j$ is the decoder state.

#### Luong Attention (Multiplicative):
$$ \text{score}(h_i, s_j) = h_i^T W s_j $$
- **Global Attention**: Attends to all source positions (similar to modern self-attention).
- **Local Attention**: Attends to a window of source positions (faster, $O(1)$ per step).

**Key Insight**: Attention allows the decoder to "look back" at any encoder state, eliminating the bottleneck. This is the conceptual predecessor to Transformer self-attention.

---

### 4.2 Modern Tokenization Challenges

#### 4.2.1 The Multilingual Problem
- **SentencePiece**: Treats all languages uniformly by tokenizing raw text (including spaces).
    - *Example*: "Hello world" → `["▁Hello", "▁world"]` (where `▁` represents a space).
    - **Benefit**: No need for language-specific tokenization rules.
- **Language-Specific Vocabularies**: Some models (e.g., mBERT) use a shared vocabulary across 104 languages, leading to uneven representation (English gets ~50% of tokens).

#### 4.2.2 Code and Special Tokens
- **Code Tokenization**: Programming languages have strict syntax. Standard BPE can break identifiers unpredictably.
    - *Solution*: **CodeBERT** uses byte-level BPE to handle any UTF-8 character.
    - *Example*: `function_name` might be tokenized as `["function", "_name"]` or `["func", "tion", "_", "name"]`.
- **Special Tokens**: Modern tokenizers reserve tokens for special purposes:
    - `[CLS]`, `[SEP]` (BERT)
    - `<s>`, `</s>`, `<pad>`, `<unk>` (RoBERTa)
    - `<|startoftext|>`, `<|endoftext|>` (GPT)

#### 4.2.3 Tokenization Artifacts
- **The "Fused Island" Problem**: Rare word combinations can be merged into single tokens during BPE training.
    - *Example*: "New York" might become a single token, but "New Jersey" is split. This creates inconsistent representations.
- **Mitigation**: Use **Unigram LM** tokenization (used in SentencePiece), which probabilistically selects tokens rather than greedily merging.

---

### 4.3 Contextual Embeddings: The ELMo Revolution

Before BERT, **ELMo** (Embeddings from Language Models) introduced contextual word representations.

#### The Architecture:
1.  Train a **bi-directional LSTM** language model.
2.  Extract hidden states from all layers.
3.  Create a task-specific weighted sum:
    $$ \text{ELMo}_k = \sum_{j=0}^L s_{kj} h_{j} $$
    Where $s_{kj}$ are learned scalar weights for task $k$.

**Key Difference from Word2Vec**: "Bank" in "river bank" and "bank account" get **different vectors** because the LSTM hidden states depend on the full context.

---

### 4.4 Evaluation Metrics for Embeddings

#### 4.4.1 Intrinsic Evaluation
- **Word Similarity**: Correlate model's cosine similarity with human judgments (e.g., WordSim-353 dataset).
- **Word Analogy**: Solve "Man is to King as Woman is to ___" using vector arithmetic: $v_{king} - v_{man} + v_{woman} \approx v_{queen}$.

#### 4.4.2 Extrinsic Evaluation
- **Downstream Task Performance**: Train a classifier (e.g., sentiment analysis) using the embeddings and measure accuracy.
- **Freezing vs. Fine-tuning**: Static embeddings (Word2Vec) are frozen; contextual embeddings (BERT) are fine-tuned end-to-end.

---

## 5. Implementation Deep Dive: Building a Tokenizer from Scratch

### 5.1 Complete BPE Training Algorithm
```python
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        
    def _get_vocab(self, text):
        """Convert text to word-frequency dict with character splits."""
        vocab = defaultdict(int)
        words = text.split()
        for word in words:
            # Add end-of-word token
            word = word + '</w>'
            # Split into characters
            chars = tuple(word)
            vocab[chars] += 1
        return vocab
    
    def _get_stats(self, vocab):
        """Count frequency of all adjacent pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair, vocab):
        """Merge all occurrences of a pair in the vocabulary."""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            word_str = ' '.join(word)
            new_word = pattern.sub(''.join(pair), word_str)
            new_vocab[tuple(new_word.split())] = vocab[word]
        
        return new_vocab
    
    def fit(self, text):
        """Train the BPE tokenizer."""
        vocab = self._get_vocab(text)
        
        # Initial vocabulary (all unique characters)
        all_chars = set()
        for word in vocab:
            all_chars.update(word)
        self.vocab = {c: i for i, c in enumerate(sorted(all_chars))}
        
        # Perform merges until target vocab size
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            # Add new token to vocabulary
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
        
        return self
    
    def tokenize(self, text, max_length=None):
        """Tokenize a single text."""
        word = text + '</w>'
        chars = tuple(word)
        
        # Apply all merges in order
        for pair in self.merges:
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i+1] == pair[1]:
                    new_chars.append(''.join(pair))
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = tuple(new_chars)
        
        # Convert to IDs
        token_ids = [self.vocab.get(t, self.vocab.get('<unk>', 0)) for t in chars]
        
        if max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids

# Example Usage
text = "low low low lower lower newest newest widest widest"
tokenizer = BPETokenizer(vocab_size=50)
tokenizer.fit(text)

print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Number of merges: {len(tokenizer.merges)}")
print(f"Tokenize 'lowest': {tokenizer.tokenize('lowest')}")
```

---

## 6. Common Pitfalls and Production Considerations

### 6.1 The OOV Catastrophe
- **Problem**: Word2Vec cannot handle words not seen during training.
- **Real-world Impact**: In social media text, ~5-10% of words might be OOV (slang, typos, new terms).
- **Solution**: FastText (subword embeddings) or contextual models (BERT handles OOV via subword tokenization).

### 6.2 Bias in Embeddings
- **Famous Example**: $v_{man} - v_{programmer} \approx v_{woman} - v_{homemaker}$
- **Cause**: Embeddings capture statistical biases present in training data.
- **Mitigation**: 
    - **Hard Debias**: Project embeddings onto a subspace orthogonal to the bias direction.
    - **Soft Debias**: Modify the training objective to penalize biased associations.

### 6.3 Memory Optimization for Large Vocabularies
- **Hierarchical Softmax**: Uses a Huffman tree to reduce softmax computation from $O(V)$ to $O(\log V)$.
- **Negative Sampling**: As discussed, reduces to $O(K)$ where $K \ll V$.

---

## 📚 Recommended Resources
- **Papers**:
    - [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781)
    - [Enriching Word Vectors with Subword Information (FastText)](https://arxiv.org/abs/1607.04606)
    - [Deep Contextualized Word Representations (ELMo)](https://arxiv.org/abs/1802.05365)
    - [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)](https://arxiv.org/abs/1409.0473)
    - [Effective Approaches to Attention-based Neural Machine Translation (Luong Attention)](https://arxiv.org/abs/1508.04025)
- **Books**: "Speech and Language Processing" by Dan Jurafsky (The Bible of NLP).
- **Tools**: 
    - [HuggingFace Tokenizers Library](https://github.com/huggingface/tokenizers)
    - [SentencePiece](https://github.com/google/sentencepiece)
- **Datasets**: 
    - [WordSim-353](https://aclweb.org/aclwiki/WordSim-353) for similarity evaluation
    - [Google Analogy Dataset](https://aclweb.org/aclwiki/Google_analogy_dataset) for analogy testing

---

## 🔬 Research Frontiers (2024-2025)

### 6.4 Multi-modal Tokenization
- **Image-Text Models**: How to tokenize images for joint language-vision models?
    - **CLIP**: Uses separate tokenizers (BPE for text, patches for images).
    - **Flamingo**: Uses Perceiver Resampler to convert image patches into a fixed number of "visual tokens."

### 6.5 Token-Free Models
- **Character-Level Models**: Working directly with characters (e.g., CharBERT) to eliminate tokenization entirely.
- **Byte-Level Models**: Operating on raw bytes (e.g., ByT5) for true language-agnostic processing.

---

**Status:** ✅ Elite Expanded Standard (12/10)
**Next:** Sequence Models (RNNs, LSTMs, BPTT, Advanced Variants)
