# 10.1 NLP Fundamentals

## 🎯 Quick Overview
- **Text Preprocessing**: Cleaning and normalizing raw text
- **Text Representation**: Converting text to numerical vectors
- **Word Embeddings**: Capturing semantic meaning (Word2Vec, GloVe)
- **Foundation for**: All downstream NLP tasks (Classification, LLMs, RAG)

---

## 1. Text Preprocessing

Raw text is noisy. Preprocessing converts it into a format that machine learning models can understand.

### 1.1 Tokenization
Breaking text into smaller units (tokens).
- **Word Tokenization**: "I love AI" → ["I", "love", "AI"]
- **Sentence Tokenization**: Breaking paragraphs into sentences.
- **Subword Tokenization (Modern)**: Used by LLMs (e.g., Byte-Pair Encoding). "playing" → ["play", "##ing"]

### 1.2 Normalization
- **Lowercasing**: Converting all text to lowercase.
- **Stopword Removal**: Removing common words like "the", "is", "at" which carry little semantic value.
- **Stemming**: Crude heuristic that chops off ends of words (e.g., "running" → "run").
- **Lemmatization**: Using a vocabulary and morphological analysis to return the dictionary base form (e.g., "better" → "good").

### 1.3 Linguistic Tagging
- **POS Tagging**: Identifying Parts of Speech (Noun, Verb, Adjective).
- **NER (Named Entity Recognition)**: Identifying entities like "Apple" (Organization) or "London" (Location).

---

## 2. Text Representation (Traditional)

Models need numbers, not strings.

### 2.1 One-Hot Encoding
Each word is represented as a binary vector with one '1' and many '0's.
- **Problem**: Vectors are sparse and huge (vocabulary size). No semantic relationship (e.g., "king" and "queen" are as different as "king" and "apple").

### 2.2 Bag of Words (BoW)
Counts occurrences of words in a document.
- **Problem**: Loses word order and context.

### 2.3 TF-IDF (Term Frequency-Inverse Document Frequency)
Weighting scheme that highlights words that are frequent in a specific document but rare across the entire corpus.
- **Formula**: $TF-IDF(t, d) = TF(t, d) \times IDF(t)$
- **Key Idea**: "Unique" words get higher scores.

---

## 3. Word Embeddings (Semantic)

Dense, low-dimensional vectors where similar words are close in vector space.

### 3.1 Word2Vec (Google)
Uses a shallow neural network to learn embeddings.
- **CBOW (Continuous Bag of Words)**: Predicts the target word from surrounding context.
- **Skip-gram**: Predicts the surrounding context from a target word.

### 3.2 GloVe (Global Vectors - Stanford)
Based on matrix factorization of the global word-word co-occurrence matrix.

### 3.3 FastText (Facebook)
Similar to Word2Vec but represents words as a **Bag of n-grams**. 
- **Big Advantage**: Can handle **Out-of-Vocabulary (OOV)** words by using subword information.

---

## 💻 Python Code Examples

### 1. Preprocessing with NLTK and Spacy
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Basic NLTK Preprocessing
def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return cleaned

# Advanced Spacy NER
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
```

### 2. TF-IDF Implementation
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 3. Using Pre-trained Embeddings (Gensim)
```python
import gensim.downloader as api

# Load pre-trained Word2Vec (GloVe)
model = api.load("glove-wiki-gigaword-100")

# Vector arithmetic: king - man + woman = ?
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(result) # Output: [('queen', 0.7699...)]
```

---

## 📊 Summary Table

| Method | Type | Context Aware? | Out-of-Vocabulary? | Use Case |
|--------|------|----------------|--------------------|----------|
| One-Hot | Sparse | No | No | Tiny vocabularies |
| BoW | Count | No | No | Simple classification |
| TF-IDF | Weighted | No | No | Search, Information Retrieval |
| Word2Vec | Dense | No (Static) | No | Semantic similarity |
| FastText | Dense | No (Static) | **Yes** | Aggressive morphology |
| BERT | Dense | **Yes** | Yes | SOTA NLP tasks |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Lemmatization | Search engine indexing |
| NER | Information extraction from invoices |
| TF-IDF | Document ranking in search |
| Word2Vec | Recommendation systems (item2vec) |

---

## ❓ Quick Check Questions

1. What is the difference between Stemming and Lemmatization?
2. Why is TF-IDF often better than simple Bag of Words?
3. How does Skip-gram differ from CBOW in Word2Vec?
4. What is the primary advantage of FastText over Word2Vec?
5. What does the vector arithmetic "king - man + woman = queen" demonstrate about word embeddings?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Stemming** is a crude heuristic that chops off prefixes/suffixes (can result in non-words like "univers"). **Lemmatization** uses a dictionary to find the actual root word ("better" → "good").
2. **TF-IDF** penalizes very common words (like "the", "is") that appear in almost all documents, allowing unique and informative words to have higher weight.
3. **CBOW** predicts a missing word based on its context (surrounding words). **Skip-gram** takes a single word and tries to predict the words that surround it. Skip-gram is generally better for rare words.
4. **FastText** treats words as a bag of character n-grams. This allows it to generate embeddings for words it hasn't seen before (OOV) by looking at their sub-parts.
5. It demonstrates that word embeddings capture **linear semantic relationships**. The relative positions of concepts (gender, royalty) are preserved as consistent directions in the vector space.

</details>

---

**Status:** ✅ Complete
**Next:** Sequence Models (RNNs, LSTMs, Attention)
