# 10.6 Advanced RAG Architectures: Context-Aware Systems

## 🎯 Quick Overview
- **Retrieval Math**: Euclidean Distance vs. Cosine Similarity vs. Dot Product
- **Indexing Internals**: HNSW (Hierarchical Navigable Small World) and FAISS
- **Advanced Retrieval**: Hybrid Search, Re-ranking, and Parent-Document Retrieval
- **Reasoning over Data**: GraphRAG and Multi-hop retrieval
- **Foundation for**: Production AI, Personal Knowledge bases, and Enterprise search

---

## 1. The Mathematics of Similarity

Vector Databases don't use standard SQL indexing. They use **ANN (Approximate Nearest Neighbor)** search based on distance metrics.

| Metric | Formula | Best For |
| :--- | :--- | :--- |
| **Euclidean (L2)** | $\sqrt{\sum (q_i - p_i)^2}$ | Fixed scale embeddings. |
| **Cosine Similarity**| $\frac{A \cdot B}{\|A\| \|B\|}$ | Comparing orientation (ignores magnitude). Standard for NLP. |
| **Dot Product** | $\sum a_i b_i$ | High-performance, takes magnitude into account (used in Recommenders). |

---

## 2. Vector Indexing: HNSW Deep Dive

Standard search is $O(N)$. For 1 billion vectors, this is impossible. **HNSW** is the industry standard for $O(\log N)$ search.

### 2.1 How HNSW Works
1.  **Multi-layer Graph**: Similar to a Skip-List but for graphs.
2.  **Greedy Search**: The search starts at the top layer (sparsest) to find the "neighborhood" and zooms in through lower layers (densest).
3.  **Navigable Small World**: Ensures that any two nodes in the graph can be reached in a small number of steps.

---

## 3. The Production RAG Pipeline

A professional RAG system uses more than just simple retrieval.

### 3.1 Hybrid Search
Combines **Dense Retrieval** (Vectors) with **Sparse Retrieval** (BM25/Keyword).
- **Reciprocal Rank Fusion (RRF)**: A mathematical way to combine results from both lists into a single ranked output.

### 3.2 The Re-ranker (Cross-Encoders)
1.  **Stage 1 (Retriever)**: Fast but "coarse." Finds top 100 candidates using Cosine Similarity.
2.  **Stage 2 (Re-ranker)**: Slow but "precise." A Cross-Encoder model looks at (Query + Document) together to give a 0-1 relevance score.

### 3.3 Parent-Document Retrieval
Instead of retrieving small chunks (which lose context), we:
1.  Search across tiny "Child Chunks" (e.g., 100 tokens).
2.  Return the **full original document** (Parent) to the LLM.

---

## 4. GraphRAG: Contextual Relationship Search

Traditional RAG fails at "summarize the relationship between Person A and Company B." 
- **GraphRAG** builds a **Knowledge Graph** from the documents first.
- It retrieves "Communities" of entities, allowing the LLM to understand high-level themes and non-linear connections.

---

## 💻 Professional Implementation

### 1. Reciprocal Rank Fusion (RRF) Logic
```python
def rrf_score(dense_rank, sparse_rank, k=60):
    """Combine dense and sparse search rankings."""
    return (1 / (k + dense_rank)) + (1 / (k + sparse_rank))

# Example usage:
# A doc ranked 1st in dense and 10th in sparse
score = rrf_score(1, 10)
```

### 2. Custom Re-ranking with Sentence-Transformers
```python
from sentence_transformers import CrossEncoder

# 1. Load a high-precision cross-encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Score pairs
query = "How do I optimize a KV-cache?"
docs = ["Use PagedAttention.", "KV-cache grows linearly.", "Buy more GPUs."]

scores = model.predict([(query, d) for d in docs])
# Higher scores mean more relevance
ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
```

---

## 📊 Summary Comparison

| Technique | Cost | Latency | Accuracy |
| :--- | :--- | :--- | :--- |
| **Simple RAG** | Low | Low | Moderate |
| **Re-ranking** | Moderate | Medium | **High** |
| **Hybrid Search**| Moderate | Low | **High** |
| **GraphRAG** | High | High | **Exceptional (Global)**|

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Multi-hop RAG** | Medical diagnosis where info is spread across different research papers. |
| **Agentic RAG** | An agent that decides *which* tool or database to search based on the query. |
| **Query Expansion** | Using an LLM to rewrite a short user query into a 3-paragraph search term. |
| **Semantic Cache** | Caching LLM responses based on query vector similarity to save costs. |

---

## ❓ Quick Check Questions

1. Why is Cosine Similarity usually better than Euclidean Distance for NLP embeddings?
2. What is the "Lost in the Middle" problem in RAG?
3. How does a Cross-Encoder differ from a Bi-Encoder?
4. Explain the "Small-to-Big" retrieval strategy.
5. In HNSW, why do we use multiple layers instead of just one flat graph?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Cosine Similarity** focuses on the *direction* of the vectors rather than their magnitude. In text, a 100-page document and a 1-page summary of that document will have vectors pointing in the same direction, even if the magnitudes are vastly different.
2. The **Lost in the Middle** problem occurs when an LLM performs well at using information at the very beginning or very end of a long prompt but ignores or "forgets" information tucked away in the middle.
3. A **Bi-Encoder** encodes the query and document separately into vectors (fast for search). A **Cross-Encoder** processes the query and document together, allowing for complex interaction between words (slow but extremely accurate for ranking).
4. **Small-to-Big** involves splitting documents into tiny chunks for searching (where the signal is precise) but returning larger, surrounding blocks of text to the LLM so it has the necessary context to form a coherent answer.
5. Multiple layers act like a "map." The top layers allow for huge "jumps" across the vector space to find the general area, while lower layers allow for fine-tuning the search to find the exact nearest neighbors, ensuring the search is $O(\log N)$.

</details>

---

## 📚 Recommended Resources
- **Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **Tool**: [FAISS (Facebook AI Similarity Search) Documentation](https://github.com/facebookresearch/faiss).
- **Blog**: [Pinecone's Guide to HNSW](https://www.pinecone.io/learn/series/vector-databases/hnsw/).

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** Fine-tuning Techniques (LoRA, QLoRA, DPO math)
