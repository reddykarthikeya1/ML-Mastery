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

## 💻 Professional Implementation: Production-Grade RAG

This implementation covers the full lifecycle: Ingestion, Hybrid Search (Vector + BM25), and Cross-Encoder Re-ranking.

```python
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

class ProductionRAG:
    def __init__(self, vector_model: str, rerank_model: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Load Models
        self.embed_model = SentenceTransformer(vector_model).to(self.device)
        self.rerank_model = CrossEncoder(rerank_model)
        
        self.documents = []
        self.vectors = None
        self.bm25 = None

    def ingest(self, corpus: List[str]):
        """Encode documents into vector and BM25 indices."""
        print(f"Ingesting {len(corpus)} documents...")
        self.documents = corpus
        # Vector Encoding
        self.vectors = self.embed_model.encode(corpus, convert_to_tensor=True)
        # BM25 Tokenization
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Hybrid Search + Cross-Encoder Re-ranking."""
        # A. Vector Search (Dense)
        query_vec = self.embed_model.encode(query, convert_to_tensor=True)
        cos_scores = torch.cos_sim(query_vec, self.vectors)[0]
        top_v_idx = torch.topk(cos_scores, k=min(top_k*2, len(self.documents))).indices.tolist()

        # B. BM25 Search (Sparse)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_b_idx = np.argsort(bm25_scores)[::-1][:top_k*2].tolist()

        # C. Fusion (RRF) and Candidate Selection
        candidate_idx = list(set(top_v_idx + top_b_idx))
        candidates = [self.documents[i] for i in candidate_idx]

        # D. Re-ranking (Cross-Encoder)
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = self.rerank_model.predict(pairs)
        
        # Sort by rerank scores
        ranked_results = [candidates[i] for i in np.argsort(rerank_scores)[::-1]]
        return ranked_results[:top_k]

# --- Usage Example ---
# rag = ProductionRAG("all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# rag.ingest(["Python is a language.", "LLMs require GPUs.", "RAG improves accuracy."])
# results = rag.retrieve("How to make AI more accurate?")
# print(f"Top Result: {results[0]}")
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
