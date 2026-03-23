# 10.6 RAG Architectures (Retrieval-Augmented Generation)

## 🎯 Quick Overview
- **RAG**: Giving LLMs access to external, real-time data without retraining
- **Vector Databases**: Storing and searching data based on semantic meaning
- **Embedding Models**: Converting text into high-dimensional vectors
- **Retrieve-Read Pattern**: The core workflow of modern AI applications
- **Foundation for**: Enterprise AI, personal assistants, and real-time knowledge bots

---

## 1. Why RAG?

LLMs have two major weaknesses:
1. **Knowledge Cutoff**: They only know what they were trained on (e.g., GPT-4 doesn't know about news from yesterday).
2. **Hallucination**: They often confidently state false information.

**RAG** solves this by letting the model "look up" facts in a library before answering.

---

## 2. The RAG Pipeline

### 2.1 Ingestion (Data Preparation)
1. **Chunking**: Breaking large documents (PDFs, Wikis) into smaller pieces (e.g., 500 characters).
2. **Embedding**: Converting each chunk into a vector (array of numbers) using an embedding model (like `text-embedding-3-small`).
3. **Indexing**: Storing vectors in a **Vector Database** (e.g., Pinecone, Milvus, Chroma).

### 2.2 Retrieval (Finding Information)
When a user asks a question:
1. The question is converted into a vector.
2. The Vector DB performs a **Similarity Search** (e.g., Cosine Similarity) to find the top $K$ most relevant chunks.

### 2.3 Generation (Augmenting)
The retrieved chunks are stuffed into the prompt:
*"Use the following context to answer the question: [Context Chunks]. Question: [User Question]"*

---

## 3. Advanced RAG Techniques

### 3.1 Re-ranking
After initial retrieval, a smaller, more accurate model (Cross-Encoder) re-scores the chunks to ensure the most relevant ones are at the top.

### 3.2 Hybrid Search
Combining **Keyword Search** (BM25) with **Semantic Search** (Vector) to get the best of both worlds (exact names + conceptual meaning).

### 3.3 Query Expansion / HyDE
Rewriting the user's query to make it more descriptive before searching the database.

---

## 💻 Python Code Examples

### 1. Minimal RAG Logic (Conceptual)
```python
# 1. User Query
query = "What is the revenue of Company X in 2023?"

# 2. Retrieval
query_vector = embed_model.encode(query)
relevant_chunks = vector_db.search(query_vector, top_k=3)

# 3. Augmentation
context = "\n".join(relevant_chunks)
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

# 4. Generation
response = llm.generate(prompt)
print(response)
```

### 2. Using LangChain for RAG
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load DB and Model
db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=db.as_retriever()
)

# Ask a question
print(qa_chain.run("What are the core pillars of the ML-Mastery project?"))
```

---

## 📊 Summary Table

| Component | Purpose | Popular Tools |
|-----------|---------|---------------|
| **Embedding Model** | Convert text to math | OpenAI, HuggingFace, Cohere |
| **Vector DB** | Fast similarity search | Pinecone, Chroma, Milvus, Qdrant |
| **Orchestrator** | Connect the steps | LangChain, LlamaIndex, Haystack |
| **Generator** | Write the final answer | GPT-4, Claude 3, Llama 3 |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Simple RAG | Internal company wikis |
| Multimodal RAG | Searching across images and text |
| GraphRAG | Analyzing complex relationships in legal data |
| Real-time RAG | Stock market analysis bots |

---

## ❓ Quick Check Questions

1. What is the "Hallucination" problem, and how does RAG help fix it?
2. What is "Chunking," and why is the chunk size important?
3. How does Cosine Similarity help in retrieval?
4. What is the difference between Sparse (Keyword) and Dense (Vector) retrieval?
5. Why is a "Re-ranker" used in advanced RAG pipelines?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Hallucination** is when an LLM generates plausible-sounding but factually incorrect information. RAG fixes this by providing the model with **ground truth context** from a trusted source, forcing the model to cite its sources and base its answer only on provided facts.
2. **Chunking** is breaking down text into smaller, meaningful segments. If chunks are too small, they lose context. If they are too large, they may contain irrelevant information and dilute the similarity score.
3. **Cosine Similarity** measures the angle between two vectors. In RAG, vectors that point in a similar direction in the embedding space represent text with similar semantic meaning, allowing the model to find relevant context even if the exact words don't match.
4. **Sparse Retrieval** (like BM25) looks for exact word matches (good for names, part numbers). **Dense Retrieval** (Vector) looks for conceptual meaning (good for "How-to" questions or synonyms).
5. Initial vector search is fast but can be imprecise. A **Re-ranker** takes the top ~20 results and uses a much more intensive model to calculate the exact relevance of each, ensuring only the highest-quality information reaches the LLM.

</details>

---

**Status:** ✅ Complete
**Next:** Fine-tuning Techniques (LoRA, QLoRA, DPO)
