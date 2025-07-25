---
title: vector_database_tutorial
---

## Vector Database Fundamentals

### What is a Vector Database?

<details open>
<summary>Definition and core concepts</summary>

---

* A **vector database** stores and indexes high-dimensional vectors (embeddings).
* Embeddings are numerical representations of data (text, image, etc.) in vector space.
* Core use case: **similarity search** – finding nearest vectors to a query vector.
* Used in GenAI, semantic search, recommendation systems, RAG systems.
* Vector DBs support operations like insert, update, delete, and nearest neighbor search.

---

#### Typical Workflow

* Convert raw data (e.g., text) to embeddings using a model.
* Store vectors in the vector DB.
* For a new query, embed the query and retrieve similar vectors.
* Combine retrieved results with downstream applications (e.g., QA, summarization).

---

</details>

### Key Concepts in Vector Search

<details open>
<summary>Technical foundations of vector-based retrieval</summary>

---

* **Embedding model**: Converts data into dense vector format (e.g., OpenAI, SBERT).
* **Vector similarity metrics**:

  * Cosine similarity
  * Euclidean distance
  * Dot product
* **Indexing method**:

  * Flat (brute-force)
  * Approximate Nearest Neighbor (ANN) → faster, lower accuracy
  * Examples: HNSW, IVF, PQ

---

#### Trade-offs

* ANN improves speed but may miss exact matches.
* Index size, recall, latency must be balanced for production.

---

</details>

---

## Vector Database Comparison

---

### Popular Vector DBs Overview

<details open>
<summary>Comparison of commonly used vector database tools</summary>

---

| Feature          | FAISS     | ChromaDB         | Weaviate        | Pinecone        |
| ---------------- | --------- | ---------------- | --------------- | --------------- |
| Backend          | Local     | Local            | Cloud / Local   | Fully Managed   |
| ANN Support      | Yes       | Yes              | Yes             | Yes             |
| Metadata         | Limited   | Moderate         | Rich (GraphQL)  | Rich            |
| Integration      | Manual    | LangChain-native | LangChain, REST | LangChain, REST |
| Indexing Options | IVF, HNSW | HNSW             | HNSW, Flat      | Proprietary     |
| Persistence      | Manual    | Built-in         | Built-in        | Built-in        |
| Best for         | Research  | Prototyping      | Scalable Apps   | Production      |

---

#### Notes

* FAISS is flexible but requires more manual control.
* ChromaDB is lightweight and fast for local use.
* Weaviate offers advanced features (hybrid search, GraphQL).
* Pinecone focuses on simplicity and scalability.

---

</details>

---

## Implementation with ChromaDB

---

### ChromaDB Tutorial

<details open>
<summary>Step-by-step guide to using ChromaDB with LangChain</summary>

---

#### Installation

* Install dependencies:

  ```bash
  pip install chromadb langchain openai tiktoken
  ```

---

#### Basic Workflow

* Initialize ChromaDB:
  ```python
  from langchain_community.vectorstores import Chroma
  from langchain_community.embeddings import OpenAIEmbeddings

  embedding_function = OpenAIEmbeddings()
  db = Chroma(embedding_function=embedding_function)
  ```

* Add documents:

  ```python
  texts = ["Vector DBs enable similarity search", "RAG uses embeddings"]
  db.add_texts(texts)
  ```

* Query:

  ```python
  results = db.similarity_search("What is a vector DB?", k=2)
  print(results)
  ```

---

#### Notes

* Supports persistence via `persist_directory`.
* Easily integrated into RAG pipeline using LangChain's retriever pattern.
* Can be combined with OpenAI, Ollama, or local embedding models.

---

</details>

---

## Best Practices

---

### Optimization Strategies

<details open>
<summary>Performance and reliability considerations</summary>

---

* Use batch insertion for large corpora.
* Tune chunk size and overlap to balance recall and speed.
* Regularly clean and rebuild indices for consistency.
* Persist vector stores for reuse between sessions.
* Monitor latency and retrieval performance.
* Secure access when running persistent DBs (e.g., ChromaDB server).

---

</details>

---

## Use Cases

---

### When to Use Vector DBs

<details open>
<summary>Common applications and scenarios</summary>

---

* **RAG systems**: Retrieve context for LLMs from long documents.
* **Semantic search**: Find similar texts, documents, or FAQs.
* **Recommendation engines**: Recommend content/items based on similarity.
* **Document deduplication**: Identify near-duplicate content.
* **Clustering & anomaly detection**: Based on vector distributions.

---

</details>

---
## Terminology
---

### Key Terms in Vector Databases

<details open>
<summary>Glossary of terms used in Task B01</summary>

---

- **Vector**: A list of numerical values representing features of data (e.g., text, image).
- **Embedding**: Process of converting data into vector representations using a model.
- **Vector Database**: A system that stores and retrieves vector embeddings based on similarity.
- **Similarity Search**: Finding vectors that are closest to a query vector in vector space.
- **ANN (Approximate Nearest Neighbor)**: Technique for fast, approximate vector search.
- **HNSW (Hierarchical Navigable Small World)**: A popular ANN algorithm for fast search.
- **FAISS**: Facebook AI Similarity Search – a library for efficient similarity search.
- **ChromaDB**: Lightweight, open-source vector DB optimized for LLM and local use.
- **Weaviate**: Scalable vector database with hybrid search and REST/GraphQL APIs.
- **Pinecone**: Fully-managed vector DB platform for production-scale GenAI applications.

---
</details>