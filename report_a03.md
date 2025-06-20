---
title: rag_and_reasoning_frameworks_tutorial
---

## RAG Fundamentals

### RAG Architecture Overview

<details open>
<summary>Introduction to Retrieval-Augmented Generation (RAG)</summary>

---

* RAG combines information retrieval with text generation using large language models (LLMs).
* Consists of two main stages: **retrieval** and **generation**.
* Retrieval component fetches relevant documents based on a query using vector similarity.
* Generation component synthesizes responses using both retrieved data and language modeling.
* Enhances factual grounding and reduces hallucinations.
* Commonly used in document-based chatbots, QA systems, and knowledge-based assistants.

---

#### Key Components

* **Retriever**: Vector database + embedding model (e.g., OpenAI, HuggingFace).
* **Chunking**: Split documents into semantically coherent chunks for embedding.
* **Embedder**: Converts chunks into vector representations.
* **Vector DB**: Stores embeddings and supports similarity search (e.g., ChromaDB, FAISS).
* **LLM Generator**: Generates final answer based on retrieved context.

---

#### Retrieval Pipeline

* User submits question → embedded → search vector DB → retrieve top-k documents → feed to LLM.
* May use re-ranking or hybrid search (dense + sparse).

  ```mermaid
  flowchart TD
      A["User submits a question"] --> B["Embed the query"]
      B --> C["Search in Vector DB (Top-k)"]
      C --> D["Retrieve matching document chunks"]
      D --> E["Send context + query to LLM"]
      E --> F["LLM generates grounded response"]
      F --> G["Return final answer to user"]
  ```
---
#### Benefits

* Scalable for large corpora.
* Reduces need for model retraining.
* Adaptable to domain-specific use cases with minimal resources.

---

</details>

### Document Chunking Strategies

<details open>
<summary>Strategies to prepare documents for semantic search</summary>

---

* **Fixed-size chunking**: Uniform length, easy but may break semantic units.
* **Semantic chunking**: Based on sentence/paragraph boundaries or topic shifts.
* **Sliding window**: Overlapping chunks to preserve context.
* Trade-off between chunk size and retrieval quality.
* Implemented using tools like `langchain.text_splitter`.

---

#### Chunk Configuration Considerations

* Chunk size: 200–800 tokens is common.
* Overlap: 10–20% improves coherence.
* Clean pre-processing (removing headers/footers, OCR artifacts) is essential.

---

</details>

---

## Reasoning Frameworks

---

### Overview of Reasoning Techniques

<details open>
<summary>Advanced techniques for logical and stepwise reasoning</summary>

---

* RAG can be enhanced with structured reasoning frameworks to answer multi-hop or complex queries.
* Chain-of-Thought (CoT): Breaks reasoning into intermediate steps.
* Tree-of-Thought (ToT): Explores multiple reasoning paths in a tree format.
* ReAct: Combines reasoning with tool usage and reflection.
* Frameworks improve interpretability, reduce hallucination.

---

#### CoT vs ToT

* CoT: Linear thought process, good for math/logic questions.
* ToT: Branching exploration, useful for open-ended or ambiguous tasks.

---

#### Example Techniques

* ReAct pattern: LLM decides → acts (calls retriever/tool) → reflects → repeats.
* Self-consistency: Samples multiple reasoning paths and selects the best one.

---

#### Use Cases

* Legal/document analysis.
* Multi-hop QA (e.g., Wikipedia or legal chain references).
* Code understanding.

---

</details>

---

## Tool Comparison

---

### LangChain vs LlamaIndex vs Haystack

<details open>
<summary>Comparison of common RAG frameworks and tools</summary>

---

#### LangChain

* Modular, supports various chains and agents.
* Integrates with OpenAI, HuggingFace, ChromaDB, Pinecone.
* Extensive ecosystem with prompt templates, tools, chains.

---

#### LlamaIndex

* Optimized for document indexing.
* Offers graph-based traversal, vector + keyword hybrid search.
* Simpler abstraction layer, tightly integrated with storage.

---

#### Haystack

* Enterprise-focused, supports pipelines and evaluation.
* Can integrate Elasticsearch, OpenSearch, or vector DBs.
* Flexible for production pipelines, includes evaluation tools.

---

#### Comparison Table

| Feature    | LangChain      | LlamaIndex     | Haystack     |
| ---------- | -------------- | -------------- | ------------ |
| Ecosystem  | Strong         | Medium         | Strong       |
| Focus      | Prompt + Agent | Index + Search | Pipeline Dev |
| Vector DBs | Yes            | Yes            | Yes          |
| Evaluation | Manual         | Basic          | Integrated   |

---

</details>

---

## Implementation Guide

---

### Step-by-Step RAG System

<details open>
<summary>Instructions and code examples for building a RAG system</summary>

---

#### Setup

* Choose LLM provider: OpenAI, Ollama, Cohere.
* Choose vector DB: ChromaDB, FAISS.
* Install libraries: `langchain`, `chromadb`, `openai`, `tiktoken`, `PyMuPDF`.

---

#### Document Ingestion

* Parse PDFs using `pdfminer.six`, `PyMuPDF`.
* Clean and split text into chunks.
* Embed chunks using `OpenAIEmbeddings`.
* Store in ChromaDB.

---

#### Query Flow

* Receive user query → embed → similarity search in DB → retrieve top-k chunks.
* Use `langchain.RetrievalQA` with `OpenAI` or `ChatOpenAI` to synthesize answer.
* Return grounded and contextual response.

---

#### Code Snippet (Python)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

embedding = OpenAIEmbeddings()
db = Chroma(persist_directory='db', embedding_function=embedding)
retriever = db.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
response = qa.run('What are the key findings in the document?')
```

---

</details>

---

## Optimization Techniques

---

### Performance, Accuracy, and Cost

<details open>
<summary>Strategies to optimize RAG-based systems</summary>

---

* Use **batch embedding** for ingestion efficiency.
* **Reduce chunk size** for faster retrieval, but balance context.
* **Filter top-k results** using similarity threshold or re-ranking.
* Cache recent query responses to reduce API calls.
* Choose cost-efficient models: GPT-3.5-turbo vs GPT-4.
* Monitor latency and track usage for cost control.
* Add logging and fallback when retrieval fails.

---

</details>

---

## Case Studies

---

### Real-World Example: GenAI PDF Bot

<details open>
<summary>Case study based on implemented project</summary>

---

* Built RAG chatbot to interact with user-uploaded PDFs.
* Used ChromaDB for embedding storage and LangChain for query interface.
* Documents chunked using `RecursiveCharacterTextSplitter`.
* Integrated Streamlit UI + Docker for deployment.
* Added fallback for OCR using `pdf2image` + Tesseract.
* Modular pipeline allows switching LLM providers (OpenAI, Ollama).
* Production deployed on VPS with Nginx proxy and SSL.

---

</details>

---
## Terminology
---

### Key Terms in RAG and Reasoning

<details open>
<summary>Glossary of terms used in Task A03</summary>

---

- **RAG (Retrieval-Augmented Generation)**: Architecture that enhances LLM output by retrieving external knowledge.
- **Vector DB**: Specialized database that stores high-dimensional vector embeddings for similarity search.
- **Chunking**: Splitting large documents into smaller semantic units for indexing and retrieval.
- **Embedding**: Numerical vector representation of data (text, image, etc.) generated by a model.
- **Retriever**: Component that searches the vector DB and returns relevant chunks.
- **LLM (Large Language Model)**: A model that generates human-like text, such as GPT-3.5, GPT-4.
- **Chain-of-Thought (CoT)**: A reasoning strategy where steps are broken down sequentially.
- **Tree-of-Thought (ToT)**: Reasoning model that explores multiple reasoning branches.
- **ReAct**: Framework combining reasoning, action execution, and reflection.
- **Self-consistency**: Technique that generates multiple reasoning paths and selects the most consistent one.

---
</details>

