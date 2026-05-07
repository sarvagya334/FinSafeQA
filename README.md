# FinSafeQA — Country-Aware Financial RAG System

FinSafeQA is a hybrid Retrieval-Augmented Generation (RAG) framework designed for explainable financial question answering over stable investment instruments such as:

- Fixed Deposits (FDs)
- Sovereign Gold Bonds (SGBs)
- Government Securities
- Retirement Schemes
- Low-risk Savings Products

The system combines:

- Adaptive financial chunking
- Hybrid FAISS + BM25 retrieval
- Country-aware indexing
- Multi-hop reasoning
- Explainable prompting
- Validation-driven response generation

---

## Features

- Hybrid semantic + lexical retrieval
- Country-aware document routing
- Financial table-aware chunking
- Multi-hop reasoning pipeline
- Explainable structured outputs
- NVIDIA NIM + Ollama support
- FAISS vector indexing
- BM25 keyword retrieval
- Chunk caching for fast ingestion

---

## Project Structure

```bash
src/
│
├── bm25_index.py
├── chunk_cache.py
├── chunking.py
├── config.py
├── country_detect.py
├── country_indexes.py
├── embeddings.py
├── hybrid_retrieval.py
├── ingestion.py
├── multi_prompt_rag.py
├── prompt.py
├── rag.py
├── retrieval.py
└── validators.py
```

---

## Installation

### 1. Clone Repository

```bash
git clone <your_repo_url>
cd FinSafeRAG
```

### 2. Create Virtual Environment

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Dependencies** (example `requirements.txt`):

```
sentence-transformers
faiss-cpu
rank-bm25
numpy
requests
langchain-core
torch
python-dotenv
```

**For Apple Silicon:**

```bash
pip install torch torchvision torchaudio
```

---

## Data Directory Structure

Create the following structure:

```
data/
│
├── raw/
├── processed/
├── faiss/
```

### Add Financial Documents

Place markdown financial documents inside:

```
data/processed/
```

**Example:**

```
data/processed/
├── india_fd.md
├── singapore_bonds.md
├── india_singapore_comparison.md
```

---

## Environment Variables

Create a `.env` file:

```bash
NVIDIA_API_KEY=your_nvidia_api_key
LLM_BACKEND=nvidia
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

### Ollama Setup (Optional)

**Install Ollama:**

```bash
brew install ollama
```

**Pull model:**

```bash
ollama pull llama3.2
```

**Start server:**

```bash
ollama serve
```

Then set:

```bash
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2
```

---

## Running the Pipeline

### 1. Ingest Documents

**Example:**

```python
from src.ingestion import ingest_markdown

docs = ingest_markdown("data/processed/india_fd.md")
```

This performs:

- Metadata inference
- Adaptive chunking
- Topic extraction
- Chunk caching

### 2. Load Embedding Model

```python
from src.embeddings import load_embedding_model

embed_model = load_embedding_model()
```

### 3. Build Country Indexes

```python
from src.country_indexes import build_country_indexes

build_country_indexes(
    documents=docs,
    embed_model=embed_model,
    out_dir="data/faiss"
)
```

This creates:

- FAISS indexes
- BM25 indexes
- Document maps

per country.

### 4. Load Indexes

```python
from src.country_indexes import load_country_indexes

country_indexes = load_country_indexes("data/faiss")
```

### 5. Initialize LLM

**NVIDIA NIM:**

```python
from src.rag import init_llm

llm = init_llm(
    api_key="YOUR_API_KEY",
    backend="nvidia"
)
```

**Ollama:**

```python
llm = init_llm(backend="ollama")
```

### 6. Run RAG Query

```python
from src.rag import rag_query

answer = rag_query(
    query="What are FD rates in India?",
    country_indexes=country_indexes,
    embed_model=embed_model,
    llm=llm
)

print(answer)
```

---

## Multi-Hop Reasoning

For complex queries:

```python
from src.multi_prompt_rag import multi_prompt_rag

result = multi_prompt_rag(
    question="Compare SGB vs Singapore Bonds",
    retrieve_fn=retrieve_function,
    llm=llm
)

print(result)
```

---

## Example Queries

- What are FD rates in India?
- Compare SGB and Singapore Government Securities.
- What are tax benefits of RBI bonds?
- Summarize all documents.
- What are low-risk retirement investments in Singapore?

---

## Retrieval Architecture

FinSafeRAG uses:

### Semantic Retrieval
- Sentence Transformers
- FAISS cosine similarity

### Lexical Retrieval
- BM25 keyword ranking

### Hybrid Fusion
- Weighted semantic + lexical scoring

---

## Country-Aware Retrieval

The system automatically detects:

- India
- Singapore
- USA
- UK
- UAE

from user queries and routes retrieval to country-specific indexes.

---

## Explainable Output Format

Generated responses include:

- Summary
- Step-by-step reasoning
- Calculations
- Assumptions
- Sources
- Confidence level

---

## Validation Layer

Outputs are validated for:

- Citation correctness
- Structure completeness
- Hallucinated references

before final delivery.

---

## Running Full Example

**Example `app.py`:**

```python
from src.ingestion import ingest_markdown
from src.embeddings import load_embedding_model
from src.country_indexes import build_country_indexes, load_country_indexes
from src.rag import init_llm, rag_query

# ingest
docs = ingest_markdown("data/processed/india_fd.md")

# embeddings
embed_model = load_embedding_model()

# build indexes
build_country_indexes(
    documents=docs,
    embed_model=embed_model,
    out_dir="data/faiss"
)

# load indexes
country_indexes = load_country_indexes("data/faiss")

# llm
llm = init_llm(backend="ollama")

# query
response = rag_query(
    query="What are FD rates in India?",
    country_indexes=country_indexes,
    embed_model=embed_model,
    llm=llm
)

print(response)
```

**Run:**

```bash
python app.py
```

---

## Methodology Overview

FinSafeRAG consists of five major stages:

1. Financial document ingestion
2. Adaptive financial chunking
3. Country-aware index construction
4. Hybrid retrieval and governance
5. Explainable response generation

### Financial Document Ingestion

Financial markdown documents are enriched with:

- Country
- Regulator
- Currency
- Financial topics
- Asset class metadata

This metadata is later used for governed retrieval and explainable reasoning.

### Adaptive Financial Chunking

The chunking pipeline preserves:

- Page structure
- Headings
- Financial tables
- Overlapping context windows

Large tables are chunked row-wise while retaining headers to preserve numerical consistency.

### Country-Aware Indexing

Instead of using one global vector database, the system creates:

- Country-specific FAISS indexes
- Country-specific BM25 indexes
- Document maps

This prevents cross-country hallucinations and improves regulatory consistency.

### Hybrid Retrieval

The retrieval pipeline combines:

- Semantic FAISS retrieval
- Lexical BM25 retrieval

using weighted score fusion.

The system additionally:

- Detects country references
- Filters ambiguous financial terms
- Enforces jurisdiction-aware retrieval

### Multi-Hop Reasoning

Complex financial queries are decomposed into:

- Smaller sub-questions
- Independent retrieval steps
- Aggregated reasoning outputs

This improves comparative financial analysis and long-form reasoning quality.

### Explainable Prompting

Generated outputs follow a structured format:

- Summary
- Step-by-step reasoning
- Calculations
- Assumptions
- Citations
- Confidence estimation

### Validation Layer

Responses are validated for:

- Citation correctness
- Structural completeness
- Hallucinated references

before final delivery.

---

## Research Contributions

FinSafeRAG introduces:

- Country-governed financial retrieval
- Adaptive table-aware chunking
- Explainable financial prompting
- Hybrid semantic-lexical retrieval
- Validation-driven financial QA
