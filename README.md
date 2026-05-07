# FinSafeQA — Benchmark Dataset and Evaluation Framework for Stable Financial Assets

FinSafeQA is a benchmark framework for evaluating Retrieval-Augmented Generation (RAG) systems on stable financial asset reasoning tasks across India and Singapore.

The benchmark focuses on low-risk and government-backed investment instruments such as:

- Fixed Deposits (FDs)
- Public Provident Fund (PPF)
- Sovereign Gold Bonds (SGBs)
- Treasury Bills
- Singapore Government Securities (SGS)
- CPF
- Savings Schemes
- Government Bonds
- Taxation Rules
- Retirement Products

Unlike traditional finance QA datasets focused on equities or trading, FinSafeQA evaluates:

- Stable asset reasoning
- Regulatory understanding
- Country-aware retrieval
- Financial calculations
- Temporal reasoning
- Hallucination robustness
- Explainability
- Multi-hop financial reasoning

---

## Repository Structure

```bash
README.md
app.py
output.md
requirements.txt
src/
data/
Test/
```

### Source Code Structure

```bash
src/
│
├── benchmarks/
├── bm25_index.py
├── chunk_cache.py
├── chunking.py
├── config.py
├── convert_docling.py
├── country_detect.py
├── country_indexes.py
├── embeddings.py
├── evaluate.py
├── hybrid_retrieval.py
├── ingestion.py
├── multi_prompt_rag.py
├── prompt.py
├── rag.py
├── retrieval.py
└── validators.py
```

---

## Dataset Structure

The dataset is divided into:

- Raw financial documents
- Processed markdown files
- FAISS indexes
- BM25 indexes
- Evaluation benchmarks

### Raw Financial Data

```
data/raw/
│
├── india_stable_assets/
├── singapore_stable_assets/
└── macro_and_fx/
```

The raw corpus contains:

- PDFs
- CSV files
- Financial tables
- Tax rules
- Government schemes
- Macroeconomic indicators

covering stable financial assets from India and Singapore.

### Processed Financial Corpus

```
data/processed/
```

The processed corpus contains markdown representations of:

- Fixed deposits
- Treasury bills
- Retirement schemes
- Sovereign bonds
- Tax regulations
- Macroeconomic indicators
- Inflation datasets
- Currency data

**Examples:**

- `India_Stable_Assets__FD.md`
- `India_Stable_Assets__PPF.md`
- `India_Stable_Assets__SGB.md`
- `Singapore Stable Assets - SGS.md`
- `Singapore Stable Assets - CPF.md`
- `Singapore Stable Assets - Tbills.md`

---

## Benchmark Categories

FinSafeQA contains multiple benchmark categories designed to evaluate different financial reasoning capabilities.

```
src/benchmarks/
```

Available benchmark tasks include:

| Benchmark File | Task Type |
|---|---|
| `retrieval_india.json` | Financial retrieval |
| `retrieval_singapore.json` | Country-aware retrieval |
| `reasoning_india.json` | Multi-hop reasoning |
| `reasoning_singapore.json` | Financial reasoning |
| `comparison_india.json` | Comparative analysis |
| `comparison_singapore.json` | Cross-instrument comparison |
| `financial_calculation.json` | Numerical reasoning |
| `tax_regulation.json` | Taxation QA |
| `temporal_reasoning.json` | Time-based reasoning |
| `hallucination_tests.json` | Hallucination robustness |
| `noisy_queries.json` | Retrieval robustness |
| `multi_scheme_reasoning.json` | Multi-document reasoning |
| `scheme_rules.json` | Regulatory understanding |
| `robustness_queries.json` | Adversarial evaluation |

---

## Key Features

FinSafeQA evaluates RAG systems on:

- Country-aware financial retrieval
- Stable asset reasoning
- Government regulation understanding
- Tax and compliance reasoning
- Financial calculations
- Temporal financial reasoning
- Hallucination resistance
- Multi-hop retrieval
- Explainable answer generation
- Noisy and adversarial query robustness

---

## Architecture Overview

The framework integrates:

- Adaptive financial chunking
- Hybrid semantic-lexical retrieval
- Country-specific indexing
- Explainable prompting
- Validation-driven generation

### Financial Document Ingestion

Financial documents are automatically enriched with metadata such as:

- Country
- Regulator
- Currency
- Asset class
- Financial topics

The ingestion pipeline additionally performs:

- Topic inference
- Chunk enrichment
- Metadata tagging
- Cache-based preprocessing

### Adaptive Financial Chunking

FinSafeQA uses financial-aware chunking that preserves:

- Page boundaries
- Section headings
- Financial tables
- Overlapping contextual information

Large tables are chunked row-wise while preserving headers to maintain numerical consistency.

### Country-Aware Indexing

The system builds:

- Separate FAISS indexes
- Separate BM25 indexes
- Document maps

for each country.

**Examples:**

```
data/faiss/
├── India.index
├── Singapore.index
└── Macro.index
```

This prevents cross-country retrieval contamination and improves regulatory consistency.

### Hybrid Retrieval

The retrieval pipeline combines:

- Semantic retrieval using FAISS
- Lexical retrieval using BM25

The system additionally:

- Detects country references
- Filters weak matches
- Handles ambiguous financial abbreviations
- Performs diversity-aware reranking

### Multi-Hop Financial Reasoning

Complex financial queries are decomposed into smaller sub-questions for:

- Independent retrieval
- Intermediate reasoning
- Final answer aggregation

This improves:

- Comparative reasoning
- Long-form financial analysis
- Multi-document synthesis

### Explainable Generation

Generated responses follow a structured explainable format containing:

- Summary
- Step-by-step reasoning
- Calculations
- Assumptions
- Sources
- Confidence estimation

### Validation Layer

Responses are validated for:

- Citation correctness
- Hallucinated references
- Structural completeness
- Reasoning consistency

---

## Installation

### Clone Repository

```bash
git clone <your_repo_url>
cd FinSafeQA
```

### Create Environment

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

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Dependencies** (example):

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

**Set backend:**

```bash
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2
```

---

## Running the Pipeline

### Step 1 — Ingest Financial Documents

```python
from src.ingestion import ingest_markdown

docs = ingest_markdown(
    "data/processed/India_Stable_Assets__FD.md"
)
```

### Step 2 — Load Embedding Model

```python
from src.embeddings import load_embedding_model

embed_model = load_embedding_model()
```

### Step 3 — Build Country Indexes

```python
from src.country_indexes import build_country_indexes

build_country_indexes(
    documents=docs,
    embed_model=embed_model,
    out_dir="data"
)
```

### Step 4 — Load Indexes

```python
from src.country_indexes import load_country_indexes

country_indexes = load_country_indexes("data")
```

### Step 5 — Initialize LLM

**Ollama:**

```python
from src.rag import init_llm

llm = init_llm(
    backend="ollama"
)
```

**or NVIDIA NIM:**

```python
llm = init_llm(
    api_key="YOUR_API_KEY",
    backend="nvidia"
)
```

### Step 6 — Run Financial QA

```python
from src.rag import rag_query

response = rag_query(
    query="Compare PPF and SGS returns",
    country_indexes=country_indexes,
    embed_model=embed_model,
    llm=llm
)

print(response)
```

---

## Example Queries

- What are FD rates in India?
- Compare PPF and Singapore Government Securities.
- What are tax benefits of Sovereign Gold Bonds?
- Summarize retirement schemes in Singapore.
- Which stable asset is inflation resistant?

---

## Running Benchmarks

Benchmark files are located in:

```
src/benchmarks/
```

Example benchmark categories:

- Retrieval benchmarks
- Reasoning benchmarks
- Hallucination tests
- Noisy query robustness
- Temporal reasoning
- Comparative financial analysis

**Run evaluation:**

```bash
python3 -m src.evaluate
```

**or programmatically:**

```python
from src.evaluate import evaluate
```

---

## Evaluation Dimensions

FinSafeQA evaluates:

- Retrieval accuracy
- Reasoning quality
- Country consistency
- Citation faithfulness
- Hallucination robustness
- Numerical correctness
- Explainability
- Temporal reasoning
- Adversarial robustness
