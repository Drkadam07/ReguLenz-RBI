# ReguLenz RBI — Regulatory Compliance RAG & Evaluation Engine

**An Evaluation-First, Grounded Retrieval-Augmented Generation (RAG) System for Regulatory Compliance and RBI Circulars**

ReguLenz RBI is a RegTech and financial information retrieval project designed to parse, index, search, and answer questions about Reserve Bank of India (RBI) Master Directions, circulars, and notifications. It combines layout-aware PDF extraction, hybrid lexical and dense retrieval, cross-encoder reranking, evidence-grounded generation, source citations, and quantitative retrieval evaluation.

Unlike a basic “chat with PDF” application, ReguLenz RBI is designed as an inspectable information-retrieval pipeline. Its goal is to make regulatory research more auditable by surfacing relevant source passages and abstaining when the available evidence is insufficient.

> **Important:** Grounding prompts and relevance thresholds can reduce unsupported answers, but they do not mathematically guarantee zero hallucinations. Always verify regulatory interpretations against the authoritative RBI publication and applicable amendments.

## Table of Contents

- [Project Overview](#project-overview)
- [Aim, Goal, and Objectives](#aim-goal-and-objectives)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Retrieval Evaluation](#retrieval-evaluation)
- [API Reference](#api-reference)
- [Target Audience and Use Cases](#target-audience-and-use-cases)
- [Privacy and Limitations](#privacy-and-limitations)
- [License](#license)

## Project Overview

| Item | Description |
|---|---|
| Project name | **ReguLenz RBI** |
| Alternative name | KAVACH-RAG |
| Domain | RegTech, financial information retrieval, compliance engineering |
| Primary purpose | Query RBI regulatory documents and return evidence-grounded answers with source attribution |
| Retrieval approach | BM25 + dense FAISS retrieval combined using Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking |
| Generation | Local LLM inference through Ollama |
| Interfaces | FastAPI REST API and Streamlit dashboard |

## Aim, Goal, and Objectives

### Aim

Reduce friction in regulatory research by converting complex RBI PDF archives into an auditable, cited, and queryable knowledge base.

### Goal

Engineer, containerize, and benchmark an end-to-end RAG pipeline that targets **over 90% Recall@5** on a labeled set of regulatory queries while supporting local inference without external proprietary LLM APIs.

This is a project target, not a guaranteed result. Actual performance must be measured on the evaluation dataset used with the current corpus and configuration.

### Objectives

- **Document-aware ingestion:** Filter repeated headers, footers, and margins using PDF layout coordinates while preserving regulatory section structure.
- **Hybrid information retrieval:** Combine BM25 lexical search for exact circular identifiers and acronyms with BGE dense embeddings for semantic queries.
- **Precision reranking:** Apply a neural cross-encoder to reorder retrieved candidates by query relevance.
- **Evidence grounding:** Require citations identifying the source document, page, and chunk; abstain when retrieved evidence is insufficient.
- **Quantitative evaluation:** Measure Recall@1, Recall@3, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).
- **Production interfaces:** Provide a FastAPI backend with asynchronous indexing and a Streamlit interface for interactive research and evaluation.

## Key Features

### 1. Retrieval and Ingestion

- **Coordinate-aware PDF parsing:** Uses PyMuPDF (`fitz`) and page bounding boxes to filter repeated headers, footers, and marginal noise.
- **Regulatory-aware chunking:** Aims to preserve clauses, sections, sub-clauses, and useful metadata rather than splitting text at arbitrary character boundaries.
- **Hybrid BM25 + FAISS search:** Combines lexical matching and dense semantic similarity through Reciprocal Rank Fusion.
- **Exact identifier retrieval:** BM25 helps surface circular numbers such as `RBI/2024-25/18` and acronyms such as `LEI` and `NBFC-UL`.
- **Cross-encoder reranking:** Uses `BAAI/bge-reranker-base` to rerank query-passage candidates before context is sent to the LLM.

### 2. Generation and Guardrails

- **Context-constrained answers:** Uses a defensive system prompt to instruct the local LLM to answer from retrieved evidence.
- **Granular provenance:** Designed to cite source document, page number, and chunk ID for answer verification.
- **Explicit abstention:** Can return an “Insufficient evidence found” response when evidence is missing or does not meet configured relevance criteria.
- **Local inference:** Uses Ollama with a compatible local model, such as Qwen 2.5.

### 3. Engineering and User Interface

- **Background indexing:** Supports a non-blocking indexing workflow for larger PDF collections.
- **FastAPI REST API:** Exposes endpoints for querying, document ingestion, health checks, and retrieval evaluation.
- **Streamlit dashboard:** Provides an interactive workspace for querying the knowledge base, inspecting citations, and running benchmarks.
- **Containerization:** Includes Docker and Docker Compose configuration for repeatable deployment.

## Architecture

```text
                  RBI Regulatory Documents (PDFs)
                               |
                               v
                 Layout-Aware PDF Parser
              (header/footer filtering, structure)
                               |
                               v
                 Regulatory-Aware Chunker
                 (metadata and chunk IDs)
                               |
                   +-----------+-----------+
                   |                       |
                   v                       v
             BM25 Lexical Search     BGE Dense Embeddings
                                           |
                                           v
                                      FAISS Index
                   |                       |
                   +-----------+-----------+
                               |
                               v
                    Reciprocal Rank Fusion
                               |
                               v
                    BGE Cross-Encoder Reranker
                               |
                               v
                    Retrieved Evidence Context
                               |
                               v
                     Grounding / Citation Prompt
                               |
                               v
                       Local Ollama LLM
                               |
                               v
                   Answer with Source Citations
                     or Evidence-Based Abstention
```

Reciprocal Rank Fusion combines rankings from different retrieval methods. A common formulation is:

\[
\operatorname{RRF}(d) = \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}
\]

where \(d\) is a candidate document, \(R\) is the set of ranked result lists, and \(k\) is a smoothing constant (commonly 60).

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| PDF parsing | PyMuPDF (`fitz`) |
| Lexical retrieval | Rank-BM25 |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Vector index | FAISS (`IndexFlatIP`) with normalized vectors |
| Reranker | `BAAI/bge-reranker-base` |
| Local LLM runtime | Ollama |
| Example LLMs | `qwen2.5:7b-instruct`, `qwen2.5:3b-instruct`, or a configured Llama 3.2 model |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| Dashboard | Streamlit |
| Configuration | Pydantic Settings |
| Deployment | Docker and Docker Compose |

## Repository Structure

```text
rbi-rag/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── build_index.py
├── README.md
├── data/
│   ├── raw/                         # Source RBI PDFs
│   ├── processed/                   # Generated index and metadata
│   └── evaluation/
│       └── eval_benchmark.json      # Labeled query/chunk pairs
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── downloader.py
│   │   ├── parser.py
│   │   └── chunker.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt.py
│   │   └── llm.py
│   └── evaluation/
│       ├── __init__.py
│       └── retrieval_eval.py
├── api/
│   ├── __init__.py
│   └── main.py
└── app/
    └── streamlit_app.py
```

## Getting Started

### Prerequisites

- Python 3.11 or later
- Git
- [Ollama](https://ollama.com/) installed and running
- Adequate RAM and disk space for the selected LLM, embedding model, and reranker

### 1. Download a local model

Start Ollama, then pull a model supported by your configuration.

```bash
ollama pull qwen2.5:7b-instruct
```

For a lighter model, if the tag is available in your Ollama setup:

```bash
ollama pull qwen2.5:3b-instruct
```

### 2. Clone the repository

Replace the placeholder URL with the actual repository URL.

```bash
git clone https://github.com/Drkadam07/ReguLenz-RBI.git
cd rbi-rag
```

### 3. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS**

```bash
cp .env.example .env
```

Review `.env.example` and set the values required by your environment. Do not commit secrets or credentials.

### 6. Download documents and build the index

```bash
python -m src.ingestion.downloader
python build_index.py
```

Confirm that the downloaded PDFs are present in `data/raw/` and that index artifacts are created in `data/processed/`.

### 7. Start the backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

FastAPI interactive documentation should be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 8. Start the Streamlit dashboard

In a second terminal, with the virtual environment activated:

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

> These commands reflect the documented project layout. Check the actual repository configuration and endpoint implementation if a command or route differs in your checkout.

## Docker Deployment

Build and start the services:

```bash
docker compose up --build
```

When the services are running, check:

- Streamlit dashboard: [http://localhost:8501](http://localhost:8501)
- FastAPI OpenAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

Ensure the Compose configuration points to a reachable Ollama service. Host networking and hostname configuration can differ across Windows, macOS, and Linux.

## Retrieval Evaluation

Evaluation is a core part of ReguLenz RBI. It measures whether relevant source chunks are retrieved, independently of how fluent the generated answer sounds.

### Benchmark dataset

The project uses `data/evaluation/eval_benchmark.json` to map questions to relevant chunk IDs. Example format:

```json
[
  {
    "question": "What is the timeline for periodic KYC updation for high-risk customers?",
    "relevant_chunks": [
      "Master_Direction_KYC_p18_0023",
      "Master_Direction_KYC_p19_0024"
    ]
  },
  {
    "question": "What are the rules regarding penal charges on loan accounts?",
    "relevant_chunks": [
      "Fair_Lending_Practice_Penal_Charges_p02_0003"
    ]
  }
]
```

### Metrics

- **Recall@k:** Whether relevant chunks appear within the top \(k\) retrieved results.
- **MRR (Mean Reciprocal Rank):** The average reciprocal rank of the first relevant result.

Run the evaluation endpoint, if implemented in the current checkout:

```bash
curl http://localhost:8000/evaluate
```

You may also be able to run the benchmark from the Streamlit sidebar.

### Illustrative benchmark results

The following results were included in the supplied project notes. They should be treated as illustrative until reproduced against the current code, corpus, and labeled evaluation set.

| Pipeline configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Dense vector search only (BGE-Small) | 52.4% | 68.1% | 74.3% | 81.2% | 0.612 |
| BM25 lexical search only | 48.0% | 64.5% | 71.0% | 77.8% | 0.558 |
| Hybrid retrieval (BM25 + FAISS via RRF) | 64.8% | 79.2% | 85.6% | 91.0% | 0.724 |
| Hybrid + neural reranker (BGE-Reranker) | 73.5% | 88.4% | 93.1% | 95.2% | 0.816 |

The project goal is to exceed **90% Recall@5**. The illustrative hybrid-plus-reranker row reports 93.1% Recall@5, but this result must be validated by rerunning the benchmark before being presented as a verified measurement.

## API Reference

The following routes are part of the documented design. Confirm exact request schemas and behavior in the running API's `/docs` page.

### `GET /health`

Reports service health and may include index statistics or background task status.

### `POST /query`

Submit a regulatory question and request a limited number of retrieved chunks.

Example request:

```json
{
  "question": "What are the requirements for periodic KYC updation?",
  "top_k": 3
}
```

Example response shape:

```json
{
  "answer": "A context-grounded answer with source citations.",
  "sources": [
    {
      "document_id": "Master_Direction_KYC",
      "page": 18,
      "chunk_id": "Master_Direction_KYC_p18_0023",
      "rerank_score": 0.9412,
      "snippet": "Relevant source passage..."
    }
  ]
}
```

### `POST /ingest`

Accepts a PDF through `multipart/form-data` and starts the document ingestion process, according to the API implementation.

### `POST /build-index-all`

Starts a background rebuild over the PDF files in `data/raw/`, if implemented in the current checkout.

### `GET /index-status`

Reports the status of the background indexing task, if implemented.

### `GET /evaluate`

Runs or returns retrieval benchmark results, depending on the implementation.

## Target Audience and Use Cases

- **Banking and NBFC compliance officers:** Find and cross-reference relevant Master Directions during compliance reviews.
- **Fintech product managers and legal teams:** Research regulatory requirements, such as KYC update cycles, digital lending rules, and prepaid payment instrument limits.
- **Internal and external auditors:** Locate supporting passages for regulatory research and audit preparation.
- **RAG and information-retrieval engineers:** Explore an evaluation-driven architecture for domain-specific document retrieval.

ReguLenz RBI is a research assistant, not a substitute for legal advice, professional compliance review, or checking the latest authoritative RBI publication.

## Privacy and Limitations

- Local inference can keep document content within the local environment when all configured services and models run locally. Verify your actual deployment and network configuration before claiming an air-gapped setup.
- Retrieval and answer quality depend on source document quality, parsing, chunking, index freshness, retrieval parameters, and the selected model.
- Prompt guardrails and abstention thresholds reduce risk but cannot guarantee that every generated claim is correct.
- Evaluation metrics are meaningful only in relation to the benchmark dataset, corpus, and retrieval settings used.
- Regulatory documents can be amended or superseded. Verify dates, applicability, and amendments before making compliance decisions.

## License

The original project notes specify the **MIT License**. Include a `LICENSE` file containing the intended MIT License text and verify that the repository's licensing choice is correct before distribution.
