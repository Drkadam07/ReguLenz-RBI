# ReguLenz RBI — Regulatory Compliance RAG & Evaluation Engine

An enterprise-grade, evaluation-first Retrieval-Augmented Generation (RAG) system for querying Reserve Bank of India (RBI) notifications, Master Directions, Master Circulars, and circulars with evidence-grounded answers and source attribution.

ReguLenz RBI uses an inspectable information-retrieval pipeline built in Python: layout-aware PDF parsing, hybrid retrieval (BM25 + FAISS with Reciprocal Rank Fusion), neural cross-encoder reranking, asynchronous background indexing, and retrieval benchmarking with Recall@k and Mean Reciprocal Rank (MRR).

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Retrieval Evaluation](#retrieval-evaluation)
- [API Reference](#api-reference)
- [Example Query](#example-query)
- [License](#license)

## Key Features

- **Layout-aware PDF ingestion:** Uses PyMuPDF (`fitz`) and bounding-box filters to remove running headers, footers, and margins while preserving regulatory clauses and sections.
- **Hybrid retrieval:** Combines BM25 lexical retrieval with dense BGE embeddings indexed in FAISS. Reciprocal Rank Fusion (RRF) combines the result lists.
- **Regulatory identifier retrieval:** Supports exact identifiers and acronyms such as circular codes, LEI, and NBFC-UL alongside semantic search.
- **Neural reranking:** Uses `BAAI/bge-reranker-base` to rerank query-document pairs and reduce semantic false positives.
- **Evidence-grounded generation:** Prompt guardrails instruct the model to answer from retrieved context, cite sources, and abstain when the evidence is insufficient.
- **Source attribution:** Answers can cite document identifiers, page numbers, and chunk IDs.
- **Asynchronous ingestion:** FastAPI background tasks allow large document indexing to run without blocking the request.
- **Retrieval evaluation:** Measures Recall@1, Recall@3, Recall@5, Recall@10, and MRR against labeled query-to-chunk ground truth.
- **Local LLM inference:** Uses Ollama with models such as Qwen 2.5 or Llama 3.2.

## Architecture

```text
RBI Regulatory PDFs
        |
        v
Layout-aware PDF parser
(header/footer filtering; clause preservation)
        |
        v
Boundary-aware regulatory chunker
(metadata and chunk IDs)
        |
        +--------------------------+
        |                          |
        v                          v
BM25 lexical search        BGE embeddings + FAISS
        |                          |
        +------------+-------------+
                     v
          Reciprocal Rank Fusion
                     |
                     v
           BGE cross-encoder reranker
                     |
                     v
          Strict context / citation prompts
                     |
                     v
             Local Ollama LLM
                     |
                     v
       Grounded answer + source citations
```

The RRF score is computed from the ranks of results across retrieval lists:

\[
\mathrm{RRF}(d) = \sum_r \frac{1}{60 + \mathrm{rank}_r(d)}
\]

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| PDF parsing | PyMuPDF (`fitz`) |
| Lexical retrieval | Rank-BM25 |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector search | FAISS (`IndexFlatIP`) with normalized vectors |
| Reranking | `BAAI/bge-reranker-base` |
| Local LLM | Ollama (`qwen2.5:7b-instruct` or `llama3.2:3b`) |
| API | FastAPI, Pydantic v2, Uvicorn |
| Dashboard | Streamlit |
| Configuration | Pydantic Settings |
| Containers | Docker and Docker Compose |

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
│   ├── processed/                   # FAISS index and metadata
│   └── evaluation/
│       └── eval_benchmark.json      # Labeled evaluation questions
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── downloader.py            # RBI document downloader
│   │   ├── parser.py                # Layout-aware PDF extraction
│   │   └── chunker.py               # Regulatory chunking
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── prompt.py
│   │   └── llm.py
│   └── evaluation/
│       └── retrieval_eval.py
├── api/
│   └── main.py                      # FastAPI endpoints
└── app/
    └── streamlit_app.py             # Streamlit workspace
```

## Getting Started

### Prerequisites

- Python 3.11 or later
- [Ollama](https://ollama.com/) installed and running
- Git
- Sufficient RAM and disk space for the selected local model and embedding/reranking models

### 1. Pull a local LLM

Start the Ollama service, then pull a model.

```bash
# Standard model
ollama pull qwen2.5:7b-instruct

# Lightweight alternative
ollama pull qwen2.5:3b-instruct
```

Use a model tag that is available in your Ollama installation. If your configuration uses Llama 3.2, pull the corresponding tag instead.

### 2. Clone the repository

Replace the placeholder URL with the actual repository URL.

```bash
git clone https://github.com/your-username/rbi-rag.git
cd rbi-rag
```

### 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy the example environment file and update values as needed.

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Review `.env.example` and configure the settings required by your local setup. Do not commit secrets or machine-specific credentials.

### 6. Download sample documents and build the index

```bash
python -m src.ingestion.downloader
python build_index.py
```

The downloader and indexer should populate `data/raw/` and `data/processed/` respectively, based on the implementation and configuration in your checkout.

### 7. Run the API and dashboard

Start the FastAPI backend in one terminal:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Streamlit dashboard in another terminal:

```bash
streamlit run app/streamlit_app.py
```

Open the dashboard at [http://localhost:8501](http://localhost:8501).

FastAPI interactive documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Docker Deployment

The repository includes a `Dockerfile` and `docker-compose.yml` for containerized deployment. The Compose configuration is intended to connect the application container to an Ollama instance running on the host.

```bash
docker compose up --build
```

After startup, the usual local endpoints are:

- Streamlit workspace: [http://localhost:8501](http://localhost:8501)
- FastAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

**Note:** Confirm that the Compose file's Ollama host address and port are correct for your operating system and deployment environment.

## Retrieval Evaluation

Retrieval quality is evaluated against labeled question-to-chunk ground truth in:

```text
data/evaluation/eval_benchmark.json
```

Example benchmark format:

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

Run the evaluation through the API:

```bash
curl http://localhost:8000/evaluate
```

Alternatively, use the **Run IR Evaluation Benchmark** action in the Streamlit sidebar, if available in your checkout.

### Illustrative ablation results

The following values are illustrative results supplied in the project notes; reproduce the benchmark on your own dataset before treating them as independently verified.

| Pipeline configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Dense vector search only (BGE-Small) | 52.4% | 68.1% | 74.3% | 81.2% | 0.612 |
| BM25 lexical search only | 48.0% | 64.5% | 71.0% | 77.8% | 0.558 |
| Hybrid retrieval (BM25 + FAISS via RRF) | 64.8% | 79.2% | 85.6% | 91.0% | 0.724 |
| Hybrid + neural reranker (BGE-Reranker) | 73.5% | 88.4% | 93.1% | 95.2% | 0.816 |

## API Reference

The following endpoints are described by the project design. Exact request and response schemas should be confirmed against the running API and its `/docs` page.

### `GET /health`

Returns health information such as the connection status, indexed chunk count, and background-ingestion status.

### `POST /query`

Ask a question against the indexed regulatory knowledge base.

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
  "answer": "A grounded answer with document and page citations.",
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

Accepts a single PDF as `multipart/form-data`, then parses, chunks, and adds it to the active index.

### `POST /build-index-all`

Starts a background indexing task for PDFs in `data/raw/`.

### `GET /index-status`

Reports the status of the background indexing task.

## Example Query

You could ask questions such as:

- “What is the timeline for periodic KYC updation for high-risk customers?”
- “What requirements apply to a particular RBI circular?”
- “What does the retrieved Master Direction state about a specified compliance obligation?”

Answers should be checked against the cited RBI document and page. This tool supports regulatory research; it is not a substitute for legal advice or an institution's formal compliance review.

## Evidence and Limitations

- Answer quality depends on the source PDFs, parsing quality, index freshness, retrieval configuration, and selected local LLM.
- Prompt instructions reduce unsupported answers but cannot mathematically guarantee zero hallucinations.
- Benchmark scores depend on the labeled evaluation dataset and should be reproduced when the corpus or retrieval pipeline changes.
- Always verify regulatory interpretations against the authoritative RBI publication and applicable amendments.

## License

The project notes specify the **MIT License**. Include the corresponding `LICENSE` file in the repository and confirm that it matches the intended licensing terms.
