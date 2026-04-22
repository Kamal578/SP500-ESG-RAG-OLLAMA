# ESG RAG System (S&P 500 Sustainability Reports)

End-to-end Retrieval-Augmented Generation (RAG) project using local Ollama models, LlamaIndex orchestration, and ChromaDB persistent vector storage.

Dataset used: https://www.kaggle.com/datasets/jaidityachopra/esg-sustainability-reports-of-s-and-p-500-companies/data

## Stack
- Python 3.11+
- LLM: Ollama `llama3.1`
- Embeddings: Ollama `nomic-embed-text`
- Vector DB: ChromaDB (local persistent)
- Orchestration: LlamaIndex
- UI: Streamlit
- Deployment: Docker + Docker Compose

## Project Structure
```text
project_root/
├── data/
│   ├── preprocessed_content.csv
│   └── raw_txt/
├── vector_db/
├── src/
│   ├── 01_data_prep.py
│   ├── 02_build_index.py
│   ├── 03_rag_pipeline.py
│   ├── 04_app.py
│   └── common.py
├── tests/
├── pytest.ini
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── project_progress.md
└── README.md
```

## Prerequisites
- Python 3.11+
- Ollama installed and running locally
- Docker + Docker Compose (for containerized run)

## Environment Variables
Defaults are built-in, but can be overridden:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_LLM_MODEL` (default: `llama3.1`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
- `CHROMA_PATH` (default: `vector_db`)
- `CHROMA_COLLECTION` (default: `sp500_esg_reports`)
- `SIMILARITY_TOP_K` (default: `3`)
- `OLLAMA_CONTEXT_WINDOW` (default: `2048`)

## Quick Start (Local)
```bash
./scripts/setup.sh
./scripts/prepare_data.sh
./scripts/build_index.sh --rebuild
./scripts/evaluate.sh
./scripts/run_app.sh
```

Open Streamlit at `http://localhost:8501`.

## Quick Start (Docker)
```bash
./scripts/setup.sh
./scripts/prepare_data.sh
./scripts/build_index.sh --rebuild
./scripts/docker_up.sh
```

The containerized app expects Ollama to run on the host and uses `http://host.docker.internal:11434`.

## Testing
Install test dependencies and run the suite:
```bash
python -m pip install -r requirements-dev.txt
./scripts/test.sh
```

Run with coverage:
```bash
./scripts/test.sh --cov=src --cov-report=term-missing
```

## Script Reference
- `scripts/setup.sh`: install Python dependencies, verify Ollama, pull required models.
- `scripts/prepare_data.sh`: split CSV rows into TXT files in `data/raw_txt`.
- `scripts/build_index.sh`: build/rebuild Chroma vector index.
- `scripts/index_status.sh`: show index process status + current vector count.
- `scripts/evaluate.sh`: baseline vs RAG comparison on sample questions.
- `scripts/run_app.sh`: run Streamlit app locally.
- `scripts/full_local.sh`: run prepare → build index → evaluate.
- `scripts/docker_up.sh`: build and start Streamlit container.
- `scripts/docker_down.sh`: stop container.
- `scripts/test.sh`: run pytest test suite.

## Manual Commands
```bash
python src/01_data_prep.py
python src/02_build_index.py --rebuild
python src/03_rag_pipeline.py
streamlit run src/04_app.py
```

## Notes
- `vector_db/` is git-ignored by design (generated local state).
- `data/` is currently ignored in `.gitignore`; keep a local copy of `preprocessed_content.csv`.
- Index building can take significant time for this dataset and model setup.
- Chroma telemetry warnings are suppressed by pinned compatibility deps (`chromadb==0.5.23`, `posthog<4`) and `anonymized_telemetry=False` client settings.
