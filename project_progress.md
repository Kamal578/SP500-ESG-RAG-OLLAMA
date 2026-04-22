# Project Progress

## Current Snapshot
- Goal: End-to-end local RAG system over S&P 500 ESG sustainability report text.
- Status: Core codebase and containerization are implemented. Index rebuild is currently in progress under a pinned Chroma version for runtime stability.

## What Has Been Implemented

### 1) Data preparation
- Implemented `src/01_data_prep.py`.
- Reads `data/preprocessed_content.csv` and validates required columns:
  - `ticker`
  - `year`
  - `preprocessed_content`
- Writes one TXT file per row to `data/raw_txt/`.
- File naming format: `{TICKER}_{YEAR}.txt`.
- Duplicate ticker-year rows are handled deterministically with suffixes (`_2`, `_3`, ...).
- Run result in this workspace:
  - Total rows: `866`
  - Files written: `866`
  - Skipped rows: `0`

### 2) Index build pipeline
- Implemented `src/02_build_index.py`.
- Uses:
  - ChromaDB persistent storage at `vector_db/`
  - Ollama embeddings: `nomic-embed-text`
  - LlamaIndex vector store integration
- Loads TXT files with `SimpleDirectoryReader`.
- Performs strict chunking bounds for stored chunks:
  - min token count: `201`
  - max token count: `499`
  - target chunk size: `420`
- Adds metadata on each stored chunk:
  - `source_file`
  - `ticker`
  - `year`
  - `token_count`
  - `chunk_id`
- Supports `--rebuild` to clear and recreate collection.

### 3) RAG + baseline evaluation
- Implemented `src/03_rag_pipeline.py`.
- Loads persistent Chroma collection and creates query engine with configurable `top-k` (default `3`).
- Compares:
  - Baseline LLM answer (no retrieval context)
  - RAG answer (retrieval + generation)
- Includes 5 default ESG questions and supports custom questions via CLI.
- Emits retrieved source file names for each question.
- Fails fast with actionable errors when index/collection is missing or empty.

### 4) Streamlit UI
- Implemented `src/04_app.py`.
- Features:
  - Question input and answer generation
  - Cached query engine load (`st.cache_resource`)
  - Expandable retrieved context inspection
  - Displays source metadata (`source_file`, `ticker`, `year`) and similarity score
  - Displays exact retrieved chunk text

### 5) Shared runtime config + utilities
- Implemented `src/common.py` with:
  - env-driven runtime defaults
  - project-root path resolution
  - filename sanitization
  - ticker/year parsing from file names
  - tokenization and strict chunking helper
- Shared environment defaults:
  - `OLLAMA_BASE_URL` (default `http://localhost:11434`)
  - `OLLAMA_LLM_MODEL` (default `llama3.1`)
  - `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`)
  - `CHROMA_PATH` (default `vector_db`)
  - `CHROMA_COLLECTION` (default `sp500_esg_reports`)

### 6) Reproducibility and deployment
- `requirements.txt` created with LlamaIndex + Ollama + Chroma + Streamlit + pandas dependencies.
- Multi-stage `Dockerfile` created.
- `docker-compose.yml` created with:
  - streamlit port mapping `8501:8501`
  - `host.docker.internal` access to host Ollama
  - bind mounts for `data/` and `vector_db/`
- Chroma dependency is now pinned to `chromadb==0.5.23` to avoid segmentation faults seen with `chromadb 1.5.8` during retrieval/query on this machine.

### 7) Repository hygiene
- Added `vector_db/` to `.gitignore`.

## Runtime Status Right Now
- Ollama models installed:
  - `nomic-embed-text:latest`
  - `llama3.1:latest`
- Current active index rebuild:
  - Command: `python src/02_build_index.py --rebuild`
  - PID at check time: `38428`
  - Collection: `sp500_esg_reports`
  - Current stored vectors (during run): `4096`
- Why rebuild is running:
  - Initial index was built with `chromadb 1.5.8`.
  - `chromadb 1.5.8` caused segmentation faults during `query`/`peek`/`count` in this environment.
  - Downgraded and pinned to `chromadb==0.5.23`.
  - Cleared old `vector_db/` and started a clean rebuild for compatibility.

## What Is Pending
- Wait for current rebuild to finish.
- Run `src/03_rag_pipeline.py` end-to-end after rebuild completion.
- Optionally launch Streamlit UI and dockerized app for interactive checks.

## Handoff Commands
Use the shell scripts in `scripts/` (added in this update):
1. `./scripts/setup.sh`
2. `./scripts/prepare_data.sh`
3. `./scripts/build_index.sh --rebuild`
4. `./scripts/evaluate.sh`
5. `./scripts/run_app.sh`
6. `./scripts/docker_up.sh`

For live indexing status:
- `./scripts/index_status.sh`
