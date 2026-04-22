# Project Progress

## Current Snapshot
- Goal: End-to-end local RAG system over S&P 500 ESG sustainability report text.
- Status: Core codebase, containerization, and automated testing are implemented. Index build, evaluation, and local app startup checks have completed successfully.

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
- Chroma compatibility hardening:
  - `chromadb` pinned to `0.5.23` (avoids segfaults observed with `1.5.8` in this environment).
  - `posthog<4` pinned to fix Chroma telemetry API mismatch (`capture() takes 1 positional argument...`).
  - Chroma clients are created with `anonymized_telemetry=False`.
  - Docker and Compose set `ANONYMIZED_TELEMETRY=FALSE`.

### 7) Repository hygiene
- Added `vector_db/` to `.gitignore`.

### 8) Testing and test execution
- Added comprehensive `pytest` suite under `tests/` covering:
  - shared utility functions (`src/common.py`)
  - data preparation behavior (`src/01_data_prep.py`)
  - index build logic with mocks (`src/02_build_index.py`)
  - RAG pipeline helper/query-engine loading with mocks (`src/03_rag_pipeline.py`)
  - Streamlit helper functions (`src/04_app.py`)
- Added `pytest.ini` for test discovery/config.
- Added `requirements-dev.txt` for development/test dependencies.
- Added canonical test runner script: `scripts/test.sh`.
- Latest test result: `19 passed`.

## Runtime Status Right Now
- Ollama models installed:
  - `nomic-embed-text:latest`
  - `llama3.1:latest`
- Latest completed index run:
  - Command: `python src/02_build_index.py --rebuild`
  - Collection: `sp500_esg_reports`
  - Documents loaded: `866`
  - Documents skipped: `3`
  - Chunks created/stored: `24687`
- Evaluation status:
  - `src/03_rag_pipeline.py` executes successfully.
  - Baseline and RAG outputs are printed for sample questions.
  - Telemetry warnings are no longer printed after dependency/config updates.
- App status:
  - Streamlit app starts successfully on `http://localhost:8501`.
- Test status:
  - `./scripts/test.sh` passes (`19 passed`).

## What Is Pending
- Optional: run dockerized app validation (`./scripts/docker_up.sh`).
- Optional: tune retrieval and generation quality (`SIMILARITY_TOP_K`, prompt wording, question set).

## Handoff Commands
Use the shell scripts in `scripts/` (added in this update):
1. `./scripts/setup.sh`
2. `./scripts/prepare_data.sh`
3. `./scripts/build_index.sh --rebuild`
4. `./scripts/evaluate.sh`
5. `./scripts/run_app.sh`
6. `./scripts/docker_up.sh`
7. `python -m pip install -r requirements-dev.txt`
8. `./scripts/test.sh`

For live indexing status:
- `./scripts/index_status.sh`
