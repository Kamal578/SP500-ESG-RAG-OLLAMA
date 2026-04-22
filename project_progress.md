# Project Progress

## Current Snapshot
- Goal: End-to-end local RAG system over S&P 500 ESG sustainability report text.
- Status: Implementation is complete for the assignment scope, with reproducible scripts, evaluation artifact export, metrics aggregation, Streamlit demo UX, automated tests, and CI.

## Implemented Components

### 1) Data preparation (`src/01_data_prep.py`)
- Reads `data/preprocessed_content.csv` with pandas.
- Validates required columns:
  - `ticker`
  - `year`
  - `preprocessed_content`
- Writes one UTF-8 TXT per valid row into `data/raw_txt/`.
- Uses deterministic duplicate suffixing (`_2`, `_3`, ...).

### 2) Index build (`src/02_build_index.py`)
- Loads TXT files with `SimpleDirectoryReader`.
- Chunking uses strict stored bounds `(201, 499)` with target `420`.
- Adds chunk metadata:
  - `source_file`
  - `ticker`
  - `year`
  - `chunk_id`
  - `token_count`
- Embeddings via local Ollama `nomic-embed-text`.
- Stores vectors in persistent Chroma collection at `vector_db/`.
- Supports `--rebuild`.

### 3) RAG pipeline + baseline eval (`src/03_rag_pipeline.py`)
- Loads persisted Chroma index and builds query engine (`top_k`, default `3`).
- Runs two paths per question:
  - baseline LLM-only answer
  - RAG answer with retrieval
- Prints retrieved source filenames.
- Supports default 5 questions and CLI overrides (`--question`, `--questions-file`).
- New: writes machine-readable outputs to `outputs/eval/`:
  - `rag_eval_*.json`
  - `rag_eval_*.csv`

### 4) Evaluation metrics (`src/05_eval_metrics.py`)
- Loads latest or specified `rag_eval_*.json`.
- Computes summary metrics:
  - non-empty answer rates
  - average answer lengths
  - average retrieved sources per question
  - question coverage with citations
  - unique retrieved source count
- Writes:
  - `rag_metrics_*.json`
  - `rag_metrics_*.csv`

### 5) Streamlit UI (`src/04_app.py`)
- Styled interface with curated demo prompts.
- Shows generated answer and exact retrieved chunks with metadata + similarity.
- New UX features:
  - one-click `Run Demo Set (3)`
  - session query history
  - export latest query as JSON
  - export full session as Markdown

### 6) Preflight checks (`src/preflight.py`, `scripts/preflight.sh`)
- Verifies dataset/index paths and optional collection readiness.
- Checks Ollama model availability (can be skipped).
- `--require-index` fails fast when vector store is missing/empty.

## Documentation Added
- `README.md` refreshed with full workflow and artifact commands.
- `docs/assignment_mapping.md`
- `docs/dataset_profile.md`
- `docs/architecture.md`
- `docs/results_summary.md`
- `docs/submission_checklist.md`
- `data/eval_questions.txt` added (30 ESG evaluation prompts).
- `.env.example` added for standard runtime configuration.

## Scripts Added/Updated
- Added:
  - `scripts/preflight.sh`
  - `scripts/eval_metrics.sh`
- Updated:
  - `scripts/full_local.sh` now runs metrics after evaluation.

## Testing and CI
- Test suite expanded with new modules:
  - `tests/test_eval_metrics.py`
  - `tests/test_preflight.py`
  - `tests/test_app_demo.py`
- Existing tests retained and updated where needed.
- Current test result: `29 passed`.
- CI added: `.github/workflows/tests.yml` runs pytest on push/PR (Python 3.11).

## Repository Hygiene
- `.gitignore` updated to:
  - keep `vector_db/` ignored
  - keep `outputs/eval/` ignored
  - keep local dataset artifacts ignored
  - allow committed `data/eval_questions.txt`

## Runtime Verification (Latest)
- Preflight check succeeded:
  - command: `./scripts/preflight.sh --skip-model-check --require-index`
- Pipeline output generation verified with one live query:
  - command: `python src/03_rag_pipeline.py --question "Which companies mention renewable energy procurement?" --top-k 3`
  - output files created under `outputs/eval/`.
- Metrics generation verified:
  - command: `python src/05_eval_metrics.py`
  - metrics files created under `outputs/eval/`.

## Handoff Commands
1. `./scripts/setup.sh`
2. `./scripts/preflight.sh`
3. `./scripts/prepare_data.sh`
4. `./scripts/build_index.sh --rebuild`
5. `./scripts/evaluate.sh --questions-file data/eval_questions.txt`
6. `./scripts/eval_metrics.sh`
7. `./scripts/run_app.sh`
8. `python -m pip install -r requirements-dev.txt`
9. `./scripts/test.sh`

For index monitoring:
- `./scripts/index_status.sh`
