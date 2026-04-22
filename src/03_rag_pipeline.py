from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import chroma_persistent_client, resolve_from_root, runtime_config_from_env


DEFAULT_QUESTIONS = [
    "Which companies in the dataset discuss Scope 1 and Scope 2 emissions reduction targets?",
    "What workforce diversity metrics are commonly reported across the ESG reports?",
    "How do companies describe board oversight of ESG or sustainability strategy?",
    "What are the most frequently mentioned renewable energy initiatives?",
    "Which social impact themes appear most often in community investment sections?",
]


def parse_args() -> argparse.Namespace:
    cfg = runtime_config_from_env()
    parser = argparse.ArgumentParser(description="Run baseline vs RAG evaluation on ESG questions.")
    parser.add_argument("--chroma-path", default=cfg.chroma_path)
    parser.add_argument("--collection", default=cfg.chroma_collection)
    parser.add_argument("--ollama-base-url", default=cfg.ollama_base_url)
    parser.add_argument("--llm-model", default=cfg.llm_model)
    parser.add_argument("--embed-model", default=cfg.embed_model)
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "2048")),
    )
    parser.add_argument("--top-k", type=int, default=int(os.getenv("SIMILARITY_TOP_K", "3")))
    parser.add_argument("--question", action="append", help="Can be provided multiple times.")
    parser.add_argument("--questions-file", help="Path to a file with one question per line.")
    return parser.parse_args()


def collection_exists(client: chromadb.PersistentClient, target: str) -> bool:
    collections = client.list_collections()
    for item in collections:
        name = item.name if hasattr(item, "name") else str(item)
        if name == target:
            return True
    return False


def count_vectors_from_sqlite(chroma_path: Path, collection_name: str) -> int | None:
    db_path = chroma_path / "chroma.sqlite3"
    if not db_path.exists():
        return None
    query = """
    SELECT COUNT(*)
    FROM embeddings e
    JOIN segments s ON e.segment_id = s.id
    JOIN collections c ON s.collection = c.id
    WHERE c.name = ?
    """
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(query, (collection_name,)).fetchone()
        if row is None:
            return 0
        return int(row[0])
    except sqlite3.Error:
        return None


def collection_has_records(collection) -> bool:
    sample = collection.peek(limit=1)
    ids = sample.get("ids", []) if isinstance(sample, dict) else []
    return bool(ids)


def load_questions(args: argparse.Namespace) -> list[str]:
    if args.question:
        return [q.strip() for q in args.question if q.strip()]
    if args.questions_file:
        path = resolve_from_root(args.questions_file)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        questions = [line for line in lines if line]
        if not questions:
            raise ValueError(f"Questions file is empty: {path}")
        return questions
    return DEFAULT_QUESTIONS


def load_query_engine(args: argparse.Namespace):
    chroma_path = resolve_from_root(args.chroma_path)
    if not chroma_path.exists():
        raise RuntimeError(
            f"Chroma path not found: {chroma_path}. Run `python src/02_build_index.py --rebuild` first."
        )

    client = chroma_persistent_client(chroma_path)
    if not collection_exists(client, args.collection):
        raise RuntimeError(
            f"Collection '{args.collection}' not found at {chroma_path}. "
            "Run `python src/02_build_index.py --rebuild` first."
        )

    collection = client.get_collection(args.collection)
    if not collection_has_records(collection):
        raise RuntimeError(
            f"Collection '{args.collection}' is empty. Run `python src/02_build_index.py --rebuild` first."
        )

    llm = Ollama(
        model=args.llm_model,
        base_url=args.ollama_base_url,
        request_timeout=180.0,
        temperature=0,
        context_window=args.context_window,
    )
    embed_model = OllamaEmbedding(model_name=args.embed_model, base_url=args.ollama_base_url)
    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    query_engine = index.as_query_engine(similarity_top_k=args.top_k, llm=llm)
    count = count_vectors_from_sqlite(chroma_path, args.collection)
    return llm, query_engine, count


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def main() -> int:
    args = parse_args()
    questions = load_questions(args)
    llm, query_engine, count = load_query_engine(args)

    if count is None:
        print(f"Loaded collection '{args.collection}' (vector count unavailable).")
    else:
        print(f"Loaded collection '{args.collection}' with {count} vectors.")
    print(f"Using LLM model: {args.llm_model}")
    print(f"Using embedding model: {args.embed_model}")
    print(f"Context window: {args.context_window}")
    print(f"Similarity top-k: {args.top_k}")

    for idx, question in enumerate(questions, start=1):
        baseline_answer = llm.complete(question).text.strip()
        rag_response = query_engine.query(question)
        rag_answer = str(rag_response).strip()

        retrieved_files: list[str] = []
        for source_node in rag_response.source_nodes or []:
            metadata = source_node.node.metadata if source_node.node else {}
            retrieved_files.append(metadata.get("source_file", "unknown"))
        retrieved_files = unique_preserve_order(retrieved_files)

        print("=" * 100)
        print(f"Question {idx}: {question}")
        print("-" * 100)
        print("Baseline (LLM only):")
        print(baseline_answer)
        print("-" * 100)
        print("RAG (retrieval + LLM):")
        print(rag_answer)
        print("-" * 100)
        print("Retrieved source files:")
        if retrieved_files:
            for source in retrieved_files:
                print(f"- {source}")
        else:
            print("- None")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
