from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import chromadb
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import chroma_persistent_client, resolve_from_root, runtime_config_from_env


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


@st.cache_resource(show_spinner=False)
def load_query_engine(
    chroma_path_value: str,
    collection_name: str,
    ollama_base_url: str,
    llm_model: str,
    embed_model_name: str,
    context_window: int,
    top_k: int,
):
    chroma_path = resolve_from_root(chroma_path_value)
    if not chroma_path.exists():
        raise RuntimeError(
            f"Vector DB path not found: {chroma_path}. Run `python src/02_build_index.py --rebuild` first."
        )

    client = chroma_persistent_client(chroma_path)
    if not collection_exists(client, collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' not found in {chroma_path}. "
            "Run `python src/02_build_index.py --rebuild` first."
        )

    collection = client.get_collection(collection_name)
    if not collection_has_records(collection):
        raise RuntimeError(
            f"Collection '{collection_name}' is empty. Run `python src/02_build_index.py --rebuild` first."
        )

    llm = Ollama(
        model=llm_model,
        base_url=ollama_base_url,
        request_timeout=180.0,
        temperature=0,
        context_window=context_window,
    )
    embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_base_url)

    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    query_engine = index.as_query_engine(similarity_top_k=top_k, llm=llm)
    vector_count = count_vectors_from_sqlite(chroma_path, collection_name)
    return query_engine, vector_count


def format_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def main() -> None:
    cfg = runtime_config_from_env()
    top_k = int(os.getenv("SIMILARITY_TOP_K", "3"))
    context_window = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "2048"))

    st.set_page_config(page_title="S&P 500 ESG RAG", layout="wide")
    st.title("S&P 500 ESG Sustainability Reports RAG")

    try:
        query_engine, vector_count = load_query_engine(
            cfg.chroma_path,
            cfg.chroma_collection,
            cfg.ollama_base_url,
            cfg.llm_model,
            cfg.embed_model,
            context_window,
            top_k,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if vector_count is None:
        st.caption(
            f"Collection: {cfg.chroma_collection} | Vector count: unavailable | Context window: {context_window}"
        )
    else:
        st.caption(
            f"Collection: {cfg.chroma_collection} | Vector count: {vector_count} | Context window: {context_window}"
        )

    with st.form("query_form"):
        question = st.text_input("Ask a question about ESG reports")
        submitted = st.form_submit_button("Generate answer")

    if not submitted:
        st.caption("Enter a question and click Generate answer.")
        return

    if not question.strip():
        st.warning("Please enter a non-empty question.")
        return

    with st.spinner("Running retrieval and generation..."):
        response = query_engine.query(question.strip())

    st.subheader("Answer")
    st.write(str(response))

    with st.expander("Retrieved Context Chunks", expanded=True):
        source_nodes = response.source_nodes or []
        if not source_nodes:
            st.info("No source chunks were returned.")
            return

        for idx, source_node in enumerate(source_nodes, start=1):
            node = source_node.node
            metadata = node.metadata if node else {}
            source_file = metadata.get("source_file", "unknown")
            ticker = metadata.get("ticker", "N/A")
            year = metadata.get("year", "N/A")
            score = format_score(source_node.score)

            st.markdown(f"**Chunk {idx}**")
            st.write(f"Source file: {source_file}")
            st.write(f"Ticker: {ticker} | Year: {year} | Similarity score: {score}")
            chunk_text = node.get_content(metadata_mode="none") if node else ""
            st.code(chunk_text)


if __name__ == "__main__":
    main()
