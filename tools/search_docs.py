"""Semantic retrieval tool over unstructured IPL season documents."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

_CACHE: dict = {
    "loaded": False,
    "index": None,
    "chunks": None,
    "metadata": None,
    "provider": None,
    "model": "text-embedding-3-small",
}


def _hash_embed(text: str, dim: int = 1536) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vec.tolist()
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _load_vectorstore() -> None:
    if _CACHE["loaded"]:
        return

    load_dotenv()
    base_dir = Path(__file__).resolve().parents[1]
    vector_dir = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore/")).expanduser()
    if not vector_dir.is_absolute():
        vector_dir = base_dir / vector_dir

    index_path = vector_dir / "index.faiss"
    metadata_path = vector_dir / "metadata.json"
    chunks_path = vector_dir / "chunks.json"

    if not (index_path.exists() and metadata_path.exists() and chunks_path.exists()):
        raise FileNotFoundError("Vectorstore files not found. Run scripts/ingest_docs.py first.")

    _CACHE["index"] = faiss.read_index(str(index_path))
    _CACHE["chunks"] = json.loads(chunks_path.read_text(encoding="utf-8"))

    metadata_obj = json.loads(metadata_path.read_text(encoding="utf-8"))
    _CACHE["metadata"] = metadata_obj.get("chunk_metadata", {})
    _CACHE["provider"] = metadata_obj.get("embedding_provider", "openai")
    _CACHE["model"] = metadata_obj.get("embedding_model", "text-embedding-3-small")
    _CACHE["loaded"] = True


def _embed_query(query: str) -> np.ndarray:
    model_name = _CACHE["model"]
    provider = _CACHE["provider"]
    if provider == "openai":
        emb = OpenAIEmbeddings(model=model_name)
        vec = emb.embed_query(query)
    else:
        vec = _hash_embed(query)
    arr = np.array([vec], dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def search_docs(query: str, top_k: int = 3) -> list[dict]:
    """
    Semantic search over IPL season review documents.

    Use this tool when the question asks about:
    - Match narratives, player performances described in text
    - Team strategies, tournament storylines, notable events
    - Reasons behind outcomes, management commentary
    - Any question whose answer would appear in a written report or article

    Do NOT use this tool for:
    - Specific statistics or numbers (use query_data instead)
    - Recent news or events after 2024 (use web_search instead)

    Args:
        query: Natural language search query (what you are looking for)
        top_k: Number of chunks to return (default 3)

    Returns:
        List of dicts, each with keys: text, source, chunk_id, relevance_score
    """
    try:
        _load_vectorstore()
        index = _CACHE["index"]
        chunks = _CACHE["chunks"]
        metadata = _CACHE["metadata"]

        if index is None or not chunks:
            return [
                {
                    "text": "No relevant documents found",
                    "source": "none",
                    "chunk_id": -1,
                    "relevance_score": 0.0,
                }
            ]

        k = max(1, min(top_k, len(chunks)))
        query_vec = _embed_query(query)
        scores, indices = index.search(query_vec, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            meta = metadata.get(str(idx), {})
            results.append(
                {
                    "text": chunks[idx],
                    "source": meta.get("source_filename", "unknown"),
                    "chunk_id": int(idx),
                    "relevance_score": float(score),
                }
            )

        if not results:
            return [
                {
                    "text": "No relevant documents found",
                    "source": "none",
                    "chunk_id": -1,
                    "relevance_score": 0.0,
                }
            ]

        return results
    except Exception as exc:
        return [
            {
                "text": f"No relevant documents found ({exc})",
                "source": "none",
                "chunk_id": -1,
                "relevance_score": 0.0,
            }
        ]
