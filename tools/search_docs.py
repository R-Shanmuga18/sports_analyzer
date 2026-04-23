"""Semantic retrieval tool over unstructured IPL season documents."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_CACHE: dict = {
    "loaded": False,
    "index": None,
    "chunks": None,       # list[str]  — one entry per chunk
    "metadata": None,     # dict[str, dict] — chunk_id → {source_filename, ...}
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "provider": "huggingface",
    "embedder": None,
    "index_mtime": 0.0,   # mtime when index was last loaded
}

# Minimum relevance score to include a chunk in results.
# For IndexFlatIP (inner product after L2 normalisation), scores are in [-1, 1].
# Anything below 0.20 is very weakly related and usually noise.
_MIN_RELEVANCE_SCORE = 0.20


def _vectorstore_paths(base_dir: Path) -> tuple[Path, Path, Path]:
    """Return (index_path, metadata_path, chunks_path) for the vectorstore."""
    vector_dir = Path(
        os.getenv("VECTORSTORE_PATH", "data/vectorstore/")
    ).expanduser()
    if not vector_dir.is_absolute():
        vector_dir = base_dir / vector_dir
    return (
        vector_dir / "index.faiss",
        vector_dir / "metadata.json",
        vector_dir / "chunks.json",
    )


def _index_is_stale(index_path: Path) -> bool:
    """Return True if the index file has been modified since we loaded it."""
    try:
        return index_path.stat().st_mtime > _CACHE["index_mtime"]
    except OSError:
        return False


def _load_vectorstore(force: bool = False) -> None:
    """
    Load FAISS index and associated metadata into the module-level cache.
    Automatically reloads if the index file has been updated since last load.
    Raises FileNotFoundError with a clear message if files are missing.
    Raises RuntimeError if the embedding provider is not 'huggingface'.
    """
    base_dir = Path(__file__).resolve().parents[1]
    index_path, metadata_path, chunks_path = _vectorstore_paths(base_dir)

    # If already loaded and index has not changed, skip
    if _CACHE["loaded"] and not force and not _index_is_stale(index_path):
        return

    # Existence checks with clear error messages
    missing = [p for p in (index_path, metadata_path, chunks_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Vectorstore files missing: {[str(p) for p in missing]}. "
            "Run scripts/ingest_docs.py first."
        )

    logger.info("Loading FAISS vectorstore from %s", index_path.parent)
    _CACHE["index"] = faiss.read_index(str(index_path))
    _CACHE["chunks"] = json.loads(chunks_path.read_text(encoding="utf-8"))

    metadata_obj = json.loads(metadata_path.read_text(encoding="utf-8"))
    _CACHE["metadata"] = metadata_obj.get("chunk_metadata", {})
    _CACHE["model"] = metadata_obj.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    provider = metadata_obj.get("embedding_provider", "unknown")

    # Hard check: we only support HuggingFace sentence-transformers embeddings.
    if provider != "huggingface":
        raise RuntimeError(
            f"Unsupported embedding provider '{provider}'. "
            "Only 'huggingface' embeddings are supported. "
            "Re-run scripts/ingest_docs.py to rebuild the vectorstore."
        )

    _CACHE["provider"] = provider
    _CACHE["embedder"] = None

    _CACHE["loaded"] = True
    try:
        _CACHE["index_mtime"] = index_path.stat().st_mtime
    except OSError:
        _CACHE["index_mtime"] = time.time()

    logger.info(
        "Vectorstore loaded: %d chunks, model=%s",
        len(_CACHE["chunks"]),
        _CACHE["model"],
    )


def _embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string using the same HuggingFace model used during ingestion.
    Returns a (1, dim) float32 array that is L2-normalised, ready for FAISS search.
    """
    from sentence_transformers import SentenceTransformer  # lazy import — only needed here

    model_name = _CACHE["model"]
    if _CACHE["embedder"] is None:
        _CACHE["embedder"] = SentenceTransformer(model_name)

    vec = _CACHE["embedder"].encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arr = np.array(vec, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def search_docs(query: str, top_k: int = 3) -> list[dict]:
    """
    Semantic search over IPL season review documents.

    Use this tool when the question asks about:
    - Match narratives, player performances described in text
    - Team strategies, tournament storylines, notable events
    - Reasons behind outcomes, match reports
    - Any question whose answer would appear in a written article

    Do NOT use this tool for:
    - Specific statistics or numbers (use query_data instead)
    - Recent news or events after 2024 (use web_search instead)

    Args:
        query: Natural language search query describing what you are looking for.
               Be specific — e.g. "KKR batting collapse 2024 final" not "KKR".
        top_k: Number of chunks to return (default 3, max 5)

    Returns:
        List of dicts with keys: text, source, chunk_id, relevance_score
        If nothing relevant is found, returns a single-item list with a
        "No relevant documents found" message so the agent knows to try
        a different tool rather than answering from nothing.
    """
    if not query or not query.strip():
        return [
            {
                "text": "Empty query — please provide a specific search phrase.",
                "source": "none",
                "chunk_id": -1,
                "relevance_score": 0.0,
            }
        ]

    try:
        _load_vectorstore()
    except FileNotFoundError as exc:
        logger.error("Vectorstore not available: %s", exc)
        return [
            {
                "text": str(exc),
                "source": "none",
                "chunk_id": -1,
                "relevance_score": 0.0,
            }
        ]
    except RuntimeError as exc:
        logger.error("Vectorstore config error: %s", exc)
        return [
            {
                "text": str(exc),
                "source": "none",
                "chunk_id": -1,
                "relevance_score": 0.0,
            }
        ]

    index = _CACHE["index"]
    chunks: list[str] = _CACHE["chunks"] or []
    metadata: dict = _CACHE["metadata"] or {}

    if index is None or not chunks:
        return [
            {
                "text": "No documents have been indexed. Run scripts/ingest_docs.py first.",
                "source": "none",
                "chunk_id": -1,
                "relevance_score": 0.0,
            }
        ]

    try:
        # Cap top_k to avoid requesting more than available chunks
        k = max(1, min(top_k, len(chunks), 5))
        query_vec = _embed_query(query)
        scores, indices = index.search(query_vec, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue  # FAISS can return -1 for padding
            # Filter out weakly relevant chunks — they add noise to the LLM context
            if float(score) < _MIN_RELEVANCE_SCORE:
                logger.debug(
                    "Chunk %d filtered out — score %.3f below threshold %.2f",
                    idx, score, _MIN_RELEVANCE_SCORE,
                )
                continue
            meta = metadata.get(str(idx), {})
            results.append(
                {
                    "text": chunks[idx],
                    "source": meta.get("source_filename", "unknown"),
                    "chunk_id": int(idx),
                    "relevance_score": round(float(score), 4),
                }
            )

        # Sort by descending relevance score (highest = most similar)
        results.sort(key=lambda r: r["relevance_score"], reverse=True)

        if not results:
            return [
                {
                    "text": (
                        "No relevant documents found for this query. "
                        "The documents may not cover this topic — try query_data or web_search."
                    ),
                    "source": "none",
                    "chunk_id": -1,
                    "relevance_score": 0.0,
                }
            ]

        return results

    except Exception as exc:
        logger.error("FAISS search failed: %s", exc)
        return [
            {
                "text": f"Document search failed: {exc}",
                "source": "none",
                "chunk_id": -1,
                "relevance_score": 0.0,
            }
        ]