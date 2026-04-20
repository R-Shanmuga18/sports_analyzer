"""Ingest unstructured IPL documents into a FAISS vector index with metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def _load_text_file(path: Path) -> str:
	return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf_file(path: Path) -> str:
	try:
		from pypdf import PdfReader
	except Exception as exc:
		raise RuntimeError("pypdf is required to ingest PDF files") from exc

	reader = PdfReader(str(path))
	pages: list[str] = []
	for page in reader.pages:
		pages.append(page.extract_text() or "")
	return "\n".join(pages)


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


def _embed_chunks(chunks: Iterable[str], model_name: str) -> tuple[list[list[float]], str]:
	texts = list(chunks)
	try:
		emb = OpenAIEmbeddings(model=model_name)
		vectors = emb.embed_documents(texts)
		return vectors, "openai"
	except Exception as exc:
		print(f"Warning: OpenAI embedding failed ({exc}). Falling back to local hash embedding.")
		vectors = [_hash_embed(t) for t in texts]
		return vectors, "hash"


def main() -> None:
	"""Load files, chunk text, embed chunks, and persist FAISS index plus metadata."""
	load_dotenv()

	base_dir = Path(__file__).resolve().parents[1]
	raw_dir = base_dir / "data" / "raw"
	vector_dir = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore/")).expanduser()
	if not vector_dir.is_absolute():
		vector_dir = base_dir / vector_dir

	vector_dir.mkdir(parents=True, exist_ok=True)
	index_path = vector_dir / "index.faiss"
	metadata_path = vector_dir / "metadata.json"
	chunks_path = vector_dir / "chunks.json"

	if index_path.exists() and metadata_path.exists() and chunks_path.exists():
		reply = input("Existing vectorstore found. Re-index and overwrite? (y/N): ").strip().lower()
		if reply not in {"y", "yes"}:
			print("Skipped re-indexing.")
			return

	files = sorted(list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.pdf")))
	print(f"Found {len(files)} files in {raw_dir}")

	splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
	all_chunks: list[str] = []
	chunk_meta: list[dict] = []
	files_processed = 0

	for file_path in files:
		try:
			if file_path.suffix.lower() == ".txt":
				text = _load_text_file(file_path)
			elif file_path.suffix.lower() == ".pdf":
				text = _load_pdf_file(file_path)
			else:
				continue

			if not text.strip():
				print(f"Skipping empty file: {file_path.name}")
				continue

			chunks = splitter.split_text(text)
			for idx, chunk in enumerate(chunks):
				all_chunks.append(chunk)
				chunk_meta.append(
					{
						"source_filename": file_path.name,
						"chunk_number": idx,
						"text_preview": chunk[:180].replace("\n", " "),
					}
				)
			files_processed += 1
			print(f"Loaded {file_path.name}: {len(chunks)} chunks")
		except Exception as exc:
			print(f"Error loading {file_path.name}: {exc}")

	if not all_chunks:
		print("No chunks generated. Nothing to index.")
		return

	model_name = "text-embedding-3-small"
	vectors, provider = _embed_chunks(all_chunks, model_name)
	matrix = np.array(vectors, dtype=np.float32)
	faiss.normalize_L2(matrix)

	index = faiss.IndexFlatIP(matrix.shape[1])
	index.add(matrix)

	faiss.write_index(index, str(index_path))

	metadata_map = {str(i): m for i, m in enumerate(chunk_meta)}
	with metadata_path.open("w", encoding="utf-8") as f:
		json.dump(
			{
				"embedding_model": model_name,
				"embedding_provider": provider,
				"total_chunks": len(all_chunks),
				"chunk_metadata": metadata_map,
			},
			f,
			indent=2,
		)

	with chunks_path.open("w", encoding="utf-8") as f:
		json.dump(all_chunks, f)

	print(
		f"Summary: {files_processed} files processed, {len(all_chunks)} chunks created, "
		f"vectorstore saved to {vector_dir}"
	)


if __name__ == "__main__":
	main()
