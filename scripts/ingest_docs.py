"""Ingest unstructured IPL documents into a FAISS vector index with metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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


def _as_bool(value: str) -> bool:
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _embed_chunks(chunks: list[str], model_name: str, batch_size: int = 100) -> list[list[float]]:
	if not chunks:
		raise ValueError("No chunks to embed.")

	embedder = SentenceTransformer(model_name)
	vectors: list[list[float]] = []

	for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks", unit="batch"):
		batch = chunks[start : start + batch_size]
		batch_vecs = embedder.encode(
			batch,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		vectors.extend(batch_vecs.tolist())

	return vectors


def main(force: bool = False) -> None:
	"""Load files, chunk text, embed chunks, and persist FAISS index plus metadata."""
	load_dotenv()

	base_dir = Path(__file__).resolve().parents[1]
	raw_dir = base_dir / "data" / "raw"
	if not raw_dir.exists() or not raw_dir.is_dir():
		raise FileNotFoundError(
			f"Input directory not found: {raw_dir}. Create it and add .txt/.pdf files before ingestion."
		)

	vector_dir = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore/")).expanduser()
	if not vector_dir.is_absolute():
		vector_dir = base_dir / vector_dir

	force_reindex = force or _as_bool(os.getenv("FORCE_REINDEX", "false"))
	min_chunk_chars = int(os.getenv("MIN_CHUNK_CHARS", "80"))
	embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

	vector_dir.mkdir(parents=True, exist_ok=True)
	index_path = vector_dir / "index.faiss"
	metadata_path = vector_dir / "metadata.json"
	chunks_path = vector_dir / "chunks.json"

	if not force_reindex and index_path.exists() and metadata_path.exists() and chunks_path.exists():
		if not os.isatty(0):
			raise RuntimeError(
				"Existing vectorstore found in non-interactive mode. "
				"Use --force or set FORCE_REINDEX=true to overwrite."
			)
		reply = input("Existing vectorstore found. Re-index and overwrite? (y/N): ").strip().lower()
		if reply not in {"y", "yes"}:
			print("Skipped re-indexing.")
			return

	files = sorted(list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.pdf")))
	if not files:
		raise RuntimeError(
			f"No input files found in {raw_dir}. Add at least one .txt or .pdf file and retry."
		)

	print(f"Found {len(files)} files in {raw_dir}")

	splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
	all_chunks: list[str] = []
	chunk_meta: list[dict] = []
	files_processed = 0
	filtered_short_chunks = 0

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
			valid_chunks = 0
			for idx, chunk in enumerate(chunks):
				clean_chunk = chunk.strip()
				if len(clean_chunk) < min_chunk_chars:
					filtered_short_chunks += 1
					continue
				all_chunks.append(clean_chunk)
				chunk_meta.append(
					{
						"source_filename": file_path.name,
						"chunk_number": idx,
						"text_preview": clean_chunk[:180].replace("\n", " "),
					}
				)
				valid_chunks += 1
			files_processed += 1
			print(f"Loaded {file_path.name}: {valid_chunks} chunks kept ({len(chunks) - valid_chunks} filtered)")
		except Exception as exc:
			print(f"Error loading {file_path.name}: {exc}")

	if not all_chunks:
		raise RuntimeError(
			"No chunks remained after filtering. Reduce MIN_CHUNK_CHARS or add richer source documents."
		)

	vectors = _embed_chunks(all_chunks, embedding_model, batch_size=100)
	matrix = np.array(vectors, dtype=np.float32)
	if matrix.size == 0:
		raise RuntimeError("Embedding returned no vectors; ingestion aborted.")
	faiss.normalize_L2(matrix)

	index = faiss.IndexFlatIP(matrix.shape[1])
	index.add(matrix)

	faiss.write_index(index, str(index_path))

	metadata_map = {str(i): m for i, m in enumerate(chunk_meta)}
	with metadata_path.open("w", encoding="utf-8") as f:
		json.dump(
			{
				"embedding_model": embedding_model,
				"embedding_provider": "huggingface",
				"total_chunks": len(all_chunks),
				"filtered_short_chunks": filtered_short_chunks,
				"min_chunk_chars": min_chunk_chars,
				"chunk_metadata": metadata_map,
			},
			f,
			ensure_ascii=False,
			indent=2,
		)

	with chunks_path.open("w", encoding="utf-8") as f:
		json.dump(all_chunks, f, ensure_ascii=False)

	print(
		f"Summary: {files_processed} files processed, {len(all_chunks)} chunks created, "
		f"{filtered_short_chunks} short chunks filtered, vectorstore saved to {vector_dir}"
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Ingest raw docs into FAISS vectorstore.")
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing vectorstore without interactive confirmation.",
	)
	args = parser.parse_args()
	main(force=args.force)
