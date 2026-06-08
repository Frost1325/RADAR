"""
Build a vector index from a corpus of markdown documents.

Reads all .md files from a directory, embeds them with a SentenceTransformer
model, and saves:
  - embeddings.npy  : Document vectors (N, D)
  - docs.jsonl      : Document metadata (id, path, text)
  - meta.json       : Index configuration

Usage:
    python -m radar.index \
        --corpus_dir ./database \
        --index_dir ./database/index \
        --embed_model all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def build_embedder(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model from HuggingFace."""
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def build_corpus_from_markdown(corpus_dir: str, pattern: str = "*.md") -> List[Dict]:
    """
    Read all markdown files from a directory tree.

    Each file becomes one document.  Relative paths are stored for traceability.
    """
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.rglob(pattern))
    docs: List[Dict] = []

    print(f"Searching for markdown files in {corpus_dir}")
    for idx, f in enumerate(files):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "id": idx,
            "path": str(f.relative_to(corpus_dir)),
            "text": text,
        })

    print(f"Loaded {len(docs)} documents")
    return docs


def build_and_save_index(
    corpus_dir: str,
    index_dir: str,
    embed_model_name: str,
) -> None:
    """Build the vector index and persist all artifacts to disk."""
    index_dir_path = Path(index_dir)
    index_dir_path.mkdir(parents=True, exist_ok=True)

    docs = build_corpus_from_markdown(corpus_dir)
    if not docs:
        raise ValueError(f"No markdown files found in {corpus_dir}")

    embedder = build_embedder(embed_model_name)

    texts = [d["text"] for d in docs]
    print("Computing embeddings...")
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    # Save embeddings
    emb_path = index_dir_path / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path}")

    # Save document metadata
    docs_path = index_dir_path / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved document metadata to {docs_path}")

    # Save index configuration
    meta = {
        "embed_model_name": embed_model_name,
        "corpus_dir": str(Path(corpus_dir).resolve()),
        "num_docs": len(docs),
        "embeddings_shape": list(embeddings.shape),
    }
    meta_path = index_dir_path / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved index metadata to {meta_path}")


def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Load a previously built vector index from disk."""
    idx_p = Path(index_dir)
    emb = np.load(idx_p / "embeddings.npy")
    with (idx_p / "docs.jsonl").open("r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    meta = json.loads((idx_p / "meta.json").read_text(encoding="utf-8"))
    print(f"Loaded index: {len(docs)} docs, embedding shape {emb.shape}")
    return emb, docs, meta


def main():
    parser = argparse.ArgumentParser(
        description="Build a vector index from markdown documents"
    )
    parser.add_argument("--corpus_dir", required=True,
                        help="Directory containing markdown files")
    parser.add_argument("--index_dir", required=True,
                        help="Directory to save index files")
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model name")
    args = parser.parse_args()

    build_and_save_index(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        embed_model_name=args.embed_model,
    )


if __name__ == "__main__":
    main()
