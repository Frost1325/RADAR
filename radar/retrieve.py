"""
Radar Retrieval: Vector Recall → Cross-Encoder Reranking → Adaptive Thresholding.

This module implements the core retrieval pipeline of the Radar system:
1. **Vector Recall**: Cosine-similarity search over a pre-built embedding index
2. **Cross-Encoder Reranking**: A fine-tuned reranker re-scores the top-N candidates
3. **Adaptive Thresholding**: Documents with rerank scores below a fraction of the
   top score are dropped, reducing noise in the final context.

Usage:
    python -m radar.retrieve \
        --input ./data/AeroQA_v2.jsonl \
        --output ./results/retrieval.jsonl \
        --index_dir ./database/index \
        --rerank_model path/to/reranker \
        --top_k 4 --threshold 0.85
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Load vector index and metadata from disk."""
    idx_p = Path(index_dir)
    emb = np.load(idx_p / "embeddings.npy")
    with (idx_p / "docs.jsonl").open("r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    meta = json.loads((idx_p / "meta.json").read_text(encoding="utf-8"))
    return emb, docs, meta


# ---------------------------------------------------------------------------
# Vector retrieval (cosine similarity)
# ---------------------------------------------------------------------------

def retrieve_vector(
    question: str,
    embeddings: np.ndarray,
    docs: List[Dict],
    embedder: SentenceTransformer,
    top_k: int = 10,
) -> List[Dict]:
    """Vector retrieval using cosine similarity."""
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]

    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    e_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(e_norm, q_norm)

    topk_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in topk_idx:
        d = docs[int(idx)].copy()
        d["score"] = float(sims[idx])
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Radar Retriever
# ---------------------------------------------------------------------------

class RadarRetriever:
    """
    Radar retrieval pipeline: Vector Search → Rerank → Adaptive Threshold.

    Parameters
    ----------
    index_dir : str
        Path to the pre-built vector index directory.
    rerank_model_path : str
        Path or HuggingFace ID of the cross-encoder reranker model.
    local_files_only : bool
        If True, only use locally cached model files.
    """

    def __init__(
        self,
        index_dir: str,
        rerank_model_path: str,
        local_files_only: bool = True,
    ):
        # Load index
        self.embeddings, self.docs, self.meta = load_index(index_dir)

        # Load embedding model (from index metadata)
        embed_model_name = self.meta.get("embed_model_name")
        print(f"Loading embedding model: {embed_model_name}")
        self.embedder = SentenceTransformer(
            embed_model_name, local_files_only=local_files_only
        )

        # Load reranker
        print(f"Loading reranker model: {rerank_model_path}")
        self.reranker = CrossEncoder(
            rerank_model_path, local_files_only=local_files_only
        )

    def search(
        self,
        question: str,
        top_k: int = 4,
        rerank_top_n: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Execute the full Radar retrieval pipeline.

        Parameters
        ----------
        question : str
            The query / question string.
        top_k : int
            Number of documents to return after reranking.
        rerank_top_n : int
            Number of candidates to recall from vector search before reranking.
        threshold : float or None
            Adaptive threshold factor.  Documents with rerank_score below
            ``threshold * max(rerank_score)`` are filtered out.

        Returns
        -------
        List[Dict]
            Retrieved documents with 'rerank_score' fields.
        """
        # Stage 1: Vector recall
        initial_results = retrieve_vector(
            question, self.embeddings, self.docs, self.embedder,
            top_k=rerank_top_n
        )

        # Stage 2: Cross-Encoder reranking
        model_inputs = [[question, d["text"]] for d in initial_results]
        scores = self.reranker.predict(model_inputs)

        for i, score in enumerate(scores):
            initial_results[i]["rerank_score"] = float(score)

        reranked = sorted(
            initial_results, key=lambda x: x["rerank_score"], reverse=True
        )
        final_results = reranked[:top_k]

        # Stage 3: Adaptive thresholding
        if threshold is not None and len(final_results) > 0:
            top_score = final_results[0]["rerank_score"]
            cutoff = top_score * threshold
            final_results = [
                d for d in final_results if d["rerank_score"] >= cutoff
            ]

        return final_results


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_search(
    input_path: str,
    output_path: str,
    index_dir: str,
    rerank_model: str,
    top_k: int = 4,
    rerank_top_n: int = 10,
    threshold: Optional[float] = None,
    local_files_only: bool = True,
):
    """Run Radar retrieval on every question in a JSONL dataset."""
    retriever = RadarRetriever(
        index_dir=index_dir,
        rerank_model_path=rerank_model,
        local_files_only=local_files_only,
    )

    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_p}")
    records = []
    with input_p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Processing {len(records)} questions...")
    with output_p.open("w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="Retrieving"):
            qid = rec.get("id")
            question = rec.get("question")
            if not question:
                continue

            results = retriever.search(
                question,
                top_k=top_k,
                rerank_top_n=rerank_top_n,
                threshold=threshold,
            )

            retrieved_files = [d.get("path", "") for d in results]
            scores = [float(d.get("rerank_score", 0.0)) for d in results]

            out_rec = {
                "id": qid,
                "question": question,
                "retrieved_files": retrieved_files,
                "scores": scores,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"Finished! Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Radar Retrieval: Vector → Rerank → Adaptive Threshold"
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL file with questions")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for retrieval results")
    parser.add_argument("--index_dir", required=True,
                        help="Pre-built vector index directory")
    parser.add_argument("--rerank_model", required=True,
                        help="Cross-Encoder reranker model path or name")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Number of documents to return (default: 4)")
    parser.add_argument("--rerank_top_n", type=int, default=10,
                        help="Number of candidates for reranking (default: 10)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Adaptive threshold factor (e.g., 0.85)")
    parser.add_argument("--allow_network", action="store_true",
                        help="Allow downloading models from the network")
    args = parser.parse_args()

    batch_search(
        input_path=args.input,
        output_path=args.output,
        index_dir=args.index_dir,
        rerank_model=args.rerank_model,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        threshold=args.threshold,
        local_files_only=not args.allow_network,
    )


if __name__ == "__main__":
    main()
