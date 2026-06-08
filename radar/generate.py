"""
Answer generation using Radar retrieval results.

Given a set of pre-retrieved documents (from radar.retrieve), this module
generates answers using a large language model.  Supports both local
Transformers and remote vLLM backends.

Usage:
    # Local Transformers
    python -m radar.generate \
        --qa_path ./data/AeroQA_v2.jsonl \
        --retrieval_file ./results/retrieval.jsonl \
        --index_dir ./database/index \
        --gen_model Qwen/Qwen2.5-7B-Instruct \
        --output ./results/answers.jsonl

    # vLLM (high-throughput)
    python -m radar.generate \
        --qa_path ./data/AeroQA_v2.jsonl \
        --retrieval_file ./results/retrieval.jsonl \
        --index_dir ./database/index \
        --gen_model Qwen/Qwen2.5-7B-Instruct \
        --output ./results/answers.jsonl \
        --use_vllm --num_threads 4
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

from radar.vllm_client import build_vllm_client, VLLMClient


# ============================================================================
# Helpers
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_thinking_and_answer(text: str) -> Tuple[str, str]:
    """Extract <think>...</think> content and the final answer from model output."""
    if not text:
        return "", ""

    thinking = ""
    answer = text

    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        thinking = parts[0].replace('<think>', '').strip()
        answer = parts[1].strip()
    elif '<think>' in text:
        parts = text.split('<think>', 1)
        answer = parts[0].strip()
        thinking = parts[1].strip()

    return thinking, answer


# ============================================================================
# Prompt templates
# ============================================================================

RAG_PROMPT_TEMPLATES = {
    "choice": {
        "system": """You are a professional aviation knowledge assistant specialized in multiple-choice questions.
Your task is to analyze the provided documents and select the correct answer from the given options, then provide a brief explanation.

Guidelines:
1. Carefully read all documents and identify information relevant to the question.
2. Compare each option against the information in the documents.
3. Select only ONE correct answer (A, B, C, or D).
4. Base your choice strictly on the provided documents.
5. Output format requirement:
   - First, output ONLY the letter of the correct answer (A, B, C, or D).
   - Then, on a new line, provide a brief explanation (1-2 sentences) explaining why this answer is correct based on the documents.
   - Format: "Answer: [Letter]\\nExplanation: [1-2 sentences]"
   - Keep the explanation concise and reference the relevant document information.""",
        "user": "Documents:\n{context_block}\n\nQuestion: {question}\n\nOptions:\n{options_str}\n\nAnswer (format: Answer: [Letter]\\nExplanation: [1-2 sentences]): "
    },
    "boolean": {
        "system": """You are a professional aviation knowledge assistant specialized in true/false questions.
Your task is to analyze the provided documents and determine whether the statement is True or False, then provide a brief explanation.

Guidelines:
1. Carefully read the statement and verify it against the documents.
2. If the statement is supported, answer "True"; if contradicted, answer "False".
3. Output format requirement:
   - First, output ONLY \"True\" or \"False\" (capitalized).
   - Then, on a new line, provide a brief explanation (1-2 sentences) explaining why the statement is true or false based on the documents.
   - Format: "Answer: [True/False]\nExplanation: [1-2 sentences]"
   - Keep the explanation concise and reference the relevant document information.""",
        "user": "Documents:\n{context_block}\n\nStatement: {question}\n\nAnswer (format: Answer: [True/False]\\nExplanation: [1-2 sentences]): "
    },
    "extended": {
        "system": """You are a professional aviation knowledge assistant.
Your goal is to provide clear, accurate, safety-focused answers using the provided documents.

Follow these principles:
1. **Accuracy** – Ensure all explanations follow the standard aviation knowledge in the documents.
2. **Clarity** – Use concise, structured explanations.
3. **Deterministic** – Always answer in an instructional style suitable for pilot training.
4. **Grounded** – If the information is not in the documents, state so directly.""",
        "user": "Documents:\n{context_block}\n\nQuestion: {question}\n\nAnswer: "
    }
}


# ============================================================================
# Core logic
# ============================================================================

def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Load vector index and metadata from disk."""
    idx_p = Path(index_dir)
    emb = np.load(idx_p / "embeddings.npy")
    with (idx_p / "docs.jsonl").open("r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    meta = json.loads((idx_p / "meta.json").read_text(encoding="utf-8"))
    print(f"[*] Index loaded: {len(docs)} documents, dimension {emb.shape}")
    return emb, docs, meta


def load_retrieved_docs_from_file(
    retrieval_file_path: str, docs: List[Dict]
) -> Dict[int, List[Dict]]:
    """Load pre-retrieved results, mapping question IDs to document lists."""
    retrieved_map = {}
    retrieval_p = Path(retrieval_file_path)

    if not retrieval_p.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_p}")

    doc_map = {d.get("path", ""): d for d in docs}

    with retrieval_p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if "meta_info" in record:
                    continue

                qid = record.get("id")
                retrieved_files = record.get("retrieved_files", [])
                scores = record.get("scores", [])

                if qid is not None and retrieved_files:
                    retrieved_docs = []
                    for i, file_path in enumerate(retrieved_files):
                        if file_path in doc_map:
                            doc = doc_map[file_path].copy()
                            doc["score"] = scores[i] if i < len(scores) else 0.0
                            retrieved_docs.append(doc)

                    if retrieved_docs:
                        retrieved_map[qid] = retrieved_docs
            except json.JSONDecodeError:
                continue

    print(f"[*] Loaded pre-retrieved results for {len(retrieved_map)} questions")
    return retrieved_map


def build_generator(
    model_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    use_vllm: bool = False,
    vllm_api_url: str = "http://localhost:8000/v1",
    num_threads: int = 1,
):
    """Build inference backend (local Transformers or vLLM client)."""
    if use_vllm:
        print(f"[*] Initializing vLLM client: {vllm_api_url}, threads: {num_threads}")
        return build_vllm_client(
            api_url=vllm_api_url,
            model_name=model_name,
            max_workers=num_threads,
        )

    print(f"[*] Loading local Transformers model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )


def build_rag_prompt(
    question: str,
    contexts: List[Dict],
    question_type: str = "extended",
    options: Optional[List[str]] = None,
) -> List[Dict]:
    """Build a RAG message list from retrieved contexts."""
    context_strs = [
        f"[Doc {i+1} | score={c['score']:.4f}]\n{c['text']}"
        for i, c in enumerate(contexts)
    ]
    context_block = "\n\n---\n\n".join(context_strs)

    tpl = RAG_PROMPT_TEMPLATES.get(question_type, RAG_PROMPT_TEMPLATES["extended"])

    opt_str = ""
    if question_type == "choice" and options:
        opt_str = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
        )

    user_content = tpl["user"].format(
        context_block=context_block, question=question, options_str=opt_str
    )

    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user", "content": user_content},
    ]


# ============================================================================
# Main entry point
# ============================================================================

def generate_answers(
    qa_dataset_path: str,
    index_dir: str,
    gen_model: str,
    output_path: str,
    retrieval_file: str,
    top_k: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    use_vllm: bool = False,
    vllm_api_url: str = "http://localhost:8000/v1",
    local_files_only: bool = True,
    num_threads: int = 1,
    seed: int = None,
) -> None:
    """Generate answers for a QA dataset using pre-retrieved documents."""
    set_seed(seed)

    qa_p, output_p = Path(qa_dataset_path), Path(output_path)
    if not qa_p.exists():
        raise FileNotFoundError(f"Dataset not found: {qa_p}")
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load QA data
    print(f"[*] Loading QA dataset: {qa_p}")
    with qa_p.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"[+] Loaded {len(records)} records")

    # 2. Load index and pre-retrieved results
    embeddings, docs, meta = load_index(index_dir)
    retrieved_map = load_retrieved_docs_from_file(retrieval_file, docs)

    # 3. Initialize generator
    gen_pipe = build_generator(
        gen_model, max_new_tokens, temperature,
        use_vllm, vllm_api_url, num_threads=num_threads,
    )

    # 4. Generate answers
    print("[*] Starting answer generation...")
    with output_p.open("w", encoding="utf-8") as fout:
        # Write metadata header
        metadata = {
            "meta_info": "generation_config",
            "mode": "Radar-RAG",
            "model": gen_model,
            "use_vllm": use_vllm,
            "index_dir": index_dir,
            "retrieval_file": retrieval_file,
            "top_k": top_k,
        }
        fout.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        def process_single(rec):
            qid = rec.get("id")
            question = rec.get("question")
            if not question:
                return None

            retrieved = retrieved_map.get(qid)
            if not retrieved:
                return None

            retrieved = retrieved[:top_k]
            messages = build_rag_prompt(
                question, retrieved,
                rec.get("type", "extended"),
                rec.get("options"),
            )

            if isinstance(gen_pipe, VLLMClient):
                outputs = gen_pipe(
                    inputs=[messages],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed,
                )
                thinking, answer = extract_thinking_and_answer(
                    outputs[0]["generated_text"]
                )
            else:
                outputs = gen_pipe(messages)
                res = outputs[0]["generated_text"]
                if isinstance(res, list):
                    res_text = next(
                        (m["content"] for m in reversed(res)
                         if m["role"] == "assistant"),
                        res[-1]["content"],
                    )
                else:
                    res_text = (
                        res.split("Answer:")[-1].strip()
                        if "Answer:" in res
                        else res.strip()
                    )
                thinking, answer = extract_thinking_and_answer(res_text)

            return {
                "id": qid,
                "type": rec.get("type", "extended"),
                "answer": answer,
                "thinking": thinking,
                "retrieved_files": [d.get("path", "") for d in retrieved],
                "scores": [d.get("score", 0.0) for d in retrieved],
            }

        if num_threads > 1:
            print(f"[*] Concurrent inference with {num_threads} threads...")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(tqdm(
                    executor.map(process_single, records),
                    total=len(records),
                    desc="Radar-RAG Inference",
                ))
                for res in results:
                    if res:
                        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
        else:
            for rec in tqdm(records, desc="Radar-RAG Inference"):
                res = process_single(rec)
                if res:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"[+] Task completed! Results saved to: {output_p}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers using Radar retrieval results"
    )
    parser.add_argument("--qa_path", required=True,
                        help="Path to the QA dataset (JSONL)")
    parser.add_argument("--retrieval_file", required=True,
                        help="Path to the retrieval results file (JSONL)")
    parser.add_argument("--index_dir", required=True,
                        help="Pre-built vector index directory")
    parser.add_argument("--gen_model", required=True,
                        help="Generator model name or path")
    parser.add_argument("--output", required=True,
                        help="Output file path for generated answers")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Number of retrieved documents to use as context")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM backend instead of local Transformers")
    parser.add_argument("--vllm_api_url", default="http://localhost:8000/v1")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of concurrent threads (vLLM mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--allow_network", action="store_true",
                        help="Allow downloading models from the network")
    args = parser.parse_args()

    generate_answers(
        qa_dataset_path=args.qa_path,
        index_dir=args.index_dir,
        gen_model=args.gen_model,
        output_path=args.output,
        retrieval_file=args.retrieval_file,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_vllm=args.use_vllm,
        vllm_api_url=args.vllm_api_url,
        local_files_only=not args.allow_network,
        num_threads=args.num_threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
