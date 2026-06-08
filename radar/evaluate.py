"""
Evaluation script for Radar retrieval + generation outputs.

Computes:
  - Retrieval metrics: MRR, Hit@K, Recall@K, Precision, F1, Noise Ratio
  - Generation metrics: Overall F1, BERTScore F1, Objective Accuracy

For choice/boolean questions, the script automatically splits model outputs
into answer_option and explanation for fine-grained evaluation.

Usage:
    python -m radar.evaluate \
        --gold ./data/AeroQA_v2.jsonl \
        --pred ./results/answers.jsonl \
        --output ./results/metrics.json
"""

import json
import re
import string
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
from bert_score import score as bert_score


# ============================================================================
# Text normalisation
# ============================================================================

def normalize_answer(s: str) -> str:
    """Normalize a string for F1 comparison."""
    if s is None:
        return ""

    def lower(text):
        return text.lower()

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def _normalize_choice_or_bool_answer(ans: str, qtype: str) -> str:
    """Extract the canonical answer label from model output."""
    if ans is None:
        return ""
    text = str(ans).strip()
    if not text:
        return ""

    if qtype == "choice":
        if text.upper() in ["A", "B", "C", "D"]:
            return text.upper()
        m = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return text

    if qtype == "boolean":
        m = re.match(r"^(true|false)\b", text, flags=re.IGNORECASE)
        if m:
            return "True" if m.group(1).lower() == "true" else "False"
        m = re.search(r"\b(true|false)\b", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        return text

    return text


def split_answer_and_explanation(text: str, qtype: str) -> Tuple[str, str]:
    """Split model output into answer and explanation parts."""
    if not text:
        return "", ""

    # Remove thinking process
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    if qtype not in ["choice", "boolean"]:
        return text, ""

    # Split on "Explanation:"
    parts = re.split(r'Explanation:', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        ans_part = parts[0].strip()
        expl_part = parts[1].strip()
    else:
        ans_part = text
        expl_part = ""

    # Clean answer part (e.g., "Answer: A" → "A")
    ans_part = re.sub(
        r'^(?:Answer|Choice|Result):\s*', '', ans_part, flags=re.IGNORECASE
    ).strip()

    return ans_part, expl_part


# ============================================================================
# Metrics
# ============================================================================

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_retrieval_metrics(
    gold_files: List[str], pred_files: List[str], k_list=(1, 2, 3, 4)
) -> Optional[Dict]:
    """Compute retrieval metrics for a single query."""
    if not gold_files:
        return None
    gold_set = set(gold_files)
    metrics = {}

    for k in k_list:
        top_k_pred = pred_files[:k]
        metrics[f"hit@{k}"] = 1 if any(f in gold_set for f in top_k_pred) else 0
        intersection = set(top_k_pred).intersection(gold_set)
        metrics[f"recall@{k}"] = (
            len(intersection) / len(gold_set) if gold_set else 0
        )

    # MRR
    mrr = 0
    for i, f in enumerate(pred_files):
        if f in gold_set:
            mrr = 1 / (i + 1)
            break
    metrics["mrr"] = mrr

    actual_num = len(pred_files)
    if actual_num > 0:
        actual_inter = set(pred_files).intersection(gold_set)
        precision = len(actual_inter) / actual_num
        recall = len(actual_inter) / len(gold_set) if gold_set else 0
        metrics["precision"] = precision
        metrics["f1"] = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        metrics["noise_ratio"] = 1 - precision
    else:
        metrics["precision"] = metrics["f1"] = metrics["noise_ratio"] = 0

    return metrics


# ============================================================================
# Main evaluation
# ============================================================================

def run_eval(
    gold_path: str,
    pred_path: str,
    output_path: Optional[str] = None,
):
    """Run full evaluation over retrieval and generation outputs."""
    # Load gold data
    gold_data = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gold_data[obj["id"]] = obj

    # Load and split predictions
    pred_data = {}
    split_records = []
    pred_p = Path(pred_path)
    split_pred_path = pred_p.parent / f"{pred_p.stem}_split.jsonl"

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "id" not in obj:
                split_records.append(obj)
                continue

            qid = obj["id"]
            qtype = gold_data.get(qid, {}).get("type", "extended")

            ans_raw = obj.get("answer", "")
            ans_part, expl_part = split_answer_and_explanation(ans_raw, qtype)

            if qtype in ["choice", "boolean"]:
                obj["answer_option"] = _normalize_choice_or_bool_answer(
                    ans_part, qtype
                )
                obj["explanation"] = expl_part

            pred_data[qid] = obj
            split_records.append(obj)

    # Save split predictions
    with open(split_pred_path, "w", encoding="utf-8") as f:
        for rec in split_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[+] Split predictions saved to: {split_pred_path}")

    # Accumulate metrics
    retrieval_results = []
    generation_results = []
    preds_for_bert = []
    golds_for_bert = []
    obj_correct, obj_total = 0, 0

    for qid, g in gold_data.items():
        if qid not in pred_data:
            continue
        p = pred_data[qid]
        qtype = g.get("type", "extended")

        # --- Retrieval metrics ---
        gold_files = g.get("source_file", []) or g.get("source", {}).get("file", [])
        if isinstance(gold_files, str):
            gold_files = [gold_files]
        pred_files = p.get("retrieved_files", [])
        ret_m = calculate_retrieval_metrics(gold_files, pred_files)
        if ret_m:
            retrieval_results.append(ret_m)

        # --- Generation metrics ---
        if qtype in ["choice", "boolean"]:
            p_ans = p.get("answer_option")
            g_ans = g.get("answer_option") or g.get("answer")

            is_correct = 1 if normalize_answer(p_ans) == normalize_answer(g_ans) else 0
            obj_correct += is_correct
            obj_total += 1

            g_text = g.get("explanation", "")
            p_text = p.get("explanation", "") or p.get("answer", "")
        else:
            g_text = g.get("answer", "")
            p_text = p.get("answer", "")

        generation_results.append(f1_score(p_text, g_text))
        preds_for_bert.append(p_text if p_text else "None")
        golds_for_bert.append(g_text if g_text else "None")

    # --- Summarise ---
    summary = {}

    print("\n" + "=" * 40)
    print("RETRIEVAL METRICS")
    print("-" * 40)
    if retrieval_results:
        summary["retrieval"] = {
            k: float(np.mean([r[k] for r in retrieval_results]))
            for k in retrieval_results[0].keys()
        }
        for k, v in summary["retrieval"].items():
            print(f"  {k:15s}: {v:.4f}")

    print("\n" + "=" * 40)
    print("GENERATION METRICS")
    print("-" * 40)
    if generation_results:
        P, R, F = bert_score(
            preds_for_bert, golds_for_bert,
            lang="en", rescale_with_baseline=True,
        )
        summary["generation"] = {
            "overall_f1": float(np.mean(generation_results)),
            "bert_f1": float(F.mean().item()),
            "obj_acc": (
                float(obj_correct / obj_total) if obj_total > 0 else 0.0
            ),
        }
        print(f"  {'Overall F1':15s}: {summary['generation']['overall_f1']:.4f}")
        print(f"  {'BERTScore F1':15s}: {summary['generation']['bert_f1']:.4f}")
        if obj_total > 0:
            print(f"  {'Objective Acc':15s}: {summary['generation']['obj_acc']:.4f}"
                  f" ({obj_correct}/{obj_total})")
    print("=" * 40 + "\n")

    # Save summary
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[+] Evaluation summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Radar retrieval + generation outputs"
    )
    parser.add_argument("--gold", required=True,
                        help="Path to gold-standard JSONL dataset")
    parser.add_argument("--pred", required=True,
                        help="Path to prediction JSONL file")
    parser.add_argument("--output",
                        help="Path to save evaluation summary (JSON)")
    args = parser.parse_args()
    run_eval(args.gold, args.pred, args.output)


if __name__ == "__main__":
    main()
