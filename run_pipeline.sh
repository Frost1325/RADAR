#!/bin/bash
#
# Radar: End-to-end RAG pipeline for aviation question answering.
#
# Usage:
#   bash run_pipeline.sh <rerank_model_path> <gen_model_name>
#
# Example:
#   bash run_pipeline.sh BAAI/bge-reranker-v2-m3 Qwen/Qwen2.5-7B-Instruct
#

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <rerank_model_path> <gen_model_name>"
    echo "Example: $0 BAAI/bge-reranker-v2-m3 Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

RERANK_MODEL="$1"
GEN_MODEL="$2"

# Optional: uncomment to use vLLM
# USE_VLLM="--use_vllm --num_threads 4"
USE_VLLM=""

echo "=============================================="
echo "  Radar RAG Pipeline"
echo "  Reranker: $RERANK_MODEL"
echo "  Generator: $GEN_MODEL"
echo "  $(date)"
echo "=============================================="

# Step 1: Build index (skip if already exists)
if [ ! -f ./knowledgebase/index/embeddings.npy ]; then
    echo ""
    echo "[Step 1/4] Building vector index..."
    python -m radar.index \
        --corpus_dir ./knowledgebase \
        --index_dir ./knowledgebase/index \
        --embed_model all-MiniLM-L6-v2
else
    echo ""
    echo "[Step 1/4] Index already exists, skipping."
fi

# Step 2: Radar retrieval
echo ""
echo "[Step 2/4] Radar Retrieval (vector → rerank → threshold)..."
python -m radar.retrieve \
    --input ./dataset/AeroQA.jsonl \
    --output ./results/retrieval.jsonl \
    --index_dir ./knowledgebase/index \
    --rerank_model "$RERANK_MODEL" \
    --top_k 4 \
    --rerank_top_n 10 \
    --threshold 0.85 \
    --allow_network

# Step 3: Answer generation
echo ""
echo "[Step 3/4] Answer Generation..."
python -m radar.generate \
    --qa_path ./dataset/AeroQA.jsonl \
    --retrieval_file ./results/retrieval.jsonl \
    --index_dir ./knowledgebase/index \
    --gen_model "$GEN_MODEL" \
    --output ./results/answers.jsonl \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --allow_network \
    $USE_VLLM

# Step 4: Evaluation
echo ""
echo "[Step 4/4] Evaluation..."
python -m radar.evaluate \
    --gold ./dataset/AeroQA.jsonl \
    --pred ./results/answers.jsonl \
    --output ./results/metrics.json

echo ""
echo "=============================================="
echo "  Pipeline Complete!"
echo "  Answers:   ./results/answers.jsonl"
echo "  Metrics:   ./results/metrics.json"
echo "=============================================="
