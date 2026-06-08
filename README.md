# Radar: Retrieval-Augmented Document Analysis for Aviation Regulations

Radar is a domain-specific Retrieval-Augmented Generation (RAG) system for
aviation knowledge question answering.  It implements a three-stage retrieval
pipeline — **Vector Recall → Cross-Encoder Reranking → Adaptive Thresholding** —
followed by LLM-based answer generation.

---

## Repository Contents

```
radar-release/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── run_pipeline.sh                 # End-to-end pipeline script
│
├── radar/                          # Core Python package
│   ├── __init__.py
│   ├── index.py                    # Build vector index from FAA documents
│   ├── retrieve.py                 # Radar retrieval (vector → rerank → threshold)
│   ├── generate.py                 # Answer generation via LLM + retrieved docs
│   ├── evaluate.py                 # Retrieval + generation evaluation
│   └── vllm_client.py              # vLLM API client (optional, for high throughput)
│
├── dataset.tar.gz                  # AeroQA + AeroQA Unanswerable (compressed)
└── knowledgebase.tar.gz            # FAA knowledge base (compressed, 7,612 chunks)
```

After extraction:
```
├── dataset/                        # Evaluation datasets
│   ├── AeroQA.jsonl                # AeroQA (2,588 questions)
│   └── AeroQA_unanswerable.jsonl   # Unanswerable subset (150 questions)
│
└── knowledgebase/                  # FAA knowledge base (7,612 chunked documents)
    ├── AFH_chunk_*.md              # Airplane Flying Handbook
    ├── AIM_chunk_*.md              # Aeronautical Information Manual
    ├── IPH_chunk_*.md              # Instrument Procedures Handbook
    ├── PHAK_chunk_*.md             # Pilot's Handbook of Aeronautical Knowledge
    └── ...
```

---

## Quick Start

### 1. Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

### 2. Extract Data

```bash
tar xzf dataset.tar.gz
tar xzf knowledgebase.tar.gz
```

### 3. Build the Index

```bash
python -m radar.index \
    --corpus_dir ./knowledgebase \
    --index_dir ./knowledgebase/index \
    --embed_model all-MiniLM-L6-v2
```

### 4. Run Retrieval

```bash
python -m radar.retrieve \
    --input ./dataset/AeroQA.jsonl \
    --output ./results/retrieval.jsonl \
    --index_dir ./knowledgebase/index \
    --rerank_model path/to/your/reranker \
    --top_k 4 \
    --threshold 0.85
```

> **Note on the reranker model:** We use a fine-tuned cross-encoder reranker
> trained with a margin-MSE objective.  The model weights will be released
> separately.  For experimentation, you can use any CrossEncoder-compatible
> model such as `BAAI/bge-reranker-v2-m3`.

### 5. Generate Answers

**Local Transformers:**
```bash
python -m radar.generate \
    --qa_path ./dataset/AeroQA.jsonl \
    --retrieval_file ./results/retrieval.jsonl \
    --index_dir ./knowledgebase/index \
    --gen_model Qwen/Qwen2.5-7B-Instruct \
    --output ./results/answers.jsonl
```

**vLLM (high-throughput):**
```bash
python -m radar.generate \
    --qa_path ./dataset/AeroQA.jsonl \
    --retrieval_file ./results/retrieval.jsonl \
    --index_dir ./knowledgebase/index \
    --gen_model Qwen/Qwen2.5-7B-Instruct \
    --output ./results/answers.jsonl \
    --use_vllm --vllm_api_url http://localhost:8000/v1 --num_threads 4
```

### 6. Evaluate

```bash
python -m radar.evaluate \
    --gold ./dataset/AeroQA.jsonl \
    --pred ./results/answers.jsonl \
    --output ./results/metrics.json
```

### One-Command Pipeline

```bash
bash run_pipeline.sh path/to/reranker Qwen/Qwen2.5-7B-Instruct
```

---

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@inproceedings{...,
  title     = {[Paper Title]},
  author    = {[Authors]},
  booktitle = {[Venue]},
  year      = {[Year]},
}
```

---

## License

This project is released under the [MIT License](LICENSE).  The FAA documents
in `knowledgebase/` are public-domain U.S. government publications.
