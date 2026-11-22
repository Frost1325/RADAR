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

## AeroQA Dataset Construction

**AeroQA** is a specialized civil aviation RAG benchmark comprising **2,588 high-quality QA pairs** grounded in **7,612 chunked passages** from four authoritative Federal Aviation Administration (FAA) manuals—*Pilot's Handbook of Aeronautical Knowledge* (PHAK, `FAA-H-8083-25`, 2023), *Airplane Flying Handbook* (AFH, `FAA-H-8083-3`, 2021), *Aeronautical Information Manual* (AIM, 2024), and *Instrument Procedures Handbook* (IPH, `FAA-H-8083-16`, 2017)—supplemented by **150 out-of-corpus unanswerable samples** (`AeroQA_unanswerable.jsonl`) for evaluating abstention.

### Four-Stage Construction Pipeline

1. **Corpus Collection and Preprocessing**
   - Authoritative FAA training, operational, and regulatory manuals (PHAK, AFH, AIM, IPH) are parsed from complex PDF layouts using **DeepSeek OCR**, followed by systematic noise removal.
   - Cleaned documents are segmented with a sliding-window strategy into **256-token chunks** with a **64-token overlap**, yielding **7,612 chunks** that preserve precise semantic units and contextual continuity.

2. **Semantic Grouping**
   - Domain experts manually aggregate related chunks into self-contained context windows of **1 to 4 chunks**, ensuring logical coherence and sufficient evidence for complex question generation.

3. **QA Generation**
   - **GPT-5.4** generates three question types from each grouped context window under strict closed-context grounding constraints:
     - **Multiple-Choice**: paired with adversarial distractors to test discriminative robustness;
     - **True/False**: designed for factual verification;
     - **Long-form QA**: targeting evidence extraction and multi-step reasoning.
   - Ground-truth chunk-level references for each item originate directly from the grouped context window used during generation.

4. **Quality Filtering and Human Expert Review**
   - From over **3,000 raw generated samples**, each QA pair is evaluated by an **LLM-as-judge committee** (**ChatGPT**, **Gemini**, and **DeepSeek**) across **eight dimensions** on a 1–5 scale:
     1. *Factual Correctness*: whether the answer is factually accurate and free from incorrect statements;
     2. *Answer Completeness*: whether the answer covers all required key points, conditions, steps, or constraints;
     3. *Question Clarity*: whether the question is clear, unambiguous, and answerable;
     4. *Question Value*: whether the question has training/evaluation value and targets core aviation knowledge;
     5. *Reference Support*: whether the reference answer can be derived solely from the provided reference chunks;
     6. *Reference Precision*: whether the reference chunks are focused on the question without excessive redundancy, noise, or irrelevant context;
     7. *Distractor Quality* (multiple-choice only): whether options are plausible, discriminative, mutually exclusive, and not obviously unreasonable;
     8. *Difficulty*: the cognitive level and difficulty of the question.
   - **Screening and Human Adjudication Rules**:
     - Samples scoring **≤ 2** on any dimension are **discarded**.
     - Samples scoring **3** on any dimension undergo **mandatory human expert review**.
     - Only samples for which a **majority of judges assign ≥ 4 across all dimensions** are **automatically retained**.
     - Human expert review primarily resolves redundant reference passages (pruning extraneous chunks from the reference set) and weak distractors (rewriting insufficiently plausible multiple-choice options).
   - Across the raw candidate pool, the major flagged issue counts (non-exclusive) were: Noisy Reference (415), Too Easy (381), Reference Too Broad (106), Weak Distractors (99), Incomplete Answer (83), Trivial Question (77), Unsupported Answer (16), and Factual Error (9), resulting in **2,588 retained high-quality QA pairs**.

### Dataset Statistics

Average question (`Q-Len`), answer (`A-Len`), and reference context (`C-Len`) lengths are measured in words. Over 80% of samples exhibit less than 5% 3-gram question–context overlap, reflecting substantial semantic paraphrasing beyond literal keyword matching.

| Question Type | Count | Q-Len | A-Len | C-Len | 3-gram Overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multiple-Choice | 839 | 18.22 | 36.28 | 391.65 | 1.96% |
| True/False | 877 | 22.48 | 28.98 | 363.99 | 6.25% |
| Long-form QA | 872 | 22.67 | 142.35 | 413.67 | 1.81% |
| **Total** | **2,588** | **21.16** | **69.54** | **389.70** | **3.36%** |

### Topic Distribution

| Aviation Topic | Count | Share |
| --- | ---: | ---: |
| Aerodynamics & Flight Mechanics | 642 | 24.81% |
| Performance, Weight & Balance | 573 | 22.14% |
| Aircraft Structure & Systems | 439 | 16.96% |
| Flight Controls & Instruments | 375 | 14.49% |
| Weather & Environment | 197 | 7.61% |
| Human Factors & Decision-Making | 156 | 6.03% |
| Fundamentals & Introduction | 117 | 4.52% |
| Documentation & Regulations | 89 | 3.44% |
| **Total** | **2,588** | **100.00%** |

> **Note on naming disambiguation:** This AeroQA benchmark (2,588 chunk-grounded RAG QA pairs over four FAA instructional manuals) is distinct from the earlier namesake AeroQA dataset introduced with KITLM at ICON 2023 (Agarwal et al., 2023), which consists of ~34k synthetic closed-book single- and multi-hop questions derived from an NTSB aircraft accident-report knowledge graph.

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
