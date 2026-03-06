# Ukrainian Traffic Rules Assistant — Agentic RAG Chatbot

Agentic RAG chatbot built with LangGraph + local LLM (Qwen3.5-9B via llama.cpp) + Streamlit UI.
Domain: **Ukrainian Traffic Rules (ПДР)** — 34 sections, hundreds of numbered articles.
**Language: Ukrainian only.** The RAG knowledge base, the LLM prompts, and all user queries must be in Ukrainian.

---

## Problem & Objectives

**Relevance:** The Ukrainian Traffic Rules document is a dense, legally structured text. Locating a specific article requires reading through the full document manually.

**User need:** Drivers and learners need fast, citation-backed answers to specific rule queries — not a general summary, but the exact article number and its content.

**Why agentic RAG:**
- Simple queries — *"Яка максимальна швидкість у місті?" (What is the speed limit in urban areas?)* → fast RAG path, no unnecessary steps
- Complex queries — *"Порівняй правила зупинки та стоянки" (Compare stopping vs parking rules)* → decomposed into sub-queries, each answered independently
- Out-of-scope queries — *"Який штраф за проїзд на червоне?" (What is the fine for running a red light?)* → web search fallback, since fines are not in ПДР
- Calculation queries — *"Гальмівний шлях при 90 км/г?" (Stopping distance at 90 km/h?)* → deterministic tool, no LLM inference needed

Plain LLM hallucinates article numbers. RAG grounds answers in the actual document. The agentic layer routes each query to the cheapest correct path.

---

## System Architecture

```
User Query
    │
    ▼
[1] classify ──simple──────► [3] rag_node ──has answers──► [5] synthesize ──► Response
    │                              │                              │
  complex                     no answers                    iterations < 2
    │                              │                              │
    ▼                              ▼                              ▼
[2] decompose ────────────► [4] web_search ──────────────► [5] synthesize
    │
calculation──► [4] calculator_node ──────────────────────► [5] synthesize

RAG Subgraph (invoked inside rag_node, not counted in main graph):
    retrieve → generate
```

**Main graph nodes (6):** `classify`, `decompose`, `rag_node`, `calculator_node`, `web_search`, `synthesize`

**RAG subgraph:** `retrieve → generate` — compiled separately in `rag.py`, invoked as a single node

**Tools (2):**
- `traffic_calculator` — deterministic stopping distance / safe following distance formula (non-retrieval)
- `DuckDuckGo` — web search for queries outside ПДР scope (fines, licensing)

**State fields:** `query` · `classification` · `sub_queries` · `rag_answers` · `web_results` · `final_answer` · `iterations`

**Routing functions (3):**
- `route_classify` → one of: `rag_node`, `decompose`, `calculator_node`, `web_search`
- `route_rag` → `synthesize` if answers found, else `web_search`
- `route_synthesize` → loop back to `rag_node` or `END`

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Qwen3.5-9B via llama.cpp | Open-source, fits in 16 GB VRAM (~5.6 GB), OpenAI-compatible API |
| Rule-based chunking | Each ПДР article is a self-contained logical unit — better retrieval precision than fixed-size chunks |
| pgvector | Single PostgreSQL stack, no extra services |
| paraphrase-multilingual-MiniLM-L12-v2 | Supports Ukrainian; lightweight, runs locally without API |
| `max_tokens=64` for classify | Classifier needs one word — 3× faster than full token limit |
| `max_tokens=512` for generate/synthesize | Fits ~250 Ukrainian words; prompts instruct the model to stay within that |

### Model & Quantization

**Qwen3.5-9B** is chosen over 7B models for better reasoning on complex legal cross-references in Ukrainian Traffic Rules. **Q5_K_M quantization** fits the full model + KV cache into 8–16 GB VRAM while maintaining near-lossless perplexity (~6.6 GB for weights). Throughput on RTX 5060 Ti is ~52 tok/s. Embedding is fully local via `paraphrase-multilingual-MiniLM-L12-v2` — zero latency overhead, no API dependency.

---

## Evaluation Results

15 questions across 4 categories. Metric: keyword hit — at least one expected keyword present in `final_answer`.

| Category     | n  | ✓  | Hit rate |
|--------------|----|----|----------|
| calculation  | 3  | 3  | 100%     |
| simple       | 5  | 5  | 100%     |
| complex      | 5  | 5  | 100%     |
| out_of_scope | 2  | 2  | 100%     |
| **Total**    | 15 | 15 | **100%** |

Evaluation covers the full agentic workflow end-to-end (`eval.py`).

---

## Load Test Results

Hardware: RTX 5060 Ti 16 GB · Qwen3.5-9B-Q5_K_M · `--parallel 1`
n=100 sequential queries (5 query types × 20 repetitions)

| Metric | Value |
|--------|-------|
| p50    | 17.0s |
| p95    | 20.2s |
| p99    | 20.7s |
| avg    | 16.3s |
| min    | 10.0s |
| max    | 20.7s |

**Bottleneck:** LLM inference (~52 tok/s on Q5_K_M 9B model).

- `min=10s` — `calculation` path: 2 LLM calls (classify + synthesize), calculator runs without inference
- `p50=17s` — `simple` path: 3 LLM calls (classify + rag.generate + synthesize)
- `complex` path (6 LLM calls) was not included in the load test; estimated ~35–40s per query

**Optimization recommendations:**
1. `--parallel 4` is feasible on this hardware: model uses ~5.6 GB of 16 GB VRAM, leaving ~10 GB for KV cache → ~4× throughput under concurrent load
2. LRU cache on `rag_node`: repeated identical sub-queries (e.g. "speed limit in urban area") do not require re-inference

---

## Stack

| Component | Choice |
|-----------|--------|
| Package manager | `uv` |
| LLM inference | llama.cpp server — OpenAI-compatible API at `http://localhost:8001` |
| Model | `Qwen3.5-9B-Q5_K_M.gguf` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local, no API key) |
| Vector DB | PostgreSQL + pgvector |
| Agent framework | LangGraph |
| UI | Streamlit |

---

## Installation & Run

### Docker (recommended)

```bash
cp .env.example .env          # fill in: HF_TOKEN, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
docker compose up --build -d  # builds images, runs ingest automatically, then starts app
# Streamlit UI: http://localhost:8501
# llama.cpp UI:  http://localhost:8001
```

### Local dev

```bash
cp .env.example .env
uv sync
docker compose up llama-cpp postgres -d
uv run python app/ingest.py
uv run streamlit run app/main.py
# UI: http://localhost:8501
```

---

## Project Structure

```
app/
  core/
    config.py      # env vars + constants (CHUNK_SIZE, TOP_K, ...)
    rag.py         # RAG subgraph: retrieve → generate
    agent.py       # main LangGraph agent (6 nodes, routing, tools)
  main.py          # Streamlit UI
  ingest.py        # scrape zakon.rada.gov.ua → chunk → embed → pgvector
  eval.py          # 15 Q&A evaluation set with keyword-hit metric
  load_test.py     # sequential load test, reports p50/p95/p99
Dockerfile
docker-compose.yml
pyproject.toml
.env.example
```
