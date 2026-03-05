# ПДР Асистент — Agentic RAG Chatbot

Agentic RAG chatbot на базі LangGraph + локальна LLM (Qwen3.5-9B via llama.cpp) + Streamlit UI.
Домен: **Правила Дорожнього Руху України** — 34 розділи, сотні пронумерованих пунктів.

---

## Problem & Objectives

Знайти конкретний пункт ПДР швидко — важко. Водії витрачають час на пошук у суцільному тексті.
Типові питання: *"Чи можна зупинятись на перехресті?"*, *"Яка дистанція при буксируванні?"* — відповідь є в документі, але знайти важко.

**Чому RAG:** ПДР — статичний, офіційний, фактологічний текст. Прямий LLM галюцинує номери пунктів. RAG повертає конкретний пункт і цитує його.

**Чому Agentic:**
- Прості питання ("що означає знак 1.5?") → швидкий шлях через RAG
- Складні ("порівняй правила зупинки та стоянки") → декомпозиція на підзапити
- Поза ПДР ("який штраф?") → web_search (КУпАП, не ПДР)
- Обчислення ("гальмівний шлях") → `traffic_calculator` tool (формули, без LLM inference)

---

## System Architecture

```
User Query
    │
    ▼
[1] classify ──simple──► [3] rag_node ──► [5] synthesize ──► Response
    │                          ▲
  complex                      │
    │                    RAG Subgraph:
    ▼                    retrieve → generate
[2] decompose
    │
    ├──► [3] rag_node  (per sub-question)
    └──► [4] web_search (штрафи, КУпАП)
              │
              └──────────► [5] synthesize
[calculation]──► [4] calculator_node ──► [5] synthesize
```

**Main graph nodes (6):** `classify`, `decompose`, `rag_node`, `web_search`, `calculator_node`, `synthesize`
**RAG subgraph (окремий):** `retrieve` → `generate`
**Tools (2):**
- `traffic_calculator` — обчислення гальмівного шляху і безпечної дистанції (не retrieval)
- `DuckDuckGo` — web search для питань поза ПДР

**State:** `query` / `classification` / `sub_queries` / `rag_answers` / `web_results` / `final_answer` / `iterations`

### Design Decisions

| Рішення | Чому |
|---------|------|
| Qwen3.5-9B via llama.cpp | Open-source, вміщується в 16GB VRAM (~5.6GB), OpenAI-compatible API |
| Rule-based chunking | Кожен пункт ПДР — самодостатня логічна одиниця. Точніший retrieval ніж fixed-size |
| pgvector | Один стек з PostgreSQL, без зайвих сервісів |
| paraphrase-multilingual-MiniLM-L12-v2 | Підтримує українську мову |
| max_tokens=64 для classify | Класифікатору потрібне одне слово — 3× швидше ніж повний ліміт |

---

## Evaluation Results

15 питань по 4 категоріях, метрика: keyword hit (ключові слова у `final_answer`).

| Category     | n  | ✓  | Hit rate |
|--------------|----|----|----------|
| calculation  | 3  | 3  | 100%     |
| simple       | 5  | 5  | 100%     |
| complex      | 5  | 5  | 100%     |
| out_of_scope | 2  | 2  | 100%     |
| **Total**    | 15 | 15 | **100%** |

---

## Load Test Results

Hardware: RTX 5060 Ti 16GB · Qwen3.5-9B-Q5_K_M · `--parallel 1`
n=100 sequential queries (5 типів × 20 повторень)

| Metric | Value |
|--------|-------|
| p50    | 17.0s |
| p95    | 20.2s |
| p99    | 20.7s |
| avg    | 16.3s |
| min    | 10.0s |
| max    | 20.7s |

**Bottleneck:** LLM inference (llama.cpp, ~52 tok/s на Q5_K_M 9B).

- `min=10s` → `calculation` path: 2 LLM calls (classify + synthesize), calculator без inference
- `p50=17s` → `simple` path: 3 LLM calls (classify + rag.generate + synthesize)
- `p99≈p95` → тест містить тільки simple/calculation; `complex` path (6 LLM calls) дав би ~35-40s

**Optimization recommendations:**
1. `--parallel 4` feasible на цьому залізі: модель займає ~5.6GB з 16GB VRAM → ~10GB вільно для KV cache → ~4× throughput при concurrent load
2. LRU cache на `rag_node`: повторні ідентичні підзапити (напр. "швидкість у місті") не потребують inference

---

## Stack

| Component | Choice |
|-----------|--------|
| Package manager | `uv` |
| LLM inference | llama.cpp server — OpenAI-compatible API |
| Model | `Qwen3.5-9B-Q5_K_M.gguf` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local) |
| Vector DB | PostgreSQL + pgvector |
| Agent framework | LangGraph |
| UI | Streamlit |

---

## Installation & Run

### Docker (recommended)

```bash
cp .env.example .env   # fill HF_TOKEN, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
docker compose up -d   # ingest runs automatically before app
# UI: http://localhost:8501
```

### Local dev

```bash
cp .env.example .env
uv sync
docker compose up llama-cpp postgres -d
uv run python app/ingest.py
uv run streamlit run app/main.py
```

---

## Project Structure

```
app/
  core/
    config.py      # env vars + constants
    rag.py         # RAG subgraph: retrieve → generate
    agent.py       # main LangGraph agent (6 nodes)
  main.py          # Streamlit UI
  ingest.py        # scrape zakon.rada.gov.ua → pgvector
  eval.py          # 15 Q&A evaluation
  load_test.py     # sequential load test, p50/p95/p99
Dockerfile
docker-compose.yml
pyproject.toml
.env.example
```
