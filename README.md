# BEACON

**B**rand **E**vidence-**A**ware **C**risis & **O**pinion **N**etwork

A multi-agent NLP system for social media brand monitoring. BEACON ingests posts from Reddit and labelled corpora, runs sentiment / topic / NER / crisis-signal analysis in parallel, grounds insights via RAG, and produces a branding-strategy report through a LangGraph orchestrator.

> COMP8420 Assignment 3 — Use Case 2: Social Media Intelligent Platform
> Macquarie University, 2026 S1

---

## What it does

- **Collects** brand-related posts (Reddit self-scraper + Sentiment140 fallback)
- **Analyses** sentiment, topics, entities, and rule-based signals in parallel
- **Detects** crisis-level discourse via LLM + rule-based classifiers
- **Grounds** insights with RAG over a brand knowledge base (FAISS / Chroma)
- **Generates** a branding-strategy report using prompt-engineered LLMs (zero/few-shot, CoT, ReAct)
- **Orchestrates** the above through a LangGraph multi-agent workflow

Advanced extensions (Week 3): multilingual analysis, dual-brand comparison, multimodal meme understanding.

---

## Repository layout

```
backend/            # FastAPI dashboard API (load JSONs, filter, search)
frontend/           # HTML/CSS/JS dashboard (Chart.js)
scripts/            # export_dashboard_inputs.py, setup_dashboard.sh, reddit_scraper.py
shared/             # config, data loader, exported function contracts
basic/              # B1–B5 notebooks (preprocessing, NER, rules, sentiment, topics)
advanced/           # A1–A7 notebooks (LLM, RAG, CoT/ReAct, crisis, agentic, multimodal, …)
pipeline/           # LangGraph orchestrator (planned)
evaluation/         # shared metrics helpers
data/               # sample data (<5MB), snapshots, FAISS index
models/             # saved checkpoints (e.g. LoRA)
outputs/            # sample reports and figures
Report/             # final PDF report
Presentation/       # slides
Video/              # demo recording
planning.md         # full project plan (architecture, phases, work division)
```

See [`planning.md`](./planning.md) for the full architecture, technique-to-rubric mapping, and phase plan.

---

## Setup

**Python 3.10+ required.**

```bash
# 1. Create and activate a virtual environment
python -m venv beacon-env
source beacon-env/bin/activate        # macOS / Linux
# beacon-env\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install a spaCy model (needed by B1, B2)
python -m spacy download en_core_web_sm

# 4. Copy config template and add your API key
cp shared/config.example.py shared/config.py
# Edit shared/config.py — set OPENAI_API_KEY
```

> **PyTorch (GPU / CPU):** `requirements.txt` omits `torch` intentionally because the
> install command depends on your hardware. Run one of:
> ```bash
> # CPU only (demo, notebooks, pipeline)
> pip install torch --index-url https://download.pytorch.org/whl/cpu
>
> # GPU (LoRA fine-tuning on Colab T4 — A1 notebook only)
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

> **Ollama (local LLM — no API key needed):** Install from [ollama.com](https://ollama.com), then pull the models:
> ```bash
> ollama pull qwen2.5:3b   # fast — demo, classification (A1/A5)
> ollama pull qwen2.5:7b   # capable — CoT/ReAct, report generation (A3)
> # qwen2-vl:7b runs on Colab T4 only (A7 multimodal)
> ```

---

## Data collection (run once, offline)

```bash
python scripts/reddit_scraper.py
# output → data/snapshots/reddit_openai_<date>.jsonl
```

The demo and all notebooks run from the committed snapshot — no live scraping at runtime.

---

## Dashboard (reactive PoC)

The brand monitor UI is **FastAPI + vanilla JS** (see [`DASHBOARD_SPEC.md`](./DASHBOARD_SPEC.md)). Use **Python 3.10–3.12** (3.13 may fail to build some pinned wheels).

```bash
# One-time setup (venv + deps + export JSONs + FAISS)
chmod +x scripts/setup_dashboard.sh run_server.sh
./scripts/setup_dashboard.sh --fast    # add --fast for stub topics (faster)

# Or step by step:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dashboard.txt
python scripts/export_dashboard_inputs.py --fast

# Run (API + static frontend on one port)
./run_server.sh
# → http://localhost:8000

# Optional: separate static server on :5500 (CORS already allowed)
cd frontend && python -m http.server 5500
```

**Windows (PowerShell) step-by-step:**

```powershell
# 1. Activate venv
beacon-env\Scripts\activate

# 2. Export data JSONs
python scripts/export_dashboard_inputs.py --fast

# 3. Start the server
uvicorn backend.main:app --reload
# → http://localhost:8000
```

**Refresh data** after notebooks produce new outputs:

```bash
source .venv/bin/activate
python scripts/export_dashboard_inputs.py        # full BERTopic topics
# restart ./run_server.sh
```

---

## Quickstart (Phase 2 pipeline)

```bash
uvicorn pipeline.api:app --reload

streamlit run pipeline/app.py
```

Individual notebooks under `basic/` and `advanced/` are self-contained — open in Jupyter and run top-to-bottom.

---

## Team

| Role | Member |
|---|---|
| Data & Baselines | TBD |
| ML & Evaluation | TBD |
| LLM & Prompting | TBD |
| RAG & Integration | TBD |

Group ID: `<REPLACE_WITH_GROUP_ID>`

---

## License

Academic submission — Macquarie University COMP8420, 2026 S1. Not licensed for external use.
