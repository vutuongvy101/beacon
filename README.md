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
Codes/
├── shared/         # config, data loader, exported function contracts
├── basic/          # B1–B5 notebooks (preprocessing, NER, rules, sentiment, topics)
├── advanced/       # A1–A6 notebooks (LLM, RAG, CoT/ReAct, multilingual, crisis, agentic)
├── pipeline/       # LangGraph orchestrator, FastAPI backend, Streamlit/Gradio frontend
└── data/           # sample data (<5MB)
Report/             # final PDF report
Presentation/       # slides
Video/              # demo recording
planning.md         # full project plan (architecture, phases, work division)
```

See [`planning.md`](./planning.md) for the full architecture, technique-to-rubric mapping, and phase plan.

---

## Quickstart (Phase 2 pipeline)

```bash
pip install -r requirements.txt

cp Codes/shared/config.example.py Codes/shared/config.py

uvicorn Codes.pipeline.api:app --reload

streamlit run Codes/pipeline/app.py
```

Individual notebooks under `Codes/basic/` and `Codes/advanced/` are self-contained — open in Jupyter and run top-to-bottom.

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
