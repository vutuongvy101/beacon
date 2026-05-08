# COMP8420 Assignment 3 — Project Planning

## Use Case 2: Social Media Intelligent Platform

> **Group ID**: `<REPLACE_WITH_GROUP_ID>` · **4 members** · Weight: 40% of final grade
>
> Presentation due: **Friday 5 June 2026 (Week 13)**
> Report, Code & Video due: **Friday 19 June 2026 (Exam Period)**

---

## 1. Delivery Strategy

We target a **two-phase delivery** to manage risk and quality:

| Phase | Target | Deadline (internal) |
| ----- | ------ | ------------------- |
| Core delivery — stable, rubric-complete, end-to-end system | Fully demo-ready; all core features working | End of Week 2 (23 May) |
| Enhanced delivery — stronger analysis, multimodal module, polished UX, Q&A-ready | Primary differentiator integrated | End of Week 3 (30 May) |
| Buffer + final submission polish | Presentation (5 Jun) + Report/Code/Video (19 Jun) | Weeks 4–6 |

**Delivery tiers — priority is strictly enforced in this order:**

| Tier | Components | Status |
| ---- | ---------- | ------ |
| **Core (must deliver)** | Reddit/Sentiment140 ingestion, preprocessing (B1), NER (B2), rule-based extraction (B3), sentiment analysis (B4), topic/trend detection (B5), RAG evidence retrieval (A2), CoT/ReAct prompting (A3), LLM fine-tuning + prompting (A1), crisis detection (A5), agentic mini-demo (A6), dashboard | Non-negotiable |
| **Enhancement (primary differentiator)** | Multimodal analysis of Reddit image/meme/screenshot posts (A7) | Deliver if core is stable by Day 12 |
| **Stretch only** | Multilingual support (A4), dual-brand comparison, LLM-as-judge, full LangGraph orchestration | Only if time remains after Enhancement |

**Fallback order if time becomes limited:** Remove multilingual (A4) first → remove dual-brand comparison second → simplify multimodal (A7) to single-VLM approach third → replace LangGraph with sequential pipeline fourth → core sentiment/topic/RAG/report pipeline is **never removed**.

---

## 2. Target User and Business Scenario

**Target user:** Marketing analyst, PR team, or brand strategist responsible for monitoring online perception of a technology brand.

**Business scenario:** The analyst inputs a brand name or query. The system ingests a pre-collected Reddit snapshot, analyses public sentiment, identifies trending topics, extracts key entities and hashtags, flags emerging crisis signals, retrieves supporting evidence, and generates a concise executive summary with a recommended brand response.

**The system answers five core questions:**
1. What are people saying about the brand right now?
2. What topics are driving sentiment — positively or negatively?
3. Is there an emerging crisis that requires immediate attention?
4. What specific posts and evidence support these conclusions?
5. What response or campaign strategy should the brand consider?

This framing makes the project a *usable product* rather than a collection of NLP notebooks, and provides a coherent narrative for the report, presentation, and demo.

---

## 3. Project Overview

### Problem

Monitor social media content, identify trends, analyze public sentiment, detect emerging topics and crises, and generate insights for brand monitoring and campaign strategy.

### Target Brand

Choose one primary brand before Day 0. Dual-brand comparison is a **stretch-only** enhancement.

**Technology (recommended)**

| Brand | Why good for this system | Key subreddits / sources | Sentiment variation |
| ----- | ------------------------ | ----------------------- | ------------------- |
| **NVIDIA** ⭐ recommended | AI boom, GPU wars, strong community, directly relevant to course | r/nvidia, r/hardware, r/MachineLearning | High — gaming (positive) vs GPU pricing (negative) vs AI hype (mixed) |
| OpenAI / ChatGPT | Course-relevant, huge LLM community, safety/pricing controversies | r/ChatGPT, r/OpenAI, r/artificial | High — enthusiasm vs concern vs backlash |
| Tesla | Extremely polarizing, product + CEO controversies, stock discourse | r/teslamotors, r/investing, r/elonmusk | Very high — easy crisis detection |
| Apple | Global brand, product cycles, privacy debates, loyal community | r/apple, r/iphone, r/technology | Medium — mostly positive, spikes on launches |
| AMD vs NVIDIA | Dual-brand rivalry — **stretch goal only** (see tier table in Section 1) | r/hardware, r/amd, r/nvidia, r/buildapc | High — brand war dynamics |
| Microsoft | Azure, Copilot, gaming — broad coverage across topics | r/microsoft, r/windows, r/gaming | Medium |

**Non-technology alternatives (if team prefers)**

| Brand | Why good for this system | Key subreddits / sources | Sentiment variation |
| ----- | ------------------------ | ----------------------- | ------------------- |
| Nike | Strong social identity, athlete controversies, campaign sentiment | r/running, r/sneakers, r/sports | Medium-high |
| Starbucks | High volume, pricing backlash, loyalty program discussions | r/starbucks, r/coffee | Medium |
| Netflix | Content decisions, cancellations, pricing — classic brand sentiment | r/netflix, r/television | High — spikes on cancellations |
| Spotify | Music + podcast, artist payout debates, AI music controversy | r/spotify, r/music | High — AI music discourse |
| Peloton | Strong comeback story, crisis + recovery arc — great for crisis detection | r/pelotoncycle, r/fitness | Very high |

**Decision:** `<REPLACE: chosen brand>` (primary) + optional dual-brand for stretch goal only.

---

### Data Sources

Different agents require different data sources. Sources are chosen per agent, not globally.

#### Per-Agent Data Sources

| Agent | Data source | Type | Notes |
| ----- | ----------- | ---- | ----- |
| Collector Agent | Self-developed Reddit scraper | Scrape in advance, save JSON snapshot | Scraping happens during development, not at demo time. Demo and video always run on saved snapshots. |
| Collector Agent (fallback) | Sentiment140 | Static, text, labeled | 1.6M English tweets; primary for baseline evaluation |
| Multilingual Agent *(stretch only)* | SemEval 2017 Task 4 (multilingual Twitter) | Static, multilingual text | Covers Arabic, English, Spanish, French, German |
| Multilingual Agent *(stretch only)* | CLEF 2022 CheckThat! Dataset | Static, multilingual text | Misinformation + multilingual claims |
| RAG Agent | Brand Wikipedia pages + news articles | Static, text | Knowledge base for grounded retrieval |
| RAG Agent | NewsAPI / GNews API | Live, text | Brand news for real-time RAG context |
| Multimodal Agent *(enhancement — primary differentiator)* | Reddit image posts (scraped via self-developed script, image URLs saved alongside text) | Live + snapshot, image + text | Memes, screenshots, product photos from brand subreddits — no separate dataset needed |
| Multimodal Agent *(optional benchmark)* | Hateful Memes Dataset (Facebook AI) | Static, image + text | Optional — use only if team wants an external benchmark to compare VLM approaches |

#### Dataset Selection Guidance

Team members can choose from the options above based on their notebook scope. When selecting, document:

- Why this dataset suits the technique (justification for report)
- Its size, language coverage, and label schema
- How it was accessed (download URL or scraper script)

> **Note on Facebook scraping:** Facebook ToS prohibits scraping and its bot detection is aggressive. Accounts risk bans, data is non-reproducible. Not recommended for any agent. Reddit self-scraping is viable — use a throttled custom script and save snapshots for stable demos.

---

## 4. System Architecture

Two architecture versions are planned — Core delivered by end of Week 2, Enhanced by end of Week 3.

---

### Version 1 — Core (end of Week 2, English text only)

```
[Data Layer]
  Sentiment140 (static, English, labeled)
  + Reddit self-scraper (live, brand subreddits, saved snapshots)
        ↓
[Preprocessing Layer]                          ← B1
  Text cleaning, tokenization, normalization
  Hashtag / mention / URL extraction
        ↓
[Parallel Analysis Layer]                      ← B2, B3, B4, B5, A5
  ┌──────────────────────────────────────────┐
  │ Sentiment Analysis        (B4)           │
  │ Topic / Trend Detection   (B5)           │
  │ NER — brands, entities    (B2)           │
  │ Rule-based Signal Extract (B3)           │
  │ Crisis Signal Detection   (A5)           │
  └──────────────────────────────────────────┘
        ↓
[RAG Evidence Layer]                           ← A2
  FAISS vector store over brand corpus
  Retrieval of grounding context per signal
        ↓
[LLM Insight Generator]                        ← A1, A3
  Prompt-engineered generation (A1)
  CoT / ReAct trend explanation (A3)
  Branding strategy recommendation
        ↓
[Output Layer]
  FastAPI backend  →  Streamlit / Gradio / HTML+JS frontend
  Dashboard: sentiment trend, topic map, crisis alert
  Automated Report: Markdown → rendered in UI
```

---

### Version 2 — Enhanced (end of Week 3, multimodal as primary differentiator)

```
[Data Layer]
  + Image / meme data (Reddit image posts, URLs scraped alongside text)
  + Optional: Dual-brand corpus (e.g., NVIDIA + AMD) — stretch only
        ↓
[Preprocessing Layer]                          ← B1 + A7 enhanced
  + Image preprocessing (resize, caption extraction)
  + Optional: Language detection + translation (A4) — stretch only
        ↓
[Parallel Analysis Layer]                      ← all B + A5 + A7
  ┌──────────────────────────────────────────┐
  │ Multimodal Understanding  (image + text) │  ← PRIMARY DIFFERENTIATOR
  │ Enhanced Crisis Detection (A5 + A3)      │
  │ Multilingual Sentiment    (A4 + B4)      │  ← stretch only
  │ Dual-brand NER + Signals  (B2 + B3)      │  ← stretch only
  └──────────────────────────────────────────┘
        ↓
[Enhanced RAG Layer]                           ← A2 enhanced
  Multimodal context (image captions + text)
  Optional: Cross-lingual retrieval — stretch only
        ↓
[Enhanced LLM Insight Generator]               ← A1 + A3 enhanced
  VLM integration for meme analysis (GPT-4V or BLIP-2)
  Optional: Multilingual prompting — stretch only
        ↓
[Enhanced Output Layer]
  Dashboard + image analysis panel
  Optional: Dual-brand comparison panel — stretch only
```

---

### Agentic Workflow (LangGraph)

```
Orchestrator (LangGraph)
  ├── Collector Agent     → Reddit self-scraper or load Sentiment140 snapshot
  ├── Sentiment Agent     → calls classify_sentiment()
  ├── Topic Agent         → calls detect_topics()
  ├── Crisis Agent        → calls detect_crisis()
  ├── RAG Agent           → calls rag_retrieve()
  ├── Insight Agent       → calls cot_analyze() + run_llm()
  └── Report Agent        → assembles branding strategy + automated report output
```

**LangGraph state schema (shared across all agents):**

```python
class GraphState(TypedDict):
    posts: list[str]
    language_tags: list[str]          # populated by A4 (stretch only)
    sentiments: list[str]             # populated by B4
    topics: list[dict]                # populated by B5
    entities: dict                    # populated by B2
    crisis_level: str                 # "low" | "medium" | "high"
    retrieved_contexts: list[str]     # populated by A2
    trend_explanations: list[str]     # populated by A3
    insights: str                     # populated by A1
    branding_strategy: str            # populated by Report Agent
    report: str                       # final assembled output
```

Conditional edge: if `crisis_level == "high"` → skip branding recommendation → escalate to alert output.

**LangGraph fallback:** If LangGraph integration takes more than 1.5 days in Phase 2, implement a simple sequential modular pipeline instead. Each module calls the same Phase 1 exported functions in order. The report describes each module as an agent conceptually (Collector Agent, Sentiment Agent, etc.). Working integration is more important than using LangGraph.

---

### Output Layer — Technology Stack

| Component | Core (Week 2) | Enhanced (Week 3) |
| --------- | ------------- | ----------------- |
| Backend API | FastAPI — serves pipeline endpoints, handles requests | FastAPI — same, with multimodal endpoints added |
| Frontend | **Team chooses one:** Streamlit (fastest to build), Gradio (interactive widgets), or HTML/CSS/JS + FastAPI (most professional appearance) | Same framework, enriched UI with image analysis panel |
| Dashboard features | Sentiment trend, topic word cloud, crisis alert, report download | + Image analysis panel; optional dual-brand comparison (stretch) |
| Report format | Markdown → rendered in UI | Markdown → PDF export |

> **Frontend choice guidance:** Streamlit is recommended for speed; HTML/CSS/JS gives the most polished visual result. Lock this choice at Day 0 — do not switch mid-project.

---

### Deployment and Runtime Strategy

This project involves GPU-dependent workflows (LoRA fine-tuning, optional VLM inference) alongside CPU-safe components (rule-based extraction, traditional ML, API-based LLMs). A clear runtime strategy avoids contradictions and ensures reproducible demos.

#### Runtime Modes

| Mode | When used | What runs | Hardware required |
| ---- | --------- | --------- | ----------------- |
| **Full GPU mode** | Development, training, multimodal inference | LoRA fine-tuning (A1), VLM/BLIP-2 inference (A7), BERTopic on large corpus (B5) | Google Colab T4 (free tier sufficient) |
| **Lightweight CPU fallback** | Demo, video, submission, markers | API-based LLMs (GPT-4o-mini), FAISS retrieval, traditional ML models (B4), pre-loaded checkpoints, cached VLM outputs | Any machine with Python 3.10+ |

**Key principle:** The demo and video always run in CPU fallback mode on saved snapshot data. GPU mode is used only during development and training. The system must never require a GPU at presentation time.

#### Execution Paths

- **LoRA fine-tuning (A1):** Google Colab T4; checkpoint saved to `models/distilbert_lora/` after training; loaded from checkpoint at inference time — no retraining needed for demo.
- **VLM inference (A7):** GPT-4V via API (CPU-safe) or BLIP-2 on Colab T4; API outputs cached to `data/snapshots/multimodal_cache.json` after first call.
- **RAG retrieval (A2):** FAISS index built once, saved to `data/faiss_index/`; loaded at demo time — no rebuild required.
- **Pipeline demo:** Runs entirely from saved snapshot data; no live scraping, no live training, no GPU required.

#### Caching Strategy

Expensive inference outputs are cached to disk to ensure stable demos, avoid repeated API costs, and support offline reproducibility.

| Component | Cache file | What is cached |
| --------- | ---------- | -------------- |
| LLM report generation (A1) | `data/snapshots/llm_cache.json` | Prompt → response pairs |
| VLM image analysis (A7) | `data/snapshots/multimodal_cache.json` | Image URL → VLM description |
| FAISS index (A2) | `data/faiss_index/` | Vector index built from brand corpus |
| Reddit snapshot | `data/snapshots/reddit_<brand>_<date>.json` | Raw scraped posts |

**Caching rule:** Cached outputs are labelled as `[cached]` in any demo output so markers can distinguish live from pre-computed inference. Live inference remains the primary implementation path during development — caching is a fallback for demo stability only.

#### Hardware Assumptions

- All team members have Python 3.10+ locally
- Google Colab T4 (free tier) is available for GPU-dependent training
- No local GPU required for demo or submission
- API costs estimated at <$5 total (GPT-4o-mini at ~$0.15/1M tokens; VLM calls ~$0.01/image)

---

## 5. NLP Techniques Coverage (Rubric Mapping)

**Do we use all techniques listed here?** Yes — B1–B5 and A1–A6 are all implemented. The rubric only requires ≥3 basic + ≥3 advanced for baseline marks, but our 5 basic + 6 advanced gives strong coverage for high marks. A7 (multimodal) is the primary Week 3 enhancement differentiator. A4 (multilingual) is stretch only — do not plan for it in the core timeline.

**Alternatives are always evaluated inside the same notebook.** Each notebook compares its primary method against at least one alternative on the same sample data. There are no separate "comparison notebooks."

---

### Basic Techniques (≥3 required, targeting 5)

| ID | Technique | Notebook | Tier | Alternatives compared (within same notebook) | Rubric coverage |
| -- | --------- | -------- | ---- | -------------------------------------------- | --------------- |
| B1 | Text Preprocessing Pipeline | `basic/B1_preprocessing.ipynb` | **Core** | NLTK tokenizer vs spaCy tokenizer vs regex-based | tokenization, normalization, stemming/lemmatization, hashtag/mention extraction |
| B2 | Named Entity Recognition | `basic/B2_ner.ipynb` | **Core** | spaCy (`en_core_web_sm`) vs spaCy (`en_core_web_trf`) vs NLTK NE chunker | entity/location/brand extraction |
| B3 | Rule-based Information Extraction | `basic/B3_rule_extraction.ipynb` | **Core** | Custom regex vs spaCy Matcher vs manual keyword list | hashtag trends, mention networks, URL extraction |
| B4 | Sentiment Analysis (Traditional ML) | `basic/B4_sentiment_ml.ipynb` | **Core** | Naive Bayes vs Logistic Regression vs SVM — all with BoW and TF-IDF features | sentiment classification, baseline ML comparison |
| B5 | Topic / Trend Detection + Clustering | `basic/B5_topic_clustering.ipynb` | **Core** | LDA vs BERTopic vs K-Means on sentence embeddings | topic detection, text clustering, trend identification |

### Advanced Techniques (≥3 required, targeting 6+)

| ID | Technique | Notebook | Tier | Alternatives compared (within same notebook) | Rubric coverage |
| -- | --------- | -------- | ---- | -------------------------------------------- | --------------- |
| A1 | LLM Foundation Models — Fine-tuning + Prompting | `advanced/A1_llm_foundation.ipynb` | **Core** | **Part 1 (fine-tuning):** LoRA fine-tuned DistilBERT vs pretrained DistilBERT vs SVM baseline on Sentiment140. **Part 2 (prompting):** GPT-4o-mini zero-shot vs few-shot vs instruction. **Part 3:** fine-tuned small model vs prompted large model — accuracy, cost, latency tradeoffs | Fine-tuning/adaptation (LoRA), foundation model integration, prompt templates. Rubric: **1.5 marks** (fine-tuned) vs 0.5 (pretrained-only) |
| A2 | RAG (Retrieval Augmented Generation) | `advanced/A2_rag.ipynb` | **Core** | FAISS as primary vector store; `text-embedding-3-small` vs `all-MiniLM-L6-v2` as embedder | RAG pipeline, retrieval quality, grounded generation |
| A3 | Chain-of-Thought + ReAct Prompting | `advanced/A3_cot_react.ipynb` | **Core** | Direct prompting vs CoT vs ReAct on trend explanation task | Reasoning quality, step-by-step analysis |
| A5 | Crisis Detection (LLM-based) | `advanced/A5_crisis_detection.ipynb` | **Core** | Rule-based keyword threshold vs LLM binary classifier vs LLM with CoT reasoning | Risk classification, crisis explanation generation |
| A6 | Agentic Design (mini-demo) | `advanced/A6_agent_demo.ipynb` | **Core** | Single sequential chain vs LangGraph conditional graph (one-cycle demo) | Agentic design pattern, graph-based orchestration |
| A7 | Multimodal LLM | `advanced/A7_multimodal.ipynb` | **Enhancement** ⭐ *primary differentiator* | GPT-4V vs BLIP-2 vs EasyOCR for image+text Reddit posts; text-only vs multimodal sentiment comparison on 20–50 post benchmark | Multimodal LLM rubric item; demonstrates enriched understanding beyond text |
| A4 | Multilingual Analysis | `advanced/A4_multilingual.ipynb` | **Stretch only** | NLLB-200 vs Google Translate API vs mBART for translation; multilingual sentiment vs English-only | Language detection, translation, multilingual sentiment |

> A6 is a Phase 1 mini-demo only (one agent cycle to validate the LangGraph pattern). The full orchestrator is built in Phase 2.
> A7 is the primary Week 3 differentiator — more visually compelling than multilingual, easier to demo, more directly relevant to social media meme/screenshot analysis.
> A4 (multilingual) is stretch only — do not block core or enhancement delivery on it.

### Techniques from the rubric we are NOT prioritising (and why)

| Rubric technique | Our position |
| ---------------- | ------------ |
| POS Tagging | Not directly useful for brand monitoring tasks; excluded to stay focused |
| Ensemble Methods | Partially covered via B4 comparison; not a dedicated notebook |
| Fine-tuning / LoRA | **Implemented in A1 from Phase 1** — LoRA fine-tuning of DistilBERT on Sentiment140 (1.5 marks on LLM foundation rubric item) |
| Automated evaluation (LLM-as-judge) | Stretch only — evaluating A1/A3 outputs is a strong differentiator if time allows after A7 |
| Privacy / Fairness / Ethics | Addressed in Section 10 (dedicated Ethics section) and report |
| Vector DB comparison (FAISS vs Chroma) | FAISS chosen as primary; document rationale in A2 — unnecessary to build both |

---

## 6. Repository Structure (Submission Format)

```
<GroupID>_Assignment3/
├── Codes/
│   ├── shared/
│   │   ├── config.py              # API keys (gitignored), model config, demo mode flag
│   │   ├── data_loader.py         # load_sample() used by all notebooks
│   │   └── interface.py           # exported function contracts
│   ├── basic/
│   │   ├── B1_preprocessing.ipynb
│   │   ├── B2_ner.ipynb
│   │   ├── B3_rule_extraction.ipynb
│   │   ├── B4_sentiment_ml.ipynb
│   │   └── B5_topic_clustering.ipynb
│   ├── advanced/
│   │   ├── A1_llm_foundation.ipynb    # fine-tuning (Part 1) + prompting (Part 2) + comparison (Part 3)
│   │   ├── A2_rag.ipynb
│   │   ├── A3_cot_react.ipynb
│   │   ├── A4_multilingual.ipynb      # stretch only
│   │   ├── A5_crisis_detection.ipynb
│   │   ├── A6_agent_demo.ipynb
│   │   └── A7_multimodal.ipynb        # Week 3 enhancement — primary differentiator
│   ├── pipeline/
│   │   ├── agents/                    # LangGraph agent nodes (or sequential pipeline modules)
│   │   ├── graph.py                   # LangGraph orchestrator (or sequential runner as fallback)
│   │   ├── api.py                     # FastAPI backend (serves pipeline endpoints)
│   │   └── app.py                     # Streamlit / Gradio / HTML frontend
│   ├── evaluation/
│   │   └── metrics.py                 # evaluation helpers: macro-F1, coherence, Recall@K, groundedness
│   └── data/
│       ├── sample.csv                 # <5MB sample for submission
│       ├── snapshots/                 # saved Reddit JSON snapshots + LLM/VLM cache files
│       │   ├── reddit_<brand>_<date>.json
│       │   ├── llm_cache.json
│       │   └── multimodal_cache.json
│       └── faiss_index/               # saved FAISS vector index (built once, reloaded at demo)
├── models/
│   └── distilbert_lora/               # saved LoRA fine-tuned checkpoint from A1 Colab training
├── outputs/
│   ├── sample_report.md               # example generated brand intelligence report
│   └── figures/                       # exported charts for report and presentation
├── Report/
│   └── <GroupID>_Report.pdf
├── Presentation/
│   └── Presentation.pdf
├── Video/
│   └── demo_video.<ext>
└── README.md
```

---

## 7. Notebook Standard Structure (mandatory for all)

Every notebook (basic and advanced) must follow this structure:

```
1. Objective           — what technique, why relevant to social media brand monitoring
2. Setup               — imports, load_sample() from shared/data_loader.py
3. Implementation      — the technique
4. Results             — metrics table, example outputs on brand/social media data
5. Comparison          — vs. at least 1 alternative (same 200–500 sample posts)
6. Justification       — why chosen over alternatives for this use case (2–3 paragraphs)
7. Limitations         — what this technique fails on or cannot handle (1 paragraph)
8. Pipeline connection — how this notebook's output connects to the full pipeline (1 paragraph)
9. Export function     — clean def with type hints + docstring (used by Phase 2)
```

**Comparison is not a full benchmark.** Run both methods on the same 200–500 posts, produce one metrics table or qualitative comparison table, and write 3–5 sentences of analysis. That is sufficient.

**Every notebook must contain:**
- one result table (with actual numbers, not placeholders)
- one figure or example output
- one limitations paragraph
- one pipeline-connection paragraph
- one justification paragraph (2–3 paragraphs)

### Exported function contracts (all notebooks must match these signatures)

```python
# B1
def preprocess(texts: list[str]) -> list[str]: ...

# B2
def extract_entities(text: str) -> dict: ...

# B3
def extract_signals(text: str) -> dict: ...  # hashtags, mentions, urls

# B4
def classify_sentiment(texts: list[str]) -> list[str]: ...  # "positive"/"negative"/"neutral"

# B5
def detect_topics(texts: list[str]) -> list[dict]: ...  # [{topic_id, keywords, posts}]

# A1 — two exports (fine-tuned model + LLM prompting)
def classify_sentiment_ft(texts: list[str]) -> list[str]: ...  # LoRA fine-tuned DistilBERT
def run_llm(prompt: str) -> str: ...                           # GPT/Gemini prompt call

# A2
def rag_retrieve(query: str, top_k: int = 5) -> list[str]: ...

# A3
def cot_analyze(posts: list[str]) -> dict: ...  # {reasoning, output}

# A4 (stretch only)
def process_multilingual(text: str) -> dict: ...  # {translated_text, lang_tag}

# A5
def detect_crisis(posts: list[str]) -> dict: ...  # {risk_level, explanation}

# A7 (Week 3 enhancement)
def analyze_multimodal(text: str, image_url: str) -> dict: ...  # {enriched_text, image_description, modality_used}
```

---

## 8. Evaluation Matrix

For each component, the following metrics must be reported in the notebook Results section and summarised in the report's Experiment & Analysis section.

### Per-component metrics

| Component | Metric(s) | Evaluation method |
| --------- | --------- | ----------------- |
| B4 — Sentiment Analysis | Accuracy, macro-F1, per-class F1 | Evaluated on held-out Sentiment140 test set |
| B5 — Topic Clustering | Topic coherence (C_v score), qualitative topic interpretability | Coherence computed with gensim; 3 human-readable topic examples reported |
| B2 — NER | Precision, recall, F1 on brand/entity mentions | Evaluated on 50-post manually labelled sample |
| B3 — Rule-based Extraction | Coverage rate, false positive rate | Manually verified on 50-post sample |
| A5 — Crisis Detection | Precision, recall, F1, false positive analysis | Evaluated on 30–50 manually labelled crisis/non-crisis posts |
| A2 — RAG | Recall@5 (does ground truth appear in top 5 results?), relevance score (LLM-rated 1–5) | Tested on 20 query-context pairs |
| A1 — LLM Report Generation | Usefulness, factual grounding, clarity, actionability (1–5 scale, human-rated) | 20–30 sampled outputs rated by team members |
| A7 — Multimodal | Sentiment/topic agreement rate between text-only vs text+image analysis | Compared on 20–50 Reddit image posts; cases where modality changes classification are highlighted |

### Human evaluation plan

For A1 (LLM-generated reports) and A7 (multimodal enrichment), conduct a small human evaluation:

- Sample 20–30 outputs (report paragraphs or multimodal enrichment results)
- Each team member rates independently on a 1–5 scale for:
  - **Usefulness** — does the output help a brand analyst make a decision?
  - **Evidence support** — is the claim grounded in retrieved posts?
  - **Clarity** — is the language clear and professional?
  - **Actionability** — does the recommendation suggest a concrete next step?
- Report mean and standard deviation per criterion in the Experiment & Analysis section

---

## 9. MVP Demo Path

The following is the non-negotiable demo target. The project is only considered demo-ready when this full path works end-to-end on saved snapshot data without live API calls or GPU access.

**Input:** Brand name or query + saved Reddit JSON snapshot

**Output (all 7 must render without error):**
1. Sentiment distribution chart (positive / negative / neutral breakdown)
2. Top 5 topics with representative keywords and example posts
3. Top entities and hashtags extracted from corpus
4. Crisis level indicator (low / medium / high) with explanation
5. 5 retrieved evidence posts (from RAG, most relevant to current sentiment/crisis signal)
6. LLM-generated executive summary (~200 words)
7. Recommended brand response strategy (1–3 actionable bullet points)

**System-level success criteria (the demo is only ready when ALL are true):**
- Full path runs from one command or one README step on saved snapshot data
- Dashboard renders all 7 outputs without error or manual intervention
- Every recommendation includes at least one retrieved evidence post (RAG grounding visible)
- System runs offline from committed snapshot — no live API required at demo time
- Demo can run from any team member's machine after `pip install -r requirements.txt`
- All expensive inference outputs are cached and labelled `[cached]` in output
- No GPU required at demo time

---

## 10. Ethics, Privacy, and Limitations

This section must be referenced in the report introduction and methodology sections.

### Data and privacy

- No collection of private user data. Only public Reddit posts are used.
- Reddit posts are stored in JSON snapshots without modification, and only for academic research purposes.
- Usernames in analysis outputs are anonymised (replaced with `[user]` or omitted from display).
- No real-time scraping occurs during demo, presentation, or video — all data is from a pre-collected snapshot.

### Representativeness and bias

- Social media sentiment is not representative of the general population. Reddit skews toward specific demographics.
- Sarcasm, memes, and irony are hard to classify correctly — the system may misclassify ironic positive statements as genuine positive sentiment. Multimodal analysis (A7) partially addresses this limitation.
- Crisis detection should **support human review**, not replace human judgment. Automated alerts require verification before any real business action is taken.

### LLM reliability

- LLM-generated reports may hallucinate facts or recommendations if not grounded with retrieved evidence. All generated insights must be paired with supporting retrieved posts (RAG output in the pipeline).
- LLM outputs are labelled as AI-generated in all report and dashboard views.

### Limitations summary

- English-only in core system (multilingual is stretch only).
- System quality depends on snapshot recency — a stale snapshot may not reflect current brand perception.
- API dependency for cloud LLMs (GPT-4o-mini, GPT-4V) introduces cost and availability risk — mitigated by caching and open-source fallbacks (see Section 15).

---

## 11. Phase Plan

### Pre-Phase (Day 0 — all members together, ~3h)

**Decisions to lock before anyone writes code:**

- Group ID confirmed
- Brand confirmed (NVIDIA as primary, AMD for stretch only)
- LLM provider + model agreed (suggested: GPT-4o-mini or Gemini 1.5 Flash — cheap, fast)
- Frontend choice locked: Streamlit vs Gradio vs HTML/CSS/JS (see Section 4 Output Layer)
- Reddit scraper script working (self-developed, throttled, saves JSON snapshots)
- LangGraph state schema reviewed and approved (see Section 4)
- Exported function contracts reviewed and approved (see Section 7)
- Notebook assignments confirmed (see Section 12)
- `shared/data_loader.py` written and pushed (one person, ~1h)
- `shared/config.py` template pushed (API key slots, gitignored actual keys, demo mode flag)
- `evaluation/metrics.py` skeleton created (metric helper stubs)
- Sample dataset prepared: 5,000 rows from Sentiment140 + 500 scraped Reddit posts for chosen brand

---

### Phase 1 — Technique Investigation (Days 1–7)

**Goal:** Each notebook explores its technique, produces evaluation results, and exports a clean callable function.

| Notebook | Key tasks | Exit criteria |
| -------- | --------- | ------------- |
| B1 | Implement preprocessing pipeline, compare 2 tokenizers/normalizers | Results table, `preprocess()` exported |
| B2 | Implement NER, compare spaCy vs NLTK on brand/entity extraction | Precision/recall table on 50-post sample, `extract_entities()` exported |
| B3 | Implement regex extraction, compare rule patterns | Coverage/accuracy table on 50-post sample, `extract_signals()` exported |
| B4 | Train Naive Bayes + LR + SVM on Sentiment140 sample, compare macro-F1 | F1 comparison table, `classify_sentiment()` exported |
| B5 | Run LDA vs BERTopic on brand posts, compare topic coherence (C_v) | Coherence scores, example topics, `detect_topics()` exported |
| A1 | **Part 1:** LoRA fine-tune DistilBERT on Sentiment140 sample (Colab T4, ~15–20 min). Compare: SVM → pretrained DistilBERT → LoRA fine-tuned. **Part 2:** GPT-4o-mini with 3 prompt styles (zero-shot, few-shot, instruction). **Part 3:** fine-tuned small model vs prompted large model — accuracy, cost, latency tradeoffs | F1 comparison table (all 5 methods), `run_llm()` and `classify_sentiment_ft()` exported; checkpoint saved to `models/distilbert_lora/`. **Expected time: 2 days** |
| A2 | Build FAISS vector store, compare 2 embedding models, evaluate on 20 query-context pairs | Recall@5 table, `rag_retrieve()` exported; FAISS index saved to `data/faiss_index/` |
| A3 | Compare direct vs CoT vs ReAct prompting for trend explanation | Qualitative comparison table, `cot_analyze()` exported |
| A5 | Implement crisis classifier, compare rule-based threshold vs LLM on 30–50 labelled posts | Precision/recall table, `detect_crisis()` exported |
| A6 | Build one full agent cycle using functions from B4, A5, A3 | Working mini-demo run, LangGraph pattern validated |

**A4 (multilingual) — stretch only.** Do not start unless all Core notebooks are complete and A7 is in progress.

**During Phase 1:** write the methodology paragraph for your notebook immediately after completing it — this becomes your report section draft.

---

### Phase 2 — Integration & Pipeline (Days 8–12)

**Goal:** Wire all Phase 1 exported functions into the orchestrator and demo UI.

| Task | Key work |
| ---- | -------- |
| Orchestrator (`pipeline/graph.py`) | LangGraph nodes or sequential fallback — implement whichever is achievable within 1.5 days |
| Agent nodes (`pipeline/agents/`) | Each node reads GraphState, calls the relevant Phase 1 export function, writes result back to state |
| Collector Agent | Load pre-scraped JSON snapshot or Sentiment140 sample → preprocessed posts into state. Never called live at demo or video time. |
| FastAPI backend (`pipeline/api.py`) | Expose pipeline as REST endpoints; frontend calls `/analyze`, `/report` etc. |
| Demo UI (`pipeline/app.py`) | Frontend: input (brand/query), calls FastAPI, displays all 7 MVP outputs (see Section 9) |
| Report Agent | Assemble all state outputs into Markdown: sentiment summary, top topics, crisis status, 5 evidence posts, branding strategy |
| Integration testing | After each agent node is added, run full pipeline on 10 sample posts before proceeding |
| LLM/VLM output caching | Save expensive inference results to `data/snapshots/llm_cache.json` and `multimodal_cache.json` |

**Testing rule:** every new agent node must pass a 10-post end-to-end test before the next node is added. Do not batch-integrate and test at the end.

---

### Phase 3 — Enhancement & Polish (Days 13–17)

**Goal:** Integrate multimodal as the primary differentiator; ablation study; report and presentation polish.

#### Do we retrain models in Phase 3?

**Generally no.** The enhancement phase is about deeper analysis, stronger coverage, and polish.

| Component | Retrain? | What to do instead |
| --------- | -------- | ----------------- |
| Traditional ML models (B4) | No | Hyperparameter tuning, feature engineering improvements |
| LLM (A1, A3) | No | Better prompt templates, few-shot examples |
| RAG (A2) | No | Improve retrieval (reranking), swap embedding model |
| Crisis detector (A5) | No | Better threshold calibration, more example labels |
| **A1 LoRA fine-tuned DistilBERT** | Already done in Phase 1 | Verify checkpoint loads correctly in pipeline; optionally re-run with full dataset |

---

| Task | Tier | Rubric impact |
| ---- | ---- | ------------- |
| A7: Multimodal notebook — Reddit image posts via GPT-4V or BLIP-2 (20–50 post benchmark, see Section 14) | **Enhancement** | Primary differentiator; Multimodal LLM rubric item |
| Ablation study: remove each advanced component, show F1/quality drop | Core polish | Evaluation & analysis (+1 mark) |
| Compare 2 foundation models (GPT-4o-mini vs Gemini Flash) on same prompts | Core polish | Comparison analysis (+1 mark) |
| Human evaluation: 20–30 sampled outputs rated 1–5 on 4 criteria (see Section 8) | Core polish | Report quality |
| Add LLM-as-judge automated evaluation for A1/A3 outputs | **Stretch** | Only if A7 is complete |
| A4: Multilingual notebook | **Stretch** | Only if all core + A7 complete |
| Dual-brand analysis (e.g., NVIDIA vs AMD) | **Stretch** | Only if A4 complete |
| Reproducibility pass: fresh clone + README clean run | Required | Submission requirement |
| Report final draft: all sections complete, figures polished | Required | Report marks |
| Presentation slides finalized (≤5 min, all members have speaking role) | Required | Presentation marks |
| Video recorded (all members appear, full workflow demo) | Required | Video marks |
| Q&A preparation: each member prepares 3 adversarial questions for the team | Required | Presentation Q&A marks |

---

## 12. Work Division

Team members choose their primary notebook ownership and collaborative contributions below. **Everyone contributes to implementation, report, presentation, and video.**

### Notebook ownership (choose from the pool)

| Role | Primary notebooks | Report section | Presentation | Video |
| ---- | ----------------- | -------------- | ------------ | ----- |
| **Data & Baselines** | B1, B2, B3 (or B1, B4) | Data & Evaluation section | 1 slide + ~60s speaking | 1 demo segment |
| **ML & Evaluation** | B4, B5 (or B3, B5) + `evaluation/metrics.py` | Experiment & Analysis section | 1 slide + ~60s speaking | 1 demo segment |
| **LLM & Prompting** | A1, A3 (or A1, A7 if multimodal is available) | Methodology — Advanced Techniques section | 1 slide + ~60s speaking | 1 demo segment |
| **RAG & Integration** | A2, A5, A6 + Phase 2 pipeline lead | Architecture + System Integration section | 1 slide + ~60s speaking | 1 demo segment |

> The **RAG & Integration** role also owns:
>
> - LangGraph state schema and exported function contracts (Pre-Phase, Day 0)
> - Phase 2 pipeline integration and testing
> - README and reproducibility pass

### Collaborative contributions (all members)

- All members commit implementation code every week
- All members peer-review at least one other member's notebook before Phase 2
- All members appear in the video and have a named speaking segment
- All members present live in Week 13 (mandatory per assignment guidelines)

---

## 13. Report Structure

Target: ≤5,000 words, PDF.

| Section | Content | Primary author |
| ------- | ------- | -------------- |
| Title page | Group info, member names, student IDs, contact emails | All |
| Introduction | Background, target user and business scenario, NLP/LLM significance | TBD |
| Problem & Task | Use case, challenges, related work, task decomposition | TBD |
| Roles & Responsibilities | Each member's contributions | All |
| Methodology | Architecture, workflow, deployment strategy, technique choices and justifications | TBD |
| Data & Evaluation | Dataset, preprocessing, baselines, evaluation matrix, why chosen | TBD |
| Experiment & Analysis | Results tables, diagrams, ablation study, human evaluation results | TBD |
| Ethics & Limitations | Privacy, representativeness bias, sarcasm limits, LLM reliability | TBD |
| Recommendation & Discussion | Findings, limitations, future work | TBD |
| Conclusion | Summary | TBD |
| References | All external sources, models, datasets | All |
| Resources | GitHub URL, model download links, video link | TBD |

---

## 14. Key Decisions Still Open

These must be resolved at Pre-Phase Day 0:

- Group ID / team name
- **Target brand confirmed** (see candidate tables in Section 3)
- LLM provider + model (GPT-4o-mini vs Gemini 1.5 Flash vs other)
- **Frontend choice locked:** Streamlit vs Gradio vs HTML/CSS/JS (see Section 4 Output Layer)
- Notebook assignments (who owns which)
- Who owns presentation structure and video editing
- Report section assignments (fill TBD rows in Section 13)
- GPU access for A1 LoRA fine-tuning: Google Colab T4 confirmed
- Multimodal (A7): confirmed as Week 3 primary differentiator using Reddit image posts
- A4 (multilingual): confirmed as stretch only — do not plan for it in core timeline

---

### Multimodal Plan (Week 3, using Reddit data)

Reddit posts regularly include images (memes, screenshots, product photos) — making it a natural fit for multimodal analysis even without a separate image dataset.

**Why it is the primary differentiator over multilingual:** A meme post may say "NVIDIA is amazing" but show a GPU pricing chart with a crying face. Text-only analysis classifies this as positive; multimodal catches the irony. Multimodal analysis is more visually compelling for the demo, more technically distinctive, and more directly relevant to modern social media behaviour than multilingual support.

**Benchmark design (20–50 Reddit image posts):**
- Collect posts containing images from brand subreddits (image URLs scraped alongside post text)
- Run text-only sentiment/topic analysis on the post caption
- Run image-enriched analysis (VLM caption or OCR → append to post text → re-analyse)
- Compare: does multimodal enrichment change the sentiment or topic classification?
- Report cases where modality changes the result — these are the most compelling for the report and demo

**Example cases to target:**
- Sarcastic meme where text looks positive but image is negative (e.g., GPU pricing chart + crying emoji)
- GPU pricing screenshot where image contains text not present in the post caption
- Product announcement meme with ironic comparison to competitor
- Customer complaint posted as a screenshot rather than typed text

**Implementation approach (pick one, document comparison in notebook):**

| Approach | Tool | When to use |
| -------- | ---- | ----------- |
| VLM direct analysis | GPT-4V / GPT-4o vision API | Easiest to implement; small API cost (~$0.01/image); outputs cached after first call |
| Open-source VLM captioning | BLIP-2 (HuggingFace, free) | Free, runs on Colab T4; generates caption → feed to existing pipeline |
| OCR for screenshot posts | EasyOCR (free) | Best for screenshot-heavy subreddits; extracts embedded text from images |

**Integration into pipeline:**
1. Reddit scraper saves image URLs alongside post text (already planned in Collector Agent)
2. New `Multimodal Agent` node: for image posts, call VLM → get description → append to post text → pass enriched text to Sentiment/Topic agents
3. Compare text-only vs text+image analysis on benchmark set; report agreement rate and notable divergences
4. Cache VLM outputs to `data/snapshots/multimodal_cache.json` — no repeated API calls

---

## 15. Risks and Mitigations

| Risk | Likelihood | Mitigation |
| ---- | ---------- | ---------- |
| Phase 1 notebooks take longer than 1.5 days each | Medium | Timebox strictly; scope down comparison to 2 methods, 200 posts |
| Phase 2 integration breaks because exported functions are incompatible | High if not caught early | Integration lead reviews all Phase 1 PRs for function signature compliance |
| LangGraph learning curve exceeds 1.5 days | Medium | Fall back to sequential modular pipeline immediately; do not over-invest in LangGraph |
| Reddit scraper blocked or produces inconsistent data | Medium | Scraping done offline during development only; demo always runs on committed JSON snapshots — scraper instability has zero impact on presentation day |
| LLM API key failure or cost spike at demo time | Medium | Cache all LLM outputs to disk after first run; provide static cached outputs for demo; GPT-4o-mini estimated <$5 total; open-source fallback (local Ollama or Gemini free tier) |
| VLM/multimodal inference latency or API failure | Medium | Cache VLM outputs after first call; demo always shows cached results; EasyOCR as CPU-safe fallback |
| CUDA incompatibility or local GPU driver issues | Medium | All GPU-dependent work runs on Google Colab T4; save checkpoint after training; demo never requires GPU |
| Colab session timeout during LoRA training | Low | Save checkpoint every epoch; resume from checkpoint; estimated training time ~15–20 min |
| Model checkpoint loading failure at demo time | Low | Test checkpoint loading on CPU before presentation day; keep API-based fallback (GPT-4o-mini) ready |
| Dependency conflict between notebooks | Medium | Pin all package versions in `requirements.txt`; test fresh-clone install before submission |
| Weak analysis marks (most common loss area) | Medium | ML & Evaluation member starts building `evaluation/metrics.py` in Phase 1, not Phase 2 |
| Unequal contributions flagged by marker | Medium | Enforce cross-deliverable checklist; each member must have code commits + report section + video segment |
| Demo instability on presentation day | Low | Demo delivered via pre-recorded video running on committed snapshots — no live pipeline or scraping during presentation |

---

## 16. Definition of Done (per deliverable)

### Notebook done criteria

Each notebook is done when:

- Follows standard structure (Sections 1–9 of notebook template in Section 7)
- Results table present with at least 2 methods compared, using metrics from Section 8 (Evaluation Matrix)
- Limitations paragraph written (1 paragraph minimum)
- Pipeline-connection paragraph written (1 paragraph minimum)
- Justification paragraph written (2–3 paragraphs)
- Export function passes a 5-call smoke test
- Peer-reviewed by one other team member

### System-level done criteria (MVP Demo Path, Section 9)

The system is demo-ready only when **all** of the following are true:

- Full pipeline executes end-to-end from one command on saved snapshot data
- Dashboard renders all 7 MVP outputs without error or manual intervention
- Generated report includes at least 5 retrieved evidence posts (RAG grounding visible)
- Backend/frontend integration is stable (FastAPI + chosen frontend)
- Demo works offline — no live API calls, no live scraping, no GPU required
- Demo runs from any team member's machine after `pip install -r requirements.txt`
- All expensive inference outputs are cached and labelled `[cached]` in output

### Reproducibility target

A new user (e.g., a marker) must be able to:
1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Follow the README to run the MVP demo on saved sample data
4. See the full dashboard and generated report without providing any API keys

This requires: saved Reddit snapshot in `data/snapshots/`, cached LLM outputs in `data/snapshots/llm_cache.json`, saved FAISS index in `data/faiss_index/`, and a `DEMO_MODE=true` flag in `shared/config.py` that loads cached outputs instead of calling live APIs.

### Phase 2 pipeline done when

- Full pipeline runs end-to-end on 10 posts without error
- Dashboard displays sentiment, topics, crisis level, and report
- README documents how to run the system from a fresh clone
- LLM/VLM outputs cached and verified

### Report done when

- All sections complete, no TBD
- Results tables have actual numbers (not placeholders)
- Evaluation matrix metrics reported for all core components
- Human evaluation results included (20–30 sampled outputs)
- GitHub URL and video link included in Resources section
- Peer-reviewed by all members

### Presentation done when

- ≤5 minutes timed (hard limit — overtime incurs mark reduction)
- All members have a named speaking segment
- Demo shown via pre-recorded video (no live scraping — pipeline runs on committed snapshot data)
- Q&A: each member has prepared answers for at least 3 questions

### Video done when

- ≤5 minutes
- Shows full MVP path: input → analysis → dashboard → report
- All members appear and narrate a segment
- Upload link is accessible to markers

---

## 17. Operational Fallback Philosophy

The architecture is ambitious. This section defines what gets cut and in what order if time becomes limited. This must be agreed by the team at Pre-Phase Day 0 so no one is surprised mid-implementation.

**Fallback order (cut in this sequence, never out of order):**

| Priority | Component | Action if time is insufficient |
| -------- | --------- | ------------------------------ |
| Cut first | Multilingual support (A4) | Remove entirely; mention as future work in report |
| Cut second | Dual-brand comparison | Remove entirely; mention as future work in report |
| Simplify third | Multimodal (A7) | Reduce benchmark from 50 posts to 20; use GPT-4V API only (skip BLIP-2 and EasyOCR comparison) |
| Simplify fourth | LangGraph orchestration | Replace with sequential pipeline; describe each module as an agent conceptually in the report |
| Cut fifth | PDF report export | Output Markdown in dashboard only |
| **Never cut** | Core pipeline | Reddit/S140 ingestion, preprocessing, sentiment, topic, NER, crisis detection, RAG, LLM report, dashboard |

**Why this order:** multilingual and dual-brand are additive with no rubric dependency. Multimodal is the primary differentiator but can be reduced. LangGraph is an implementation detail — the agent concept survives even with a sequential fallback. The core pipeline is the rubric-complete deliverable.

**Signal to cut:** if any single Phase 1 notebook is not complete by Day 5, immediately scope down its comparison (2 methods only, 200 posts) and move on. Do not let one notebook block Phase 2.
