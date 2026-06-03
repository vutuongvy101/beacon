# DASHBOARD SPEC — Reactive Brand Monitor

> **This is the dashboard-only spec.** Backend (FastAPI) + frontend (HTML/CSS/JS).
>
> **Assumptions locked:**
> - Search: semantic FAISS with entity-match badge
> - Default date range: full snapshot window
> - Suggested question click: prefills search box + streams cached answer
> - Influencer panel ranks **entities mentioned in posts**, not Reddit authors

---

## 1. Tech additions

```txt
# add to requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
pydantic==2.7.1
```

No frontend framework. Vanilla JS + Chart.js (CDN) + Tabler Icons (CDN) only.

---

## 2. Project structure additions

```
project/
├── backend/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── data_loader.py           # loads all JSONs into memory on startup
│   ├── filters.py               # applies date + search to DataFrames
│   ├── search.py                # FAISS semantic search + entity exact match
│   ├── endpoints/
│   │   ├── overview.py
│   │   ├── topics.py
│   │   ├── entities.py
│   │   ├── influencers.py
│   │   ├── posts.py
│   │   ├── evidence.py
│   │   └── qa.py
│   └── models.py                # Pydantic request/response models
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js                   # entry + state + refresh orchestration
│   ├── panels/
│   │   ├── overview.js
│   │   ├── topics.js
│   │   ├── entities.js
│   │   ├── influencers.js
│   │   ├── evidence.js
│   │   └── summary.js
│   ├── drawer.js                # level-2 drill-down drawer
│   └── api.js                   # fetch wrapper
├── scripts/
│   └── build_qa_bank.py         # pre-computes 20 cached LLM Q&A pairs
└── run_server.sh                # uvicorn launcher
```

---

## 3. FilterState — the single source of truth

Request body for every panel endpoint:

```python
# backend/models.py
from pydantic import BaseModel
from typing import Optional

class FilterState(BaseModel):
    date_from: Optional[int] = None    # unix timestamp, inclusive
    date_to: Optional[int] = None      # unix timestamp, inclusive
    search: Optional[str] = None       # free-form query string

class SearchResult(BaseModel):
    match_type: str                    # "semantic" | "entity" | "hashtag" | "author" | "none"
    matched_entity: Optional[str] = None
    matched_entity_count: Optional[int] = None
    post_ids: list[str]                # filtered post_ids after applying search
```

Frontend mirrors this exactly:
```js
// frontend/app.js
const filterState = {
  date_from: null,
  date_to: null,
  search: ""
};
```

---

## 4. Data loading (once, at startup)

```python
# backend/data_loader.py
import json
import pandas as pd
import faiss
import numpy as np
from pathlib import Path

DATA = {}  # global, populated once

def load_all():
    base = Path(__file__).parent.parent / "data" / "outputs"

    # Load JSONs
    posts = pd.DataFrame(json.loads((base.parent / "outputs" / "cleaned_posts.json").read_text()))
    sentiment = pd.DataFrame(json.loads((base / "sentiment.json").read_text()))
    entities = json.loads((base / "entities.json").read_text())
    topics = json.loads((base / "topics.json").read_text())
    trend = json.loads((base / "trend_timeseries.json").read_text())
    rag = json.loads((base / "rag_evidence.json").read_text())
    summaries = json.loads((base / "summaries.json").read_text())
    crisis = json.loads((base / "crisis.json").read_text())
    qa_bank = json.loads((base / "qa_bank.json").read_text())

    # Merge sentiment into posts for fast filtering
    posts = posts.merge(sentiment, on="post_id", how="left")

    # Pre-compute entity index: entity_text → list of post_ids
    entity_index = {}
    for row in entities:
        for ent in row["entities"]:
            key = ent["text"].lower().strip()
            entity_index.setdefault(key, []).append(row["post_id"])

    # Load FAISS index
    faiss_index = faiss.read_index(str(base.parent / "faiss_index" / "posts.index"))
    id_map = json.loads((base.parent / "faiss_index" / "id_map.json").read_text())

    DATA.update({
        "posts": posts, "entities": entities, "topics": topics, "trend": trend,
        "rag": rag, "summaries": summaries, "crisis": crisis, "qa_bank": qa_bank,
        "entity_index": entity_index, "faiss_index": faiss_index, "id_map": id_map
    })
```

Call `load_all()` in `main.py` startup event. Endpoints read from `DATA` dict — never re-load JSONs per request.

---

## 5. Search logic

```python
# backend/search.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from .data_loader import DATA

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_posts(query: str, top_k: int = 200) -> dict:
    """Returns: {match_type, matched_entity, matched_entity_count, post_ids}"""
    if not query or not query.strip():
        return {"match_type": "none", "matched_entity": None,
                "matched_entity_count": None, "post_ids": list(DATA["posts"]["post_id"])}

    q = query.strip()
    q_lower = q.lower()

    # 1. Hashtag exact match
    if q.startswith("#"):
        matching = [p["post_id"] for _, p in DATA["posts"].iterrows()
                    if q_lower in [h.lower() for h in (p.get("hashtags") or [])]]
        return {"match_type": "hashtag", "matched_entity": q,
                "matched_entity_count": len(matching), "post_ids": matching}

    # 2. Author exact match
    if q.startswith("u/"):
        author = q[2:].lower()
        matching = DATA["posts"][DATA["posts"]["author"].str.lower() == author]["post_id"].tolist()
        return {"match_type": "author", "matched_entity": q,
                "matched_entity_count": len(matching), "post_ids": matching}

    # 3. Entity exact match (returned as badge — also fall through to semantic)
    entity_match = None
    if q_lower in DATA["entity_index"]:
        entity_match = {
            "matched_entity": q,
            "matched_entity_count": len(DATA["entity_index"][q_lower])
        }

    # 4. Semantic search via FAISS
    q_vec = embed_model.encode([q])
    faiss.normalize_L2(q_vec)
    scores, indices = DATA["faiss_index"].search(q_vec, top_k)
    # Threshold: only keep results with similarity > 0.3
    semantic_ids = [DATA["id_map"][int(i)] for s, i in zip(scores[0], indices[0]) if s > 0.3]

    result = {
        "match_type": "semantic",
        "matched_entity": entity_match["matched_entity"] if entity_match else None,
        "matched_entity_count": entity_match["matched_entity_count"] if entity_match else None,
        "post_ids": semantic_ids
    }
    return result
```

---

## 6. Filter application

```python
# backend/filters.py
import pandas as pd
from .data_loader import DATA
from .search import search_posts

def apply_filters(filter_state) -> tuple[pd.DataFrame, dict]:
    """Returns: (filtered_posts_df, search_result_metadata)"""
    df = DATA["posts"].copy()

    # Date filter
    if filter_state.date_from:
        df = df[df["created_utc"] >= filter_state.date_from]
    if filter_state.date_to:
        df = df[df["created_utc"] <= filter_state.date_to]

    # Search filter
    search_meta = {"match_type": "none", "matched_entity": None, "matched_entity_count": None}
    if filter_state.search:
        search_result = search_posts(filter_state.search)
        df = df[df["post_id"].isin(search_result["post_ids"])]
        search_meta = {k: v for k, v in search_result.items() if k != "post_ids"}

    return df, search_meta
```

Every endpoint starts with `df, search_meta = apply_filters(req)`.

---

## 7. API endpoints — request/response contracts

All endpoints accept `FilterState` as POST body. All return JSON.

### POST `/api/overview`

```json
// response
{
  "kpis": {
    "total_posts": 487,
    "positive": 198, "negative": 154, "neutral": 135,
    "positive_pct": 40.7, "negative_pct": 31.6, "neutral_pct": 27.7,
    "unique_authors": 312,
    "total_entities": 89
  },
  "crisis": {
    "level": "amber",
    "reason": "Elevated negative sentiment...",
    "negative_ratio": 0.32
  },
  "sentiment_donut": [
    {"label": "positive", "count": 198, "color": "#4CAF50"},
    {"label": "negative", "count": 154, "color": "#F44336"},
    {"label": "neutral", "count": 135, "color": "#9E9E9E"}
  ],
  "sentiment_trend": [
    {"week": "2024-W18", "positive": 23, "negative": 18, "neutral": 14},
    {"week": "2024-W19", "positive": 31, "negative": 22, "neutral": 19}
  ],
  "search_meta": {"match_type": "semantic", "matched_entity": "GPT-4", "matched_entity_count": 87}
}
```

### POST `/api/topics`

```json
// response
{
  "topics": [
    {
      "topic_id": 0,
      "label": "GPT-4 pricing complaints",
      "keywords": ["price", "expensive", "subscription"],
      "post_count": 87,
      "sentiment_breakdown": {"positive": 12, "negative": 58, "neutral": 17},
      "dominant_sentiment": "negative"
    }
  ],
  "trend": [
    {"topic_id": 0, "label": "GPT-4 pricing", "weekly": [{"week": "2024-W18", "count": 14}]}
  ]
}
```

Filter logic: a topic appears only if at least 1 of its `post_ids` is in the filtered set. `post_count` is the *filtered* count, not the total.

### POST `/api/entities`

```json
// response
{
  "entities": [
    {
      "text": "Sam Altman",
      "label": "PERSON",
      "mention_count": 47,
      "post_count": 38,
      "sentiment_breakdown": {"positive": 18, "negative": 12, "neutral": 8},
      "dominant_sentiment": "positive"
    }
  ]
}
```

Aggregate across filtered posts. Return top 30.

### POST `/api/influencers`

This is the **corrected influencer panel**. Ranks entities mentioned in posts, not Reddit authors.

```json
// response
{
  "influencers": [
    {
      "rank": 1,
      "entity": "Sam Altman",
      "entity_label": "PERSON",
      "mention_count": 47,
      "engagement_score": 12340,
      "dominant_sentiment": "positive",
      "top_topics": ["AI safety", "company direction"]
    },
    {
      "rank": 2,
      "entity": "GPT-4",
      "entity_label": "PRODUCT",
      "mention_count": 89,
      "engagement_score": 9876,
      "dominant_sentiment": "negative",
      "top_topics": ["pricing", "context window"]
    }
  ]
}
```

**Ranking formula:** `engagement_score = mention_count × log(1 + sum of post upvotes mentioning entity)`.
This balances frequency with reach — a frequently-mentioned entity in low-engagement posts outranks rare mentions in viral posts only if mentions are very high.

### POST `/api/posts` (for drill-down)

```json
// request
{ "date_from": null, "date_to": null, "search": "Sam Altman", "limit": 20, "offset": 0 }

// response
{
  "total": 47,
  "posts": [
    {
      "post_id": "abc123",
      "title": "...",
      "clean_text": "snippet ≤ 300 chars",
      "score": 1204,
      "num_comments": 87,
      "author": "username",
      "created_utc": 1748000000,
      "sentiment": "positive",
      "sentiment_confidence": 0.87,
      "entities": [{"text": "Sam Altman", "label": "PERSON"}],
      "url": "https://reddit.com/..."
    }
  ]
}
```

### POST `/api/evidence`

```json
// request
{ "date_from": null, "date_to": null, "search": null, "topic_id": 0 }

// response
{
  "query": "GPT-4 pricing complaints",
  "evidence": [
    {
      "post_id": "abc123",
      "text": "snippet",
      "similarity_score": 0.91,
      "sentiment": "negative",
      "score": 1204
    }
  ]
}
```

If no `topic_id` provided, uses search query as the retrieval query.

### GET `/api/suggested-qa`

```json
// response
{
  "questions": [
    {"id": "q1", "text": "What is the dominant sentiment about GPT-4 pricing?", "category": "sentiment"},
    {"id": "q2", "text": "Are there any emerging crisis signals?", "category": "crisis"},
    {"id": "q3", "text": "Which OpenAI products are getting positive reactions?", "category": "products"}
  ]
}
```

### POST `/api/qa`

```json
// request
{ "question_id": "q1" }

// response
{
  "question": "What is the dominant sentiment about GPT-4 pricing?",
  "answer": "Sentiment toward GPT-4 pricing is predominantly negative (63%)...",
  "supporting_posts": ["abc123", "def456", "ghi789"],
  "suggested_search": "GPT-4 pricing"
}
```

`suggested_search` is what the frontend prefills into the search box when this question is clicked.

---

## 8. Frontend state + refresh orchestration

```js
// frontend/app.js
const filterState = { date_from: null, date_to: null, search: "" };
let isLoading = false;

const refreshDashboard = debounce(async () => {
  isLoading = true;
  showLoadingState();
  try {
    const [overview, topics, entities, influencers] = await Promise.all([
      api.post("/api/overview", filterState),
      api.post("/api/topics", filterState),
      api.post("/api/entities", filterState),
      api.post("/api/influencers", filterState),
    ]);
    renderOverview(overview);
    renderTopics(topics);
    renderEntities(entities);
    renderInfluencers(influencers);
    updateSearchBadge(overview.search_meta);
  } catch (e) {
    showErrorState(e);
  } finally {
    isLoading = false;
    hideLoadingState();
  }
}, 300);

document.addEventListener("DOMContentLoaded", async () => {
  setupDatePicker();
  setupSearchInput();
  setupSuggestedQA();
  await refreshDashboard();
  syncStateFromURL();  // restore filters from URL params
});

// URL syncing — every filter change updates URL
function syncStateToURL() {
  const params = new URLSearchParams();
  if (filterState.date_from) params.set("from", filterState.date_from);
  if (filterState.date_to) params.set("to", filterState.date_to);
  if (filterState.search) params.set("q", filterState.search);
  history.replaceState(null, "", `?${params}`);
}
```

---

## 9. HTML layout

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OpenAI Brand Monitor</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.5.0/tabler-icons.min.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

  <!-- HEADER -->
  <header class="topbar">
    <div class="brand">
      <i class="ti ti-radar-2"></i>
      <div>
        <div class="brand-title">OpenAI Brand Monitor</div>
        <div class="brand-sub">Reddit intelligence · r/ChatGPT · r/OpenAI</div>
      </div>
    </div>
    <div class="crisis-badge" id="crisis-badge"><!-- populated --></div>
  </header>

  <!-- CONTROL BAR (the only filters) -->
  <div class="control-bar">
    <div class="control-group">
      <label><i class="ti ti-calendar"></i> Date range</label>
      <select id="date-preset">
        <option value="all" selected>All time</option>
        <option value="7d">Last 7 days</option>
        <option value="30d">Last 30 days</option>
        <option value="custom">Custom...</option>
      </select>
      <div id="date-custom" hidden>
        <input type="date" id="date-from">
        <input type="date" id="date-to">
      </div>
    </div>
    <div class="control-group search-group">
      <label><i class="ti ti-search"></i> Search</label>
      <input type="text" id="search-input" placeholder="Try: GPT-4, pricing, #openai, u/spez...">
      <span class="search-badge" id="search-badge" hidden></span>
    </div>
    <button id="clear-filters" class="btn-ghost">
      <i class="ti ti-x"></i> Clear
    </button>
  </div>

  <!-- KPI ROW -->
  <div class="kpi-row" id="kpi-row"></div>

  <!-- MAIN GRID -->
  <div class="dashboard-grid">

    <!-- Suggested QA -->
    <section class="panel panel-full" id="panel-qa">
      <header><h2>Ask the data</h2></header>
      <div class="qa-grid" id="qa-grid"></div>
      <div class="qa-answer" id="qa-answer" hidden></div>
    </section>

    <!-- Sentiment + Trend -->
    <section class="panel" id="panel-sentiment">
      <header><h2>Sentiment breakdown</h2></header>
      <canvas id="chart-sentiment-donut"></canvas>
      <canvas id="chart-sentiment-trend"></canvas>
    </section>

    <!-- Topics -->
    <section class="panel" id="panel-topics">
      <header><h2>Topic clusters</h2></header>
      <canvas id="chart-topics-bar"></canvas>
    </section>

    <!-- Entities -->
    <section class="panel" id="panel-entities">
      <header><h2>Entity mentions</h2></header>
      <div id="entity-list"></div>
    </section>

    <!-- Influencers -->
    <section class="panel" id="panel-influencers">
      <header><h2>Top voices (entities)</h2></header>
      <table id="influencer-table"></table>
    </section>

    <!-- Crisis detail -->
    <section class="panel" id="panel-crisis">
      <header><h2>Crisis monitor</h2></header>
      <div id="crisis-content"></div>
    </section>

    <!-- LLM Summary -->
    <section class="panel panel-wide" id="panel-summary">
      <header><h2>Executive summary</h2></header>
      <div id="summary-content"></div>
    </section>

  </div>

  <!-- DRAWER (level 2 drill-down) -->
  <aside class="drawer" id="drawer" hidden>
    <header class="drawer-header">
      <h3 id="drawer-title"></h3>
      <button id="drawer-close"><i class="ti ti-x"></i></button>
    </header>
    <div class="drawer-content" id="drawer-content"></div>
  </aside>
  <div class="drawer-overlay" id="drawer-overlay" hidden></div>

  <!-- Loading overlay -->
  <div class="loading-overlay" id="loading-overlay" hidden>
    <div class="spinner"></div>
  </div>

  <script src="api.js"></script>
  <script src="drawer.js"></script>
  <script src="panels/overview.js"></script>
  <script src="panels/topics.js"></script>
  <script src="panels/entities.js"></script>
  <script src="panels/influencers.js"></script>
  <script src="panels/evidence.js"></script>
  <script src="panels/summary.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

---

## 10. Panel behaviours — clickable surfaces

Each panel has defined click targets that open the drawer.

| Panel | Clickable | Drawer shows |
|---|---|---|
| Sentiment donut | a sentiment slice | filtered post list with that sentiment |
| Sentiment trend | a week | filtered post list for that week |
| Topic bar | a topic | topic detail: keywords, top posts, RAG evidence |
| Entity item | an entity | entity detail: all posts mentioning, co-mentions, sentiment timeline |
| Influencer row | an entity row | same as entity detail |
| RAG evidence card | a post | post detail: full text, entities highlighted, similar posts |

All clicks pass `{type, id}` to `openDrawer(type, id)` in `drawer.js`.

---

## 11. Drawer states

```js
// frontend/drawer.js
async function openDrawer(type, id) {
  document.getElementById("drawer-overlay").hidden = false;
  document.getElementById("drawer").hidden = false;

  switch (type) {
    case "post":
      await renderPostDrawer(id);   // fetch /api/posts with post_id filter
      break;
    case "entity":
      await renderEntityDrawer(id); // fetch /api/posts + /api/entities for the entity
      break;
    case "topic":
      await renderTopicDrawer(id);  // fetch /api/evidence with topic_id
      break;
    case "sentiment":
      await renderSentimentDrawer(id); // filtered post list
      break;
  }
}
```

Drawer never modifies `filterState`. Closing returns to the same dashboard view.

---

## 12. Suggested questions interaction

```js
// frontend/app.js
async function onSuggestedQuestionClick(questionId) {
  const result = await api.post("/api/qa", { question_id: questionId });

  // 1. Prefill search box with suggested search
  document.getElementById("search-input").value = result.suggested_search;
  filterState.search = result.suggested_search;

  // 2. Stream the answer character-by-character (fake streaming for AI feel)
  const answerEl = document.getElementById("qa-answer");
  answerEl.hidden = false;
  answerEl.textContent = "";
  for (const char of result.answer) {
    answerEl.textContent += char;
    await sleep(8); // 8ms per char = readable streaming
  }

  // 3. Trigger dashboard refresh with new filter
  refreshDashboard();
}
```

---

## 13. QA bank generation script

```python
# scripts/build_qa_bank.py
"""Run once after pipeline completes. Generates ~20 cached LLM answers."""
import json
from pathlib import Path
from src.ollama_client import query_ollama

QUESTIONS = [
    {"id": "q1", "text": "What is the dominant sentiment about GPT-4 pricing?",
     "category": "sentiment", "search": "GPT-4 pricing"},
    {"id": "q2", "text": "Are there any emerging crisis signals?",
     "category": "crisis", "search": ""},
    {"id": "q3", "text": "Which OpenAI products are getting the most positive reactions?",
     "category": "products", "search": ""},
    {"id": "q4", "text": "What are developers' top concerns?",
     "category": "community", "search": "developer api"},
    {"id": "q5", "text": "How has sentiment about Sam Altman trended?",
     "category": "people", "search": "Sam Altman"},
    # ...20 total
]

PROMPT_TEMPLATE = """You are a brand analyst. Based on this Reddit data, answer the question concisely (3-4 sentences).

Data context:
{context}

Question: {question}

Answer:"""

def build_context_for_question(question, all_data):
    """Build a focused context based on the question's category and search filter."""
    # Use sentiment.json + topics.json + crisis.json + relevant filtered posts
    # Return formatted string ~1500 chars
    pass

def main():
    output = []
    for q in QUESTIONS:
        context = build_context_for_question(q, all_data)
        answer = query_ollama(PROMPT_TEMPLATE.format(context=context, question=q["text"]))
        output.append({
            "id": q["id"],
            "text": q["text"],
            "category": q["category"],
            "answer": answer.strip(),
            "supporting_posts": [],  # extract post_ids referenced
            "suggested_search": q["search"]
        })
    Path("data/outputs/qa_bank.json").write_text(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
```

---

## 14. Styling rules

- **Layout:** CSS Grid for main dashboard, 2-column on wide screens, 1-column under 900px
- **Panels:** white background, `border-radius: 12px`, subtle shadow `0 1px 3px rgba(0,0,0,0.06)`, `padding: 1.25rem`
- **KPI cards:** large number (24px, weight 500), small label (12px, muted)
- **Crisis badge in header:** colour pill — red `#F44336`, amber `#FF9800`, green `#4CAF50` — with icon and short reason text
- **Sentiment colours:** positive `#22C55E`, negative `#EF4444`, neutral `#94A3B8`
- **Drawer:** slides in from right, 480px wide, dark overlay behind (`rgba(0,0,0,0.4)`)
- **Loading state:** panels show a skeleton (animated grey blocks), not a global spinner — only show global spinner for initial load
- **Empty state per panel:** centered icon + "No data for current filters" text + a "Clear filters" button
- **Search badge:** shows next to search input when entity match found: `"Also matches entity: Sam Altman (47 posts)"` — clickable to switch to exact entity filter

---

## 15. Run commands

```bash
# Terminal 1: backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: frontend (any static server)
cd frontend
python -m http.server 5500

# Open browser: http://localhost:5500
# Backend API: http://localhost:8000
```

**CORS:** add to `main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5500"],
    allow_methods=["*"], allow_headers=["*"]
)
```

---

## 16. Integration rules

1. **Never re-load JSONs per request.** Read from `DATA` dict only.
2. **Apply filters first, then compute panel data.** Pattern: `df, meta = apply_filters(req); compute(df)`.
3. **Search debounce: 300ms.** Date change: immediate.
4. **All endpoints must handle empty filtered sets gracefully** — return empty arrays, never 500.
5. **Use `Promise.all` for parallel endpoint calls** on every refresh. Never sequential.
6. **The drawer fetches its own data** — does not share state with main panels.
7. **URL params sync on every state change** via `history.replaceState` (no page reload).
