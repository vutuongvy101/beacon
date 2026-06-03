# Beacon dashboard diagrams

Mermaid source files for the OpenAI Brand Monitor architecture and feature flows.

## Files

| File | Description |
|------|-------------|
| `00-architecture-overview.mmd` | System architecture (startup, data, backend, frontend) |
| `01-app-bootstrap.mmd` | Initial page load sequence |
| `02-filter-refresh.mmd` | Date/search filter → parallel API refresh |
| `03-kpi-row.mmd` | KPI cards from `/api/overview` |
| `04-crisis-monitor.mmd` | Crisis badge and panel |
| `05-chart-sentiment-donut.mmd` | Sentiment doughnut chart (`chart-sentiment-donut`) |
| `06-chart-sentiment-trend.mmd` | Weekly stacked sentiment bar (`chart-sentiment-trend`) |
| `07-chart-topics-bar.mmd` | Topic clusters horizontal bar (`chart-topics-bar`) |
| `08-panel-entity-mentions.mmd` | Entity mentions list |
| `09-panel-influencers-table.mmd` | Top voices table |
| `10-panel-executive-summary.mmd` | Executive summary (`GET /api/summary`) |
| `11-panel-suggested-qa.mmd` | Suggested Q&A chips |
| `12-panel-qa-answer.mmd` | Q&A chip click and dashboard refresh |
| `13-drawer-post-detail.mmd` | Drawer: single post |
| `14-drawer-entity-sentiment-week.mmd` | Drawer: entity, sentiment, or week drill-down |
| `15-drawer-topic-evidence.mmd` | Drawer: topic with RAG evidence |

## Render

Preview in VS Code/Cursor with a Mermaid extension, or export to SVG/PNG:

```bash
# Install once: npm install -g @mermaid-js/mermaid-cli
for f in diagrams/*.mmd; do
  mmdc -i "$f" -o "${f%.mmd}.svg"
done
```

Or paste any `.mmd` file into [mermaid.live](https://mermaid.live).

## Chart → file map

| UI element | Diagram |
|------------|---------|
| Sentiment donut | `05-chart-sentiment-donut.mmd` |
| Sentiment trend | `06-chart-sentiment-trend.mmd` |
| Topic clusters bar | `07-chart-topics-bar.mmd` |

Non-chart panels and drawers have their own sequence diagrams as listed above.
