function renderEntities(data) {
  const list = document.getElementById("entity-list");
  clearPanelSkeleton("panel-entities");
  const entities = data.entities || [];

  if (!entities.length) {
    list.innerHTML = emptyStateHtml();
    return;
  }

  list.innerHTML = entities.map((e) => `
    <div class="entity-item" data-entity="${escapeAttr(e.text)}">
      <div>
        <span class="entity-name">${escapeHtml(e.text)}</span>
        <span class="entity-meta"> · ${escapeHtml(e.label)} · ${e.mention_count} mentions</span>
      </div>
      <span class="sentiment-chip ${e.dominant_sentiment}">${e.dominant_sentiment}</span>
    </div>
  `).join("");

  list.querySelectorAll(".entity-item").forEach((el) => {
    el.addEventListener("click", () => {
      if (typeof openDrawer === "function") {
        openDrawer("entity", el.dataset.entity);
      }
    });
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function escapeAttr(s) {
  return String(s).replace(/"/g, "&quot;");
}
