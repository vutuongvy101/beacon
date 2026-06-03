function renderEvidenceCards(evidence, query) {
  if (!evidence.length) {
    return `<p>No evidence for "${escapeHtml(query || "")}".</p>`;
  }
  return evidence.map((e) => `
    <div class="post-card" data-post-id="${escapeAttr(e.post_id)}">
      <h4>${escapeHtml(e.post_id)} · score ${e.score}</h4>
      <p>${escapeHtml(e.text)}</p>
      <span class="sentiment-chip ${e.sentiment}">${e.sentiment}</span>
      <small> similarity ${(e.similarity_score || 0).toFixed(2)}</small>
    </div>
  `).join("");
}

function bindEvidenceClicks(container) {
  container.querySelectorAll(".post-card[data-post-id]").forEach((card) => {
    card.addEventListener("click", () => {
      if (typeof openDrawer === "function") {
        openDrawer("post", card.dataset.postId);
      }
    });
  });
}
