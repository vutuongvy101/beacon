async function loadSummary() {
  const el = document.getElementById("summary-content");
  try {
    const data = await api.get("/api/summary");
    if (!data.summary_text) {
      el.innerHTML = emptyStateHtml();
      return;
    }
    el.innerHTML = `
      <p><small>Strategy: ${escapeHtml(data.strategy || "")}</small></p>
      <p>${escapeHtml(data.summary_text)}</p>
      <div class="rec"><strong>Recommendation</strong><p>${escapeHtml(data.recommendation || "")}</p></div>
    `;
  } catch {
    el.innerHTML = "<p>Summary unavailable.</p>";
  }
}
