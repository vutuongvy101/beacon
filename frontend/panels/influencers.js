function renderInfluencers(data) {
  const table = document.getElementById("influencer-table");
  clearPanelSkeleton("panel-influencers");
  const rows = data.influencers || [];

  if (!rows.length) {
    table.innerHTML = `<tbody><tr><td colspan="5">${emptyStateHtml()}</td></tr></tbody>`;
    return;
  }

  table.innerHTML = `
    <thead>
      <tr><th>#</th><th>Entity</th><th>Mentions</th><th>Engagement</th><th>Sentiment</th></tr>
    </thead>
    <tbody>
      ${rows.map((r) => `
        <tr data-entity="${escapeAttr(r.entity)}">
          <td>${r.rank}</td>
          <td>${escapeHtml(r.entity)} <small>${escapeHtml(r.entity_label)}</small></td>
          <td>${r.mention_count}</td>
          <td>${Math.round(r.engagement_score)}</td>
          <td><span class="sentiment-chip ${r.dominant_sentiment}">${r.dominant_sentiment}</span></td>
        </tr>
      `).join("")}
    </tbody>
  `;

  table.querySelectorAll("tbody tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      if (typeof openDrawer === "function") {
        openDrawer("entity", tr.dataset.entity);
      }
    });
  });
}
