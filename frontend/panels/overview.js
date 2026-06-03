let donutChart = null;
let trendChart = null;

function renderOverview(data) {
  const k = data.kpis || {};
  const row = document.getElementById("kpi-row");
  row.innerHTML = `
    <div class="kpi-card"><div class="kpi-value">${k.total_posts ?? 0}</div><div class="kpi-label">Posts</div></div>
    <div class="kpi-card"><div class="kpi-value">${k.positive_pct ?? 0}%</div><div class="kpi-label">Positive</div></div>
    <div class="kpi-card"><div class="kpi-value">${k.negative_pct ?? 0}%</div><div class="kpi-label">Negative</div></div>
    <div class="kpi-card"><div class="kpi-value">${k.neutral_pct ?? 0}%</div><div class="kpi-label">Neutral</div></div>
    <div class="kpi-card"><div class="kpi-value">${k.unique_authors ?? 0}</div><div class="kpi-label">Authors</div></div>
    <div class="kpi-card"><div class="kpi-value">${k.total_entities ?? 0}</div><div class="kpi-label">Entities</div></div>
  `;

  const crisis = data.crisis || {};
  const badge = document.getElementById("crisis-badge");
  const level = (crisis.level || "green").toLowerCase();
  badge.className = `crisis-badge ${level}`;
  badge.innerHTML = `<i class="ti ti-alert-triangle"></i> ${level.toUpperCase()}: ${crisis.reason || ""}`;

  const crisisPanel = document.getElementById("crisis-content");
  crisisPanel.innerHTML = `
    <p><strong>Level:</strong> ${level}</p>
    <p>${crisis.reason || "No crisis data."}</p>
    <p><strong>Negative ratio (filtered):</strong> ${((crisis.negative_ratio || 0) * 100).toFixed(1)}%</p>
  `;

  renderDonut(data.sentiment_donut || []);
  renderTrend(data.sentiment_trend || []);
}

function renderDonut(slices) {
  const canvas = document.getElementById("chart-sentiment-donut");
  const panel = document.getElementById("panel-sentiment");
  if (!slices.length) {
    if (donutChart) { donutChart.destroy(); donutChart = null; }
    return;
  }
  if (donutChart) donutChart.destroy();
  donutChart = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: slices.map((s) => s.label),
      datasets: [{
        data: slices.map((s) => s.count),
        backgroundColor: slices.map((s) => s.color),
      }],
    },
    options: {
      onClick: (_, els) => {
        if (els.length && typeof openDrawer === "function") {
          const label = slices[els[0].index].label;
          openDrawer("sentiment", label);
        }
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

function renderTrend(trend) {
  const canvas = document.getElementById("chart-sentiment-trend");
  if (!trend.length) {
    if (trendChart) { trendChart.destroy(); trendChart = null; }
    return;
  }
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: trend.map((t) => t.week),
      datasets: [
        { label: "Positive", data: trend.map((t) => t.positive), backgroundColor: "#22C55E" },
        { label: "Negative", data: trend.map((t) => t.negative), backgroundColor: "#EF4444" },
        { label: "Neutral", data: trend.map((t) => t.neutral), backgroundColor: "#94A3B8" },
      ],
    },
    options: {
      scales: { x: { stacked: true }, y: { stacked: true } },
      onClick: (_, els) => {
        if (els.length && typeof openDrawer === "function") {
          const week = trend[els[0].index].week;
          openDrawer("week", week);
        }
      },
    },
  });
}

function updateSearchBadge(meta) {
  const el = document.getElementById("search-badge");
  if (!meta || !meta.matched_entity) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.textContent = `Also matches entity: ${meta.matched_entity} (${meta.matched_entity_count || 0} posts)`;
  el.onclick = () => {
    document.getElementById("search-input").value = meta.matched_entity;
    if (typeof filterState !== "undefined") {
      filterState.search = meta.matched_entity;
      syncStateToURL();
      refreshDashboard();
    }
  };
}

function showPanelSkeleton(id) {
  const el = document.getElementById(id);
  if (el && !el.querySelector(".skeleton")) {
    const sk = document.createElement("div");
    sk.className = "skeleton";
    el.appendChild(sk);
  }
}

function clearPanelSkeleton(id) {
  document.querySelectorAll(`#${id} .skeleton`).forEach((n) => n.remove());
}
