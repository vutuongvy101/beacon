let topicsChart = null;

function renderTopics(data) {
  const panel = document.getElementById("panel-topics");
  clearPanelSkeleton("panel-topics");
  const topics = data.topics || [];
  const canvas = document.getElementById("chart-topics-bar");

  if (!topics.length) {
    if (topicsChart) { topicsChart.destroy(); topicsChart = null; }
    panel.querySelector(".chart-wrap").innerHTML = emptyStateHtml();
    return;
  }

  if (!canvas.parentElement) {
    panel.querySelector(".chart-wrap").innerHTML = '<canvas id="chart-topics-bar"></canvas>';
  }

  const c = document.getElementById("chart-topics-bar");
  if (topicsChart) topicsChart.destroy();
  topicsChart = new Chart(c, {
    type: "bar",
    data: {
      labels: topics.map((t) => t.label.slice(0, 40)),
      datasets: [{
        label: "Posts",
        data: topics.map((t) => t.post_count),
        backgroundColor: "#6366f1",
      }],
    },
    options: {
      indexAxis: "y",
      onClick: (_, els) => {
        if (els.length && typeof openDrawer === "function") {
          openDrawer("topic", topics[els[0].index].topic_id);
        }
      },
    },
  });
}

function emptyStateHtml() {
  return `<div class="empty-state"><i class="ti ti-chart-bar"></i>No data for current filters<button type="button" class="btn-ghost" onclick="clearAllFilters()">Clear filters</button></div>`;
}
