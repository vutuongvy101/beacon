const filterState = {
  date_from: null,
  date_to: null,
  search: "",
};

let isLoading = false;
let initialLoad = true;
let dateBounds = { min: null, max: null };

function debounce(fn, ms) {
  let t;
  return (...args) =>
    new Promise((resolve, reject) => {
      clearTimeout(t);
      t = setTimeout(async () => {
        try {
          resolve(await fn(...args));
        } catch (e) {
          reject(e);
        }
      }, ms);
    });
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function syncStateToURL() {
  const params = new URLSearchParams();
  if (filterState.date_from) params.set("from", filterState.date_from);
  if (filterState.date_to) params.set("to", filterState.date_to);
  if (filterState.search) params.set("q", filterState.search);
  const qs = params.toString();
  history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
}

function syncStateFromURL() {
  const params = new URLSearchParams(window.location.search);
  const from = params.get("from");
  const to = params.get("to");
  const q = params.get("q");
  if (from) filterState.date_from = parseInt(from, 10);
  if (to) filterState.date_to = parseInt(to, 10);
  if (q !== null) {
    filterState.search = q;
    document.getElementById("search-input").value = q;
  }
}

function showLoadingState() {
  if (initialLoad) {
    document.getElementById("loading-overlay").hidden = false;
  }
  ["panel-topics", "panel-entities", "panel-influencers"].forEach(showPanelSkeleton);
}

function hideLoadingState() {
  document.getElementById("loading-overlay").hidden = true;
  initialLoad = false;
}

function showErrorState(e) {
  const banner = document.getElementById("error-banner");
  banner.textContent = e.message || String(e);
  banner.hidden = false;
  setTimeout(() => { banner.hidden = true; }, 5000);
}

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
    updateDateBoundsFromTrend(overview.sentiment_trend);
  } catch (e) {
    showErrorState(e);
  } finally {
    isLoading = false;
    hideLoadingState();
  }
}, 300);

function updateDateBoundsFromTrend(trend) {
  if (!trend || !trend.length || dateBounds.max) return;
}

function setupDatePicker() {
  const preset = document.getElementById("date-preset");
  const custom = document.getElementById("date-custom");
  const fromInput = document.getElementById("date-from");
  const toInput = document.getElementById("date-to");

  preset.addEventListener("change", () => {
    const v = preset.value;
    custom.hidden = v !== "custom";
    const now = Math.floor(Date.now() / 1000);
    if (v === "all") {
      filterState.date_from = null;
      filterState.date_to = null;
    } else if (v === "7d") {
      filterState.date_from = now - 7 * 86400;
      filterState.date_to = now;
    } else if (v === "30d") {
      filterState.date_from = now - 30 * 86400;
      filterState.date_to = now;
    } else if (v === "custom") {
      applyCustomDates();
      return;
    }
    syncStateToURL();
    refreshDashboard();
  });

  function applyCustomDates() {
    if (fromInput.value) {
      filterState.date_from = Math.floor(new Date(fromInput.value).getTime() / 1000);
    }
    if (toInput.value) {
      const d = new Date(toInput.value);
      d.setHours(23, 59, 59, 999);
      filterState.date_to = Math.floor(d.getTime() / 1000);
    }
    syncStateToURL();
    refreshDashboard();
  }

  fromInput.addEventListener("change", applyCustomDates);
  toInput.addEventListener("change", applyCustomDates);
}

function setupSearchInput() {
  const input = document.getElementById("search-input");
  input.addEventListener("input", () => {
    filterState.search = input.value.trim();
    syncStateToURL();
    refreshDashboard();
  });
}

async function setupSuggestedQA() {
  try {
    const { questions } = await api.get("/api/suggested-qa");
    const grid = document.getElementById("qa-grid");
    grid.innerHTML = questions.map((q) => `
      <button type="button" class="qa-chip" data-id="${escapeAttr(q.id)}">${escapeHtml(q.text)}</button>
    `).join("");
    grid.querySelectorAll(".qa-chip").forEach((btn) => {
      btn.addEventListener("click", () => onSuggestedQuestionClick(btn.dataset.id));
    });
  } catch (e) {
    document.getElementById("qa-grid").innerHTML = "<p>Could not load questions.</p>";
  }
}

async function onSuggestedQuestionClick(questionId) {
  try {
    const result = await api.post("/api/qa", { question_id: questionId });
    if (result.suggested_search) {
      document.getElementById("search-input").value = result.suggested_search;
      filterState.search = result.suggested_search;
    }
    const answerEl = document.getElementById("qa-answer");
    answerEl.hidden = false;
    answerEl.textContent = "";
    for (const char of result.answer || "") {
      answerEl.textContent += char;
      await sleep(8);
    }
    syncStateToURL();
    refreshDashboard();
  } catch (e) {
    showErrorState(e);
  }
}

function clearAllFilters() {
  filterState.date_from = null;
  filterState.date_to = null;
  filterState.search = "";
  document.getElementById("search-input").value = "";
  document.getElementById("date-preset").value = "all";
  document.getElementById("date-custom").hidden = true;
  document.getElementById("search-badge").hidden = true;
  syncStateToURL();
  refreshDashboard();
}

document.getElementById("clear-filters").addEventListener("click", clearAllFilters);

document.addEventListener("DOMContentLoaded", async () => {
  setupDatePicker();
  setupSearchInput();
  syncStateFromURL();
  await setupSuggestedQA();
  await loadSummary();
  await refreshDashboard();
});
