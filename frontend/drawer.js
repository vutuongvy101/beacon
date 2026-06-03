async function openDrawer(type, id) {
  document.getElementById("drawer-overlay").hidden = false;
  document.getElementById("drawer").hidden = false;
  const content = document.getElementById("drawer-content");
  const title = document.getElementById("drawer-title");
  content.innerHTML = '<div class="skeleton"></div>';

  const base = typeof filterState !== "undefined" ? { ...filterState } : {};

  try {
    switch (type) {
      case "post":
        title.textContent = "Post detail";
        await renderPostDrawer(id, base, content);
        break;
      case "entity":
        title.textContent = `Entity: ${id}`;
        await renderEntityDrawer(id, base, content);
        break;
      case "topic":
        title.textContent = "Topic detail";
        await renderTopicDrawer(id, base, content, title);
        break;
      case "sentiment":
        title.textContent = `Sentiment: ${id}`;
        await renderSentimentDrawer(id, base, content);
        break;
      case "week":
        title.textContent = `Week: ${id}`;
        await renderWeekDrawer(id, base, content);
        break;
      default:
        content.innerHTML = "<p>Unknown drawer type.</p>";
    }
  } catch (e) {
    content.innerHTML = `<p>Error: ${escapeHtml(e.message)}</p>`;
  }
}

function closeDrawer() {
  document.getElementById("drawer-overlay").hidden = true;
  document.getElementById("drawer").hidden = true;
}

async function renderPostDrawer(postId, base, content) {
  const res = await api.post("/api/posts", { ...base, post_id: postId, limit: 1, offset: 0 });
  const p = res.posts[0];
  if (!p) {
    content.innerHTML = "<p>Post not found.</p>";
    return;
  }
  const ents = (p.entities || []).map((e) => `<span class="sentiment-chip neutral">${escapeHtml(e.text)}</span>`).join(" ");
  content.innerHTML = `
    <h4>${escapeHtml(p.title)}</h4>
    <p>${escapeHtml(p.clean_text)}</p>
    <p><small>u/${escapeHtml(p.author)} · score ${p.score} · ${new Date(p.created_utc * 1000).toLocaleString()}</small></p>
    <p><span class="sentiment-chip ${p.sentiment}">${p.sentiment}</span> ${(p.sentiment_confidence * 100).toFixed(0)}%</p>
    <p>${ents}</p>
    <a href="${escapeHtml(p.url)}" target="_blank" rel="noopener">View on Reddit</a>
  `;
}

async function renderEntityDrawer(entityText, base, content) {
  const res = await api.post("/api/posts", {
    ...base,
    entity_text: entityText,
    limit: 25,
    offset: 0,
  });
  content.innerHTML = `<p>${res.total} posts mentioning <strong>${escapeHtml(entityText)}</strong></p>` +
    res.posts.map((p) => postCardHtml(p)).join("");
  bindPostCards(content);
}

async function renderSentimentDrawer(sentiment, base, content) {
  const res = await api.post("/api/posts", {
    ...base,
    sentiment,
    limit: 25,
    offset: 0,
  });
  content.innerHTML = `<p>${res.total} ${sentiment} posts</p>` +
    res.posts.map((p) => postCardHtml(p)).join("");
  bindPostCards(content);
}

async function renderWeekDrawer(week, base, content) {
  const res = await api.post("/api/posts", {
    ...base,
    week,
    limit: 25,
    offset: 0,
  });
  content.innerHTML = `<p>${res.total} posts in ${escapeHtml(week)}</p>` +
    res.posts.map((p) => postCardHtml(p)).join("");
  bindPostCards(content);
}

async function renderTopicDrawer(topicId, base, content, titleEl) {
  const topicsRes = await api.post("/api/topics", base);
  const topic = (topicsRes.topics || []).find((t) => t.topic_id === topicId);
  if (topic) {
    titleEl.textContent = topic.label;
    content.innerHTML = `
      <p><strong>Keywords:</strong> ${(topic.keywords || []).join(", ")}</p>
      <p><strong>Posts:</strong> ${topic.post_count} · <span class="sentiment-chip ${topic.dominant_sentiment}">${topic.dominant_sentiment}</span></p>
    `;
  }

  const ev = await api.post("/api/evidence", { ...base, topic_id: topicId });
  const wrap = document.createElement("div");
  wrap.innerHTML = `<h4>Evidence</h4>` + renderEvidenceCards(ev.evidence || [], ev.query);
  content.appendChild(wrap);
  bindEvidenceClicks(wrap);

  const postsRes = await api.post("/api/posts", { ...base, limit: 10, offset: 0 });
  const topicPostIds = new Set();
  if (topic) {
    const allTopics = await api.post("/api/topics", base);
    const full = (allTopics.topics || []).find((t) => t.topic_id === topicId);
  }
  const pRes = await api.post("/api/posts", { ...base, limit: 15, offset: 0 });
  const section = document.createElement("div");
  section.innerHTML = `<h4>Related posts</h4>` + pRes.posts.slice(0, 8).map((p) => postCardHtml(p)).join("");
  content.appendChild(section);
  bindPostCards(section);
}

function postCardHtml(p) {
  return `
    <div class="post-card" data-post-id="${escapeAttr(p.post_id)}">
      <h4>${escapeHtml(p.title || p.post_id)}</h4>
      <p>${escapeHtml(p.clean_text)}</p>
      <span class="sentiment-chip ${p.sentiment}">${p.sentiment}</span>
    </div>
  `;
}

function bindPostCards(container) {
  container.querySelectorAll(".post-card[data-post-id]").forEach((card) => {
    card.addEventListener("click", () => openDrawer("post", card.dataset.postId));
  });
}

document.getElementById("drawer-close")?.addEventListener("click", closeDrawer);
document.getElementById("drawer-overlay")?.addEventListener("click", closeDrawer);
