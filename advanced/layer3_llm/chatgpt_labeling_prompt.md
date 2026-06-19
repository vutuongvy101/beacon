# ChatGPT labeling prompt — Layer 3 pseudo-labels

Copy everything below the line into ChatGPT (GPT-4o or later recommended), then paste the contents of `outputs/posts_for_labeling.json` immediately after it.

---

You are a brand-monitoring analyst labelling Reddit posts about **OpenAI / ChatGPT**.

For **each post** in the JSON array I provide, assign three labels. Return **one JSON array only** — no markdown fences, no commentary — with one object per post, in the **same order** as the input.

## Output schema (per post)

```json
{
  "post_id": "<copy from input>",
  "crisis_severity": 0,
  "sentiment_score": 0.0,
  "topic": "general_discussion",
  "rationale": "one sentence"
}
```

## crisis_severity (integer 0–3)

| Level | Meaning | Examples |
|-------|---------|----------|
| **0** | No brand concern | Product praise, neutral questions, technical tips, memes |
| **1** | Minor complaint | Single-user frustration, billing annoyance, feature gripe |
| **2** | Escalating concern | Widespread backlash, trust erosion, policy controversy, model removals angering many users |
| **3** | Active crisis | Outages breaking production, data breach / safety scandal, lethal-weapons / ethics allegations with reputational risk, boycott calls |

**Calibration tips:**
- Enthusiastic developer praise (Codex, new models) → **0**, not 3.
- Image-model comparisons or capability demos → **0**.
- Individual “I’m upset they removed 4o” → **1** or **2** depending on tone; not **3** unless mass mobilisation.
- “ChatGPT is down” with many affected users → **3** for outage posts.
- Allegations OpenAI builds lethal autonomous weapons → **3**, strongly negative sentiment.

## sentiment_score (float −1.0 to +1.0)

Continuous score for overall tone **toward OpenAI/ChatGPT** in the post (title + comments included):
- **+1.0** = strongly positive (excitement, praise)
- **0.0** = neutral / mixed
- **−1.0** = strongly negative (outrage, betrayal)

Judge the **dominant** sentiment in the thread, not individual comment outliers.

## topic (exactly one of these strings)

- `product_releases` — new models, features, Sora, DALL·E, voice mode, image gen
- `pricing_subscriptions` — Plus/Pro pricing, billing, plan changes
- `api_developer` — API, Codex, rate limits, tokens, developer tooling
- `safety_ethics` — alignment, privacy, surveillance, harm, weapons, content policy
- `corporate_leadership` — Sam Altman, board, governance, company direction
- `competition` — vs Anthropic, Google, Grok, open-source alternatives
- `reliability_outages` — downtime, errors, performance, “is it down?”
- `general_discussion` — everything else

## Few-shot examples

**Post:** "GPT-4o voice mode is amazing, best update yet"
→ `{"post_id":"ex0","crisis_severity":0,"sentiment_score":0.9,"topic":"product_releases","rationale":"Positive product praise, no concern."}`

**Post:** "Subscription went up again, not sure it's worth $20"
→ `{"post_id":"ex1","crisis_severity":1,"sentiment_score":-0.4,"topic":"pricing_subscriptions","rationale":"Individual pricing complaint."}`

**Post:** "OpenAI deleted 8 models overnight with no warning — people are grieving"
→ `{"post_id":"ex2","crisis_severity":2,"sentiment_score":-0.7,"topic":"product_releases","rationale":"Mass user backlash over forced migration."}`

**Post:** "Is ChatGPT down? I can't send anything, outage for hours"
→ `{"post_id":"ex3","crisis_severity":3,"sentiment_score":-0.8,"topic":"reliability_outages","rationale":"Active service outage affecting users."}`

**Post:** "Head of Robotics resigned — OpenAI building lethal AI weapons with no human authorisation"
→ `{"post_id":"ex4","crisis_severity":3,"sentiment_score":-0.9,"topic":"safety_ethics","rationale":"Active ethics/reputational crisis allegation."}`

---

## Your task

Label every post in the JSON array below. Return **only** the JSON array of label objects.

**Input posts:**
