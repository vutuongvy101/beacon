"""Brand recommendation generation for the BEACON pipeline.

Extracted from advanced/A4_ToT_Recommendations.ipynb.

Public interface
----------------
recommend_brand_actions(state, method) -> dict
    Generate a recommendation set for a single brand state.

export_recommendations(states, method) -> list[dict]
    Build the full recommendations.json payload for the dashboard.
    Implements a tiered routing policy by default:
      - red / amber crisis states  -> Tree-of-Thought (tot)
      - green crisis states        -> Chain-of-Thought (cot)

Technique justification
-----------------------
Four methods were evaluated on 12 stratified brand-state scenarios in A4:
  CoT          (1 call/scenario)  single linear reasoning chain
  Self-Refine  (3 calls/scenario) draft, critique, revise
  Persona      (4 calls/scenario) parallel expert perspectives, then synthesise
  ToT          (4 calls/scenario) branch generation, evaluation, synthesis

ToT was selected for high-stakes states on the basis of:
  - Highest faithfulness (0.80 vs 0.52-0.55 for peers): recommendations are
    grounded in retrieved evidence posts rather than generic best practice.
  - Consistent pairwise preference over all peers (58-75% win-rate), with the
    Self-Refine and Persona comparisons at 75%.
  - Conditional difficulty advantage: 60-70% pairwise win-rate on red/amber
    states versus 50% on green states, motivating the tiered policy.

CoT is retained as the routine-state default because it matches ToT on
pointwise quality (5.77 vs 5.83) at one-quarter of the token cost.

The ablation showed ToT's advantage requires a capable base model (>=7B).
The tiered policy should be revisited if the deployed model changes.

Model
-----
Generator and judge: qwen2.5:7b via Ollama (local, no API key required).
Configured via shared/config.py (QWEN_LARGE, OLLAMA_BASE_URL).
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from openai import APIConnectionError, APITimeoutError, OpenAI

try:
    from shared.config import OLLAMA_BASE_URL, QWEN_LARGE, QWEN_SMALL
except ImportError:
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    QWEN_LARGE      = "qwen2.5:7b"
    QWEN_SMALL      = "qwen2.5:3b"

# constants

MODEL           = QWEN_LARGE           # generator and judge
WEAK_MODEL      = QWEN_SMALL           # used only in the A4 ablation, not here
TEMP_DET        = 0.2                  # near-deterministic: CoT, judge, synthesis
TEMP_DIV        = 0.7                  # diverse: branch generation
TOT_BRANCHES    = 2                    # candidate strategies per ToT call
PERSONAS        = [                    # expert roles for Persona method
    "a crisis communications director",
    "a product marketing lead",
]
OLLAMA_NUM_CTX  = 4096                 # context window; safe for 6-8 GB VRAM
REQUEST_TIMEOUT = 180                  # seconds; local inference can be slow

REC_SCHEMA = (
    "Return ONLY a JSON array of 4 to 6 objects, no markdown and no prose. "
    'Each object has: "action" (a concrete recommendation), "rationale" '
    "(why, tied to the state), \"evidence\" (which driver or post_id supports "
    'it), "priority" (one of "high", "medium", "low").'
)

# client

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key="ollama",
            base_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1"
            if not OLLAMA_BASE_URL.endswith("/v1")
            else OLLAMA_BASE_URL,
        )
    return _client


# low-level helpers

def _chat(
    prompt: str,
    temperature: float,
    model: str = MODEL,
    max_retries: int = 4,
) -> tuple[str, dict]:
    """Single LLM call with timeout retry. Returns (text, usage_dict)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                temperature=temperature,
                timeout=REQUEST_TIMEOUT,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"options": {"num_ctx": OLLAMA_NUM_CTX}},
            )
            break
        except APITimeoutError as exc:
            last_err = exc
            time.sleep(10 * (attempt + 1))
        except APIConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is 'ollama serve' running?"
            )
    else:
        raise RuntimeError(
            f"Ollama call failed after {max_retries} attempts: {last_err}"
        )
    u = resp.usage
    usage = {
        "calls": 1,
        "in":  getattr(u, "prompt_tokens",     0) or 0,
        "out": getattr(u, "completion_tokens",  0) or 0,
    }
    return resp.choices[0].message.content, usage


def _merge(*usages: dict) -> dict:
    out = {"calls": 0, "in": 0, "out": 0}
    for u in usages:
        for k in out:
            out[k] += u[k]
    return out


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _parse_json(text: str, default):
    try:
        return json.loads(_strip_fences(text))
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", _strip_fences(text), flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return default


def _state_to_text(state: dict) -> str:
    """Render a brand-state dict as a plain-text block for prompting."""
    s = state["sentiment"]
    ev = "\n".join(
        f'  - [{e["post_id"]}] "{e["excerpt"]}" (score {e["score"]})'
        for e in state.get("evidence", [])
    )
    return (
        f'Topic: {state["topic"]}\n'
        f'Crisis level: {state["crisis_level"]}\n'
        f'Sentiment: positive {s["positive"]:.0%}, '
        f'neutral {s["neutral"]:.0%}, '
        f'negative {s["negative"]:.0%}\n'
        f'Negative drivers: {", ".join(state.get("negative_drivers", [])) or "none"}\n'
        f'Positive drivers: {", ".join(state.get("positive_drivers", [])) or "none"}\n'
        f"Supporting evidence posts:\n{ev}"
    )


# recommendation methods

def cot_recommend(state: dict, model: str = MODEL) -> dict:
    """Single linear reasoning chain, then recommend (Wei et al., 2022)."""
    prompt = (
        "You advise OpenAI's brand team. Reason step by step before recommending.\n\n"
        f"{_state_to_text(state)}\n\n"
        "Step 1: Identify the core risk or opportunity.\n"
        "Step 2: Identify affected stakeholders.\n"
        "Step 3: Derive responses that address the drivers, grounded in the evidence.\n\n"
        f"After reasoning, output the recommendations. {REC_SCHEMA}"
    )
    text, usage = _chat(prompt, TEMP_DET, model)
    return {"recommendations": _parse_json(text, []), "_meta": usage}


def self_refine_recommend(state: dict, model: str = MODEL) -> dict:
    """Draft via CoT, self-critique, then revise (Madaan et al., 2023)."""
    draft = cot_recommend(state, model)
    recs  = draft["recommendations"]
    crit  = (
        "Critically review the recommendation set below for this brand state. "
        "Identify concrete weaknesses: coverage gaps, unsupported claims, "
        "vague actions, redundancy. Be specific.\n\n"
        f"State:\n{_state_to_text(state)}\n\nSet (JSON):\n{json.dumps(recs)[:4000]}"
    )
    crit_text, u_crit = _chat(crit, TEMP_DET, model)
    rev = (
        "Revise the recommendation set to fix every weakness in the critique. "
        "Keep what is strong, make vague actions concrete, remove redundancy.\n\n"
        f"State:\n{_state_to_text(state)}\n\n"
        f"Original (JSON):\n{json.dumps(recs)[:3000]}\n\n"
        f"Critique:\n{crit_text[:2000]}\n\n{REC_SCHEMA}"
    )
    rev_text, u_rev = _chat(rev, TEMP_DET, model)
    return {
        "recommendations": _parse_json(rev_text, recs),
        "_meta": _merge(draft["_meta"], u_crit, u_rev),
    }


def persona_recommend(
    state: dict,
    personas: list[str] = PERSONAS,
    model: str = MODEL,
) -> dict:
    """One recommendation set per expert persona, then synthesise."""
    sets, usages = [], []
    for persona in personas:
        p = (
            f"You are {persona} advising OpenAI's brand team. "
            "From your professional perspective specifically, "
            f"recommend brand responses for this state.\n\n"
            f"{_state_to_text(state)}\n\n{REC_SCHEMA}"
        )
        t, u = _chat(p, TEMP_DET, model)
        sets.append(_parse_json(t, []))
        usages.append(u)
    syn = (
        f"Expert advisors ({', '.join(personas)}) each proposed recommendations. "
        "Synthesise one coherent set integrating their complementary views, "
        "keeping the strongest actions and removing redundancy.\n\n"
        f"State:\n{_state_to_text(state)}\n\n"
        f"Advisor sets (JSON):\n{json.dumps(sets)[:5000]}\n\n{REC_SCHEMA}"
    )
    st, us = _chat(syn, TEMP_DET, model)
    return {"recommendations": _parse_json(st, []), "_meta": _merge(*usages, us)}


def tot_recommend(
    state: dict,
    k: int = TOT_BRANCHES,
    model: str = MODEL,
) -> dict:
    """Generate k candidate strategies, evaluate each, synthesise the best.

    Selected as the recommended method for red/amber brand states based on
    A4 experiment results: highest faithfulness (0.80) and consistent pairwise
    preference over all peer methods (58-75% win-rate).
    """
    gen = (
        f"You advise OpenAI's brand team. Propose {k} DISTINCT candidate "
        "response strategies differing in approach (for example reactive "
        "crisis control versus proactive narrative building).\n\n"
        f"{_state_to_text(state)}\n\n"
        f"Return ONLY a JSON array of {k} objects, each with "
        '"strategy_name", "approach" (2 to 3 sentences), '
        '"actions" (3 to 5 concrete actions).'
    )
    gen_text, u_gen = _chat(gen, TEMP_DIV, model)
    branches = _parse_json(gen_text, [])
    if not isinstance(branches, list) or not branches:
        branches = [{"strategy_name": "fallback", "approach": gen_text, "actions": []}]

    scored, eval_usages = [], []
    for b in branches:
        ev = (
            "Score this strategy 1 to 5 on: impact, feasibility, evidence_fit.\n\n"
            f"State:\n{_state_to_text(state)}\n\nStrategy:\n{json.dumps(b)}\n\n"
            'Return ONLY: {"impact": int, "feasibility": int, "evidence_fit": int}.'
        )
        et, eu = _chat(ev, TEMP_DET, model)
        sc = _parse_json(et, {"impact": 3, "feasibility": 3, "evidence_fit": 3})
        sc["total"] = sc.get("impact", 3) + sc.get("feasibility", 3) + sc.get("evidence_fit", 3)
        scored.append({"branch": b, "score": sc})
        eval_usages.append(eu)

    scored.sort(key=lambda x: x["score"]["total"], reverse=True)
    top = scored[: max(1, k - 1)]
    syn = (
        "From the highest-scoring strategies below, synthesise one coherent "
        "recommendation set. Prefer the strongest but fold in complementary "
        "actions without redundancy.\n\n"
        f"State:\n{_state_to_text(state)}\n\n"
        f"Top strategies (JSON):\n{json.dumps(top)[:5000]}\n\n{REC_SCHEMA}"
    )
    syn_text, u_syn = _chat(syn, TEMP_DET, model)
    return {
        "recommendations": _parse_json(syn_text, []),
        "branches": scored,
        "_meta": _merge(u_gen, *eval_usages, u_syn),
    }


# public interface

_DISPATCH = {
    "cot":          cot_recommend,
    "self_refine":  self_refine_recommend,
    "persona":      persona_recommend,
    "tot":          tot_recommend,
}

Method = Literal["cot", "self_refine", "persona", "tot"]


def recommend_brand_actions(
    state: dict,
    method: Method = "tot",
) -> dict:
    """Generate brand-response recommendations for a single brand state.

    Args:
        state: Brand-state dict with keys:
            topic (str), sentiment (dict with positive/neutral/negative floats),
            negative_drivers (list[str]), positive_drivers (list[str]),
            crisis_level (str: "red" | "amber" | "green"),
            evidence (list[dict] with post_id, excerpt, score).
        method: One of "cot", "self_refine", "persona", "tot".
            Defaults to "tot" (recommended for high-stakes states).

    Returns:
        dict with keys:
            method (str): method used.
            recommendations (list[dict]): action objects with action,
                rationale, evidence, priority.
            meta (dict): calls, in-tokens, out-tokens.

    Raises:
        ValueError: If method is not recognised.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(_DISPATCH)}."
        )
    out = _DISPATCH[method](state)
    return {
        "method":          method,
        "recommendations": out["recommendations"],
        "meta":            out["_meta"],
    }


def export_recommendations(
    states: list[dict],
    method: str = "tiered",
    strong: Method = "tot",
    routine: Method = "cot",
) -> list[dict]:
    """Build the recommendations.json payload for the dashboard.

    Args:
        states: List of brand-state dicts (one per issue cluster).
        method: "tiered" routes by crisis_level (red/amber -> strong,
            green -> routine). Any other value is passed directly as the
            method for every state.
        strong: Method used for red/amber states under tiered routing.
            Default "tot" (justified by A4 experiment results).
        routine: Method used for green states under tiered routing.
            Default "cot" (matches ToT quality at one-quarter the cost).

    Returns:
        List of dicts ready to serialise as recommendations.json.
        Each dict has: scenario_id, topic, crisis_level, method,
        recommendations.
    """
    payload = []
    for s in states:
        if method == "tiered":
            chosen = strong if s.get("crisis_level") in {"red", "amber"} else routine
        else:
            chosen = method
        res = recommend_brand_actions(s, method=chosen)
        payload.append({
            "scenario_id":    s.get("scenario_id"),
            "topic":          s.get("topic"),
            "crisis_level":   s.get("crisis_level"),
            "method":         res["method"],
            "recommendations": res["recommendations"],
        })
    return payload