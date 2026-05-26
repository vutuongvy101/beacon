# ner_config.py
# B2 NER — External configuration file
# Contains: CUSTOM_PATTERNS (EntityRuler) and GROUND_TRUTH (evaluation labels)
# Import in notebook with:
#   from ner_config import CUSTOM_PATTERNS, GROUND_TRUTH

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CUSTOM ENTITY DICTIONARY
# Add new entities here as the project evolves.
# Format: {"label": "ENTITY_TYPE", "pattern": "exact text to match"}
# Entity types used: ORG, PERSON, PRODUCT, LAW
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_PATTERNS = [
    # ── OpenAI — Organization ─────────────────────────────────────────────
    {"label": "ORG", "pattern": "OpenAI"},
    {"label": "ORG", "pattern": "Open AI"},

    # ── OpenAI — Products / Models ────────────────────────────────────────
    {"label": "PRODUCT", "pattern": "ChatGPT"},
    {"label": "PRODUCT", "pattern": "GPT-3.5"},
    {"label": "PRODUCT", "pattern": "GPT-4"},
    {"label": "PRODUCT", "pattern": "GPT-4o"},
    {"label": "PRODUCT", "pattern": "GPT-4o mini"},
    {"label": "PRODUCT", "pattern": "GPT-5"},
    {"label": "PRODUCT", "pattern": "GPT-5.4"},
    {"label": "PRODUCT", "pattern": "GPT-5.5"},
    {"label": "PRODUCT", "pattern": "GPT-5.5 Codex"},
    {"label": "PRODUCT", "pattern": "GPT-Realtime-2"},
    {"label": "PRODUCT", "pattern": "Codex"},
    {"label": "PRODUCT", "pattern": "OpenAI Codex"},
    {"label": "PRODUCT", "pattern": "fine-tuning API"},
    {"label": "PRODUCT", "pattern": "fine tuning API"},
    {"label": "PRODUCT", "pattern": "o3"},
    {"label": "PRODUCT", "pattern": "o4"},
    {"label": "PRODUCT", "pattern": "o4-mini"},
    {"label": "PRODUCT", "pattern": "Sora"},
    {"label": "PRODUCT", "pattern": "Sora 2"},
    {"label": "PRODUCT", "pattern": "DALL-E"},
    {"label": "PRODUCT", "pattern": "DALL-E 3"},
    {"label": "PRODUCT", "pattern": "Whisper"},
    {"label": "PRODUCT", "pattern": "Operator"},
    {"label": "PRODUCT", "pattern": "Deep Research"},

    # ── OpenAI — Key People ───────────────────────────────────────────────
    {"label": "PERSON", "pattern": "Sam Altman"},
    {"label": "PERSON", "pattern": "Altman"},
    {"label": "PERSON", "pattern": "Ilya Sutskever"},
    {"label": "PERSON", "pattern": "Ilya"},
    {"label": "PERSON", "pattern": "Greg Brockman"},
    {"label": "PERSON", "pattern": "Brockman"},
    {"label": "PERSON", "pattern": "Mira Murati"},
    {"label": "PERSON", "pattern": "Murati"},
    {"label": "PERSON", "pattern": "Andrej Karpathy"},
    {"label": "PERSON", "pattern": "Karpathy"},

    # ── Competitors — Organizations ───────────────────────────────────────
    {"label": "ORG", "pattern": "Anthropic"},
    {"label": "ORG", "pattern": "Google"},
    {"label": "ORG", "pattern": "Google DeepMind"},
    {"label": "ORG", "pattern": "DeepMind"},
    {"label": "ORG", "pattern": "Microsoft"},
    {"label": "ORG", "pattern": "Azure"},
    {"label": "ORG", "pattern": "GitHub"},
    {"label": "ORG", "pattern": "Slack"},
    {"label": "ORG", "pattern": "Bing"},
    {"label": "ORG", "pattern": "Cursor"},
    {"label": "ORG", "pattern": "xAI"},
    {"label": "ORG", "pattern": "SpaceX"},
    {"label": "ORG", "pattern": "Mistral AI"},
    {"label": "ORG", "pattern": "Mistral"},
    {"label": "ORG", "pattern": "Cohere"},
    {"label": "ORG", "pattern": "Runway"},
    {"label": "ORG", "pattern": "Pika"},
    {"label": "ORG", "pattern": "Safe Superintelligence"},
    {"label": "ORG", "pattern": "SSI"},
    {"label": "ORG", "pattern": "European Union"},
    {"label": "ORG", "pattern": "News Corp"},

    # ── Competitor Products / Models ──────────────────────────────────────
    {"label": "PRODUCT", "pattern": "Gemini"},
    {"label": "PRODUCT", "pattern": "Gemini Omni"},
    {"label": "PRODUCT", "pattern": "Gemini 3.1"},
    {"label": "PRODUCT", "pattern": "Claude"},
    {"label": "PRODUCT", "pattern": "Claude 3"},
    {"label": "PRODUCT", "pattern": "Claude Code"},
    {"label": "PRODUCT", "pattern": "Claude Haiku"},
    {"label": "PRODUCT", "pattern": "Sonnet"},
    {"label": "PRODUCT", "pattern": "Opus"},
    {"label": "PRODUCT", "pattern": "Opus 4.7"},
    {"label": "PRODUCT", "pattern": "Mythos"},
    {"label": "PRODUCT", "pattern": "Grok"},
    {"label": "PRODUCT", "pattern": "Llama"},
    {"label": "PRODUCT", "pattern": "Llama 3"},
    {"label": "PRODUCT", "pattern": "Kimi 2.6"},
    {"label": "PRODUCT", "pattern": "DeepSeek V4"},
    {"label": "PRODUCT", "pattern": "Mistral 7B"},
    {"label": "PRODUCT", "pattern": "Copilot"},
    {"label": "PRODUCT", "pattern": "GitHub Copilot"},
    {"label": "PRODUCT", "pattern": "Azure OpenAI"},
    {"label": "PRODUCT", "pattern": "Google Assistant"},
    {"label": "PRODUCT", "pattern": "Alexa"},
    {"label": "PRODUCT", "pattern": "Stable Diffusion"},
    {"label": "PRODUCT", "pattern": "Midjourney"},
    {"label": "PRODUCT", "pattern": "Adobe Firefly"},

    # ── Competitor People ─────────────────────────────────────────────────
    {"label": "PERSON", "pattern": "Demis Hassabis"},
    {"label": "PERSON", "pattern": "Yann LeCun"},
    {"label": "PERSON", "pattern": "Geoffrey Hinton"},
    {"label": "PERSON", "pattern": "Mark Zuckerberg"},
    {"label": "PERSON", "pattern": "Zuckerberg"},
    {"label": "PERSON", "pattern": "Elon Musk"},
    {"label": "PERSON", "pattern": "Musk"},
    {"label": "PERSON", "pattern": "Daniel Gross"},

    # ── Regulatory / Policy ───────────────────────────────────────────────
    {"label": "LAW", "pattern": "EU AI Act"},
    {"label": "LAW", "pattern": "AI Act"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REGEX ENTITY PATTERNS
# Used to capture flexible AI product/model names that may not be matched
# perfectly by exact EntityRuler patterns.
# Format: (LABEL, REGEX_PATTERN)
# ─────────────────────────────────────────────────────────────────────────────

REGEX_PATTERNS = [
    # GPT models: GPT-4, GPT-4o, GPT-5.5, GPT-5.5 Codex, etc.
    ("PRODUCT", r"\bGPT[- ]?\d(?:\.\d+)?[a-zA-Z]*(?:\s+Codex)?\b"),

    # OpenAI realtime models: GPT-Realtime-2
    ("PRODUCT", r"\bGPT[- ]?Realtime[- ]?\d+\b"),

    # Claude models: Claude 3, Claude Code, Claude Haiku
    ("PRODUCT", r"\bClaude(?:\s+(?:\d+|Code|Haiku|Sonnet|Opus))?\b"),

    # Opus/Sonnet/Haiku versions: Opus 4.7, Sonnet 4, etc.
    ("PRODUCT", r"\b(?:Opus|Sonnet|Haiku)\s+\d(?:\.\d+)?\b"),

    # Gemini versions: Gemini 3.1, Gemini Omni
    ("PRODUCT", r"\bGemini(?:\s+(?:\d(?:\.\d+)?|Omni))?\b"),

    # Sora versions: Sora 2
    ("PRODUCT", r"\bSora(?:\s+\d+)?\b"),

    # Other common AI model/product names
    ("PRODUCT", r"\b(?:Grok|Llama|Copilot|Midjourney|Whisper|DALL-E|DeepSeek)\b"),

    # xAI as organisation
    ("ORG", r"\bxAI\b"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GROUND TRUTH LABELS
# Manually labeled entities for evaluation against the three NER approaches.
# Format: post_id -> {entity_type -> [list of entity strings]}
# Add more labeled posts here as you annotate additional Reddit data.
# Aim for 30+ posts before the final report submission.
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH = {
    "1tcpe8y": {
        "ORG":     ["Anthropic", "GitHub", "Slack"],
        "PRODUCT": ["GPT-5.5 Codex", "Opus 4.7", "Claude Code", "Sonnet"],
    },
    "1tbhvyn": {
        "PERSON":  ["Musk", "Sam Altman"],
        "ORG":     ["OpenAI"],
    },
    "1tb7r0d": {
        "PERSON":  ["Sam Altman", "Musk"],
        "ORG":     ["OpenAI"],
    },
    "1taqgfd": {
        "PERSON":  ["Mira Murati"],
        "ORG":     ["OpenAI"],
        "PRODUCT": ["GPT-Realtime-2"],
    },
    "1taaec9": {
        "ORG":     ["OpenAI", "Cursor"],
        "PRODUCT": ["GPT-5.5", "GPT-5.4", "Codex", "Opus 4.7"],
    },
    "1ta99ss": {
        "ORG":     ["OpenAI", "Bing"],
        "PRODUCT": ["Gemini Omni", "Sora 2"],
    },
    "1t9vuw5": {
        "ORG":     ["Anthropic"],
        "PRODUCT": ["Claude", "Claude Code", "Opus 4.7", "Mythos"],
    },
    "1t7bhhl": {
        "PERSON":  ["Elon Musk"],
        "ORG":     ["OpenAI"],
        "PRODUCT": ["ChatGPT"],
    },
    "1t76dqd": {
        "ORG":     ["OpenAI"],
        "PRODUCT": ["GPT-5.5"],
    },
    "1t6sisf": {
        "ORG":     ["OpenAI"],
        "PRODUCT": ["fine-tuning API"],
    },
    "1t5tn1n": {
        "PERSON":  ["Sam Altman", "Mira Murati"],
    },
    "1t5kz8t": {
        "PERSON":  ["Elon"],
        "ORG":     ["Anthropic", "OpenAI", "xAI", "SpaceX"],
        "PRODUCT": ["Opus 4.7", "Claude Code", "GPT-5.5", "Codex"],
    },
    "1t14fpg": {
        "PERSON":  ["Sam Altman"],
        "ORG":     ["Anthropic"],
    },
    "1szoe78": {
        "ORG":     ["Anthropic", "Adobe", "Autodesk", "Canva"],
        "PRODUCT": ["Claude", "Claude Code", "Adobe Creative Cloud",
                    "Photoshop", "Premiere", "Illustrator",
                    "Blender", "Autodesk Fusion", "Ableton", "Affinity"],
    },
    "1szj0s8": {
        "PERSON":  ["Musk", "Altman", "Brockman"],
        "ORG":     ["OpenAI"],
    },
}