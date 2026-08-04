"""Model and chat-type catalogs for chat.qwen.ai.

The complete model list is fetched at runtime from `/api/models` (see
`fetch_models` in client.py). These static lists mirror the ones shipped
in gpt4free's Qwen provider and serve as fallback when no token is
available or the endpoint is unreachable.
"""

#: All chat types supported by the Qwen web API.
CHAT_TYPES = [
    "t2t",          # plain text chat
    "search",       # web search
    "artifacts",    # code artifacts
    "web_dev",      # web development mode
    "deep_research",
    "t2i",          # text-to-image
    "image_edit",
    "t2v",          # text-to-video
    "travel",       # travel planner
    "learn",        # learning mode
    "slides",       # presentation slides
]

#: Supported aspect ratios for image/video generation.
ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]

text_models = [
    'qwen3.8-max', 'qwen3.8-max-preview', 'qwen3.7-plus', 'qwen3.7-max', 'qwen3.6-plus', 'qwen3.6-max-preview', 'qwen3.6-27b',
    'qwen-latest-series-invite-beta-v24', 'qwen-latest-series-invite-beta-v16', 'qwen3.5-plus',
    'qwen3.5-omni-plus', 'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-max-2026-03-08',
    'qwen3.6-plus-preview', 'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b', 'qwen3.5-omni-flash', 'qwen3.5-27b',
    'qwen3.5-35b-a3b', 'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28', 'qwen3-coder-plus', 'qwen3-vl-plus',
    'qwen3-omni-flash-2025-12-01']

image_models = [
    'qwen3.8-max', 'qwen3.8-max-preview', 'qwen3.7-plus', 'qwen3.7-max', 'qwen3.6-plus', 'qwen3.6-27b', 'qwen3.5-plus', 'qwen3.5-omni-plus',
    'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b', 'qwen3.5-omni-flash',
    'qwen3.5-27b', 'qwen3.5-35b-a3b', 'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28', 'qwen3-coder-plus',
    'qwen3-vl-plus', 'qwen3-omni-flash-2025-12-01']

vision_models = [
    'qwen3.8-max', 'qwen3.8-max-preview', 'qwen3.7-plus', 'qwen3.6-plus', 'qwen3.6-27b', 'qwen-latest-series-invite-beta-v16', 'qwen3.5-plus',
    'qwen3.5-omni-plus', 'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b',
    'qwen3.5-omni-flash', 'qwen3.5-27b', 'qwen3.5-35b-a3b', 'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28',
    'qwen3-coder-plus', 'qwen3-vl-plus', 'qwen3-omni-flash-2025-12-01']

#: Full fallback list (union of the above).
models = [
    'qwen3.8-max', 'qwen3.8-max-preview', 'qwen3.7-plus', 'qwen3.7-max', 'qwen3.6-plus', 'qwen3.6-max-preview', 'qwen3.6-27b',
    'qwen-latest-series-invite-beta-v24', 'qwen-latest-series-invite-beta-v16', 'qwen3.5-plus',
    'qwen3.5-omni-plus', 'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-max-2026-03-08', 'qwen3.6-plus-preview',
    'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b', 'qwen3.5-omni-flash', 'qwen3.5-27b', 'qwen3.5-35b-a3b',
    'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28', 'qwen3-coder-plus', 'qwen3-vl-plus',
    'qwen3-omni-flash-2025-12-01']

#: Models that default to thinking-on only when no reasoning_effort is explicitly
#: given. An explicit "none" disables thinking for all models (verified
#: supported by chat.qwen.ai for every model).
MODELS_REQUIRE_THINKING = [
    "qwen3.8-max-preview", "qwen3.8-max", "qwen3.7-plus", "qwen3.7-max", "qwen3.6-plus",
]

#: Valid values for ``reasoning_effort``.
REASONING_EFFORT_LEVELS = ("none", "low", "medium", "high")

DEFAULT_MODEL = "qwen3.8-max"
