import pytest

from qwen_reverse.client import _build_feature_config
from qwen_reverse.models import MODELS_REQUIRE_THINKING, REASONING_EFFORT_LEVELS
from qwen_reverse.client import QwenError


def test_explicit_none_disables_thinking_for_require_list_models():
    for model in MODELS_REQUIRE_THINKING:
        fc = _build_feature_config(model, "none", None)
        assert fc["thinking_enabled"] is False, model
        assert fc["output_schema"] == "phase"
        assert "thinking_budget" not in fc


def test_explicit_none_disables_thinking_for_non_require_models():
    fc = _build_feature_config("qwen3-coder-plus", "none", None)
    assert fc["thinking_enabled"] is False


def test_none_is_case_insensitive():
    fc = _build_feature_config("qwen3.8-max", "NONE", None)
    assert fc["thinking_enabled"] is False


def test_default_none_effort_enables_auto_thinking():
    # None means "web default": the SPA sends thinking_enabled=true / Auto.
    fc = _build_feature_config("qwen3.8-max", None, None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Auto"
    assert fc["auto_thinking"] is True
    assert fc["auto_search"] is True


def test_low_uses_fast_mode():
    fc = _build_feature_config("qwen3-coder-plus", "low", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Fast"
    assert fc["auto_thinking"] is False
    assert fc["auto_search"] is True


def test_medium_uses_auto_mode():
    fc = _build_feature_config("qwen3-coder-plus", "medium", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Auto"
    assert fc["auto_thinking"] is True


def test_high_uses_auto_mode():
    fc = _build_feature_config("qwen3-coder-plus", "high", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Auto"


def test_no_thinking_budget_is_sent():
    # Live 0.3.26 feature_config has no thinking_budget key at any level.
    for effort in REASONING_EFFORT_LEVELS:
        fc = _build_feature_config("qwen3.8-max", effort, None)
        assert "thinking_budget" not in fc, effort


def test_effort_levels_are_exhaustive():
    for effort in REASONING_EFFORT_LEVELS:
        fc = _build_feature_config("qwen3.8-max", effort, None)
        assert isinstance(fc, dict)


def test_invalid_effort_raises():
    for bad in ("ultra", "max", "", "none ", " LOW"):
        with pytest.raises(QwenError):
            _build_feature_config("qwen3.8-max", bad, None)


def test_custom_thinking_mode_honored_when_enabled():
    fc = _build_feature_config("qwen3.8-max", "high", "Thinking")
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Thinking"
    assert fc["auto_thinking"] is False  # "Thinking" != "Auto"


def test_disabled_branch_has_no_thinking_keys():
    fc = _build_feature_config("qwen3.8-max", "none", None)
    assert "research_mode" not in fc
    assert "auto_search" not in fc
    assert "thinking_mode" not in fc
