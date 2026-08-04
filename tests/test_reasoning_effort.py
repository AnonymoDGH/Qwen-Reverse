import pytest

from qwen_reverse.client import _build_feature_config
from qwen_reverse.models import MODELS_REQUIRE_THINKING, REASONING_EFFORT_LEVELS
from qwen_reverse.client import QwenError


def test_explicit_none_disables_thinking_for_require_list_models():
    # Previously the MODELS_REQUIRE_THINKING override forced thinking ON here.
    for model in MODELS_REQUIRE_THINKING:
        fc = _build_feature_config(model, "none", None)
        assert fc["thinking_enabled"] is False, model
        assert fc["output_schema"] == "phase"
        assert "thinking_budget" in fc  # fast-mode branch


def test_explicit_none_disables_thinking_for_non_require_models():
    fc = _build_feature_config("qwen3-coder-plus", "none", None)
    assert fc["thinking_enabled"] is False


def test_none_is_case_insensitive():
    fc = _build_feature_config("qwen3.8-max", "NONE", None)
    assert fc["thinking_enabled"] is False


def test_low_uses_fast_mode_and_lower_budget():
    fc = _build_feature_config("qwen3-coder-plus", "low", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Fast"
    assert fc["thinking_budget"] == 40960
    assert fc["auto_search"] is True


def test_medium_uses_auto_mode_and_default_budget():
    fc = _build_feature_config("qwen3-coder-plus", "medium", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Auto"
    assert fc["thinking_budget"] == 81920
    assert fc["auto_search"] is True


def test_high_uses_auto_mode_and_higher_budget():
    fc = _build_feature_config("qwen3-coder-plus", "high", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Auto"
    assert fc["thinking_budget"] == 163840
    assert fc["auto_search"] is True


def test_effort_levels_are_exhaustive():
    """All four REASONING_EFFORT_LEVELS produce a valid config."""
    for effort in REASONING_EFFORT_LEVELS:
        fc = _build_feature_config("qwen3.8-max", effort, None)
        assert isinstance(fc, dict)


def test_invalid_effort_raises():
    for bad in ("ultra", "max", "", "none ", " LOW"):
        with pytest.raises(QwenError):
            _build_feature_config("qwen3.8-max", bad, None)


def test_require_model_default_enables_thinking_when_effort_not_none():
    # A non-"none" effort on a require-list model still enables thinking.
    fc = _build_feature_config("qwen3.8-max", "high", None)
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Auto"


def test_custom_thinking_mode_honored_when_enabled():
    fc = _build_feature_config("qwen3.8-max", "high", "Thinking")
    assert fc["thinking_enabled"] is True
    assert fc["thinking_mode"] == "Thinking"
    assert fc["auto_thinking"] is False  # "Thinking" != "Auto"


def test_fast_mode_when_disabled_has_no_auto_search():
    fc = _build_feature_config("qwen3.8-max", "none", None)
    # disabled branch must not leak thinking-on keys
    assert "research_mode" not in fc
    assert "auto_search" not in fc
    assert fc["thinking_budget"] == 81920
