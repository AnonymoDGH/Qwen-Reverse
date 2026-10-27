from qwen_reverse.server.app import (
    _parse_model_suffix,
    _map_size_to_aspect_ratio,
    _anthropic_to_openai,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicTool,
)

def test_parse_model_suffix():
    assert _parse_model_suffix("qwen3.8-max") == ("qwen3.8-max", "t2t", None)
    assert _parse_model_suffix("qwen3.8-max-search") == ("qwen3.8-max", "search", None)
    assert _parse_model_suffix("qwen3.8-max-thinking") == ("qwen3.8-max", "t2t", "high")
    assert _parse_model_suffix("qwen3.8-max-search-thinking") == ("qwen3.8-max", "search", "high")
    assert _parse_model_suffix("qwen3.8-max-image") == ("qwen3.8-max", "t2i", None)
    assert _parse_model_suffix("claude-opus-4-6-search") == ("qwen3.8-max", "search", None)

def test_map_size_to_aspect_ratio():
    assert _map_size_to_aspect_ratio("1024x1024") == "1:1"
    assert _map_size_to_aspect_ratio("1792x1024") == "16:9"
    assert _map_size_to_aspect_ratio("1024x1792") == "9:16"
    assert _map_size_to_aspect_ratio("1024x768") == "4:3"
    assert _map_size_to_aspect_ratio("1280x720") == "16:9"
    assert _map_size_to_aspect_ratio("768x1024") == "3:4"


def test_anthropic_translation_keeps_tools_and_emit_mode():
    body = AnthropicRequest(
        model="claude-opus-4-6-thinking",
        messages=[AnthropicMessage(role="user", content="hi")],
        tools=[AnthropicTool(name="get_weather", input_schema={"type": "object"})],
    )
    kw = _anthropic_to_openai(body)
    assert kw["emit_tool_calls"] is True
    assert kw["reasoning_effort"] == "high"
    assert kw["tools"][0]["function"]["name"] == "get_weather"
