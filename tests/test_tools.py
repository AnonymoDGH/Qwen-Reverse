import asyncio
import json

from qwen_reverse import (
    parse_tool_response,
    inject_tools,
    build_tool_system_prompt,
    execute_tool_call,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_parse_tool_response_json():
    content = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "BA"}}]}'
    calls = parse_tool_response(content, ["get_weather"])
    assert calls and calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["city"] == "BA"


def test_parse_tool_response_filters_unknown():
    content = '{"tool_calls": [{"name": "nope", "arguments": {}}]}'
    assert parse_tool_response(content, ["get_weather"]) is None


def test_parse_tool_response_stringified():
    content = '[Tool call: get_weather] (id=call_0)\nArguments: {"city": "BA"}'
    calls = parse_tool_response(content, ["get_weather"])
    assert calls and calls[0]["function"]["name"] == "get_weather"


def test_inject_tools_adds_system_prompt():
    msgs = [{"role": "user", "content": "hola"}]
    out = inject_tools(msgs, [WEATHER_TOOL])
    assert out[0]["role"] == "system"
    assert "get_weather" in out[0]["content"]


def test_inject_tools_noop_without_tools():
    msgs = [{"role": "user", "content": "hola"}]
    assert inject_tools(msgs, None) is msgs


def test_build_tool_system_prompt_lists_names():
    prompt = build_tool_system_prompt([WEATHER_TOOL])
    assert "get_weather" in prompt


def test_execute_tool_call_unknown():
    result = asyncio.run(execute_tool_call(
        {"function": {"name": "ghost", "arguments": "{}"}},
        [WEATHER_TOOL],
    ))
    assert "Unknown tool" in result
