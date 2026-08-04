import asyncio
import json
from unittest.mock import patch, AsyncMock

import pytest

from qwen_reverse import client as client_mod
from qwen_reverse.client import create_chat, _run_tool_calls

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


def _state_with(conv="c1", parent="p1", usage=None):
    return {"conv_id": conv, "parent_id_out": parent, "content": "", "usage": usage or {}}


async def _collect(agen):
    return [e async for e in agen]


def _make_send_message_mock(response_chunks, conv_id="c1", parent_id="p1", usage=None):
    """Build an async generator mock for _send_message.

    response_chunks: list of {"type": ..., "data": ...} dicts that will be yielded
    """
    async def fake_send_message(*args, _state=None, **kwargs):
        if _state is not None:
            _state.update(_state_with(conv_id, parent_id, usage))
        for chunk in response_chunks:
            yield chunk
    return fake_send_message


async def test_no_tools_streams_content_token_by_token():
    chunks = [
        {"type": "content", "data": "Hel"},
        {"type": "content", "data": "lo"},
        {"type": "usage", "data": {"input_tokens": 5, "output_tokens": 2}},
        {"type": "done"},
    ]
    fake = _make_send_message_mock(chunks)
    with patch.object(client_mod, "_send_message", side_effect=fake):
        events = await _collect(create_chat(
            "qwen3.8-max",
            [{"role": "user", "content": "hi"}],
            tools=None,
        ))
    contents = [e["data"] for e in events if e["type"] == "content"]
    assert contents == ["Hel", "lo"]
    usage_events = [e for e in events if e["type"] == "usage"]
    assert len(usage_events) == 1
    done = [e for e in events if e["type"] == "done"][0]
    assert done["conversation_id"] == "c1"
    assert done["parent_id"] == "p1"


async def test_emit_tool_calls_yields_tool_calls_and_done():
    chunks = [
        {"type": "content", "data": '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "BA"}}]}'},
        {"type": "usage", "data": {"input_tokens": 3, "output_tokens": 4}},
        {"type": "done"},
    ]
    fake = _make_send_message_mock(chunks)
    with patch.object(client_mod, "_send_message", side_effect=fake):
        events = await _collect(create_chat(
            "qwen3.8-max",
            [{"role": "user", "content": "weather?"}],
            tools=[WEATHER_TOOL],
            emit_tool_calls=True,
        ))
    tool_calls_events = [e for e in events if e["type"] == "tool_calls"]
    assert len(tool_calls_events) == 1
    assert tool_calls_events[0]["data"][0]["function"]["name"] == "get_weather"
    done = [e for e in events if e["type"] == "done"][0]
    assert done["conversation_id"] == "c1"
    content_events = [e for e in events if e["type"] == "content"]
    assert content_events == [], "emit_tool_calls should not re-emit content"


async def test_execute_tool_calls_runs_followup_and_returns_done():
    initial_chunks = [
        {"type": "content", "data": '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "BA"}}]}'},
        {"type": "done"},
    ]
    followup_chunks = [
        {"type": "content", "data": "Sunny"},
        {"type": "done"},
    ]

    call_log = []

    async def fake_send_message(model, conv, parent, prompt, *args, _state=None, **kwargs):
        call_log.append({"conv": conv, "parent": parent, "prompt": prompt})
        if "Tool result" in prompt:
            if _state is not None:
                _state.update(_state_with(conv or "c1", parent or "p1"))
            for chunk in followup_chunks:
                yield chunk
        else:
            if _state is not None:
                _state.update(_state_with(conv or "c1", parent or "p1"))
            for chunk in initial_chunks:
                yield chunk

    with patch.object(client_mod, "_send_message", side_effect=fake_send_message):
        with patch.object(client_mod, "execute_tool_call", new=AsyncMock(return_value='{"result":"27C"}')):
            events = await _collect(create_chat(
                "qwen3.8-max",
                [{"role": "user", "content": "weather?"}],
                tools=[WEATHER_TOOL],
                emit_tool_calls=False,
            ))

    assert any(e["type"] == "content" and e["data"] == "Sunny" for e in events)
    done_events = [e for e in events if e["type"] == "done"]
    assert done_events
    assert done_events[-1]["conversation_id"] == "c1"
    followup_call = next(c for c in call_log if "Tool result" in c["prompt"])
    assert "27C" in followup_call["prompt"]


async def test_no_tool_calls_detected_falls_back_to_normal_done():
    chunks = [
        {"type": "content", "data": "Just text, no JSON tool calls here"},
        {"type": "done"},
    ]
    fake = _make_send_message_mock(chunks)
    with patch.object(client_mod, "_send_message", side_effect=fake):
        events = await _collect(create_chat(
            "qwen3.8-max",
            [{"role": "user", "content": "hi"}],
            tools=[WEATHER_TOOL],
            emit_tool_calls=False,
        ))
    assert any(e["type"] == "content" and e["data"] == "Just text, no JSON tool calls here" for e in events)
    assert [e for e in events if e["type"] == "done"]


GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {"type": "object", "properties": {}},
    },
}


async def test_multiple_tool_calls_all_executed():
    initial_chunks = [
        {"type": "content", "data": '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "BA"}}, {"name": "get_time", "arguments": {}}]}'},
        {"type": "done"},
    ]
    followup_chunks = [{"type": "content", "data": "ok"}, {"type": "done"}]
    call_log = []

    async def fake_send_message(model, conv, parent, prompt, *args, _state=None, **kwargs):
        call_log.append(prompt)
        if _state is not None:
            _state.update(_state_with(conv or "c1", parent or "p1"))
        if "Tool result" in prompt:
            for chunk in followup_chunks:
                yield chunk
        else:
            for chunk in initial_chunks:
                yield chunk

    two_tools = [WEATHER_TOOL, GET_TIME_TOOL]
    with patch.object(client_mod, "_send_message", side_effect=fake_send_message):
        with patch.object(client_mod, "execute_tool_call", new=AsyncMock(side_effect=lambda tc, tools, sess: '{"result": "x"}')):
            events = await _collect(create_chat(
                "qwen3.8-max",
                [{"role": "user", "content": "do both"}],
                tools=two_tools,
                emit_tool_calls=False,
            ))

    tool_result_prompts = [p for p in call_log if "Tool result" in p]
    assert len(tool_result_prompts) == 2, tool_result_prompts
    assert any(e["type"] == "content" and e["data"] == "ok" for e in events)
