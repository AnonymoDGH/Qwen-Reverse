import json

from qwen_reverse.client import (
    _build_chat_payload,
    _build_message_payload,
    _events_from_chunk,
    _sse_events_from_text,
    _build_feature_config,
)
from qwen_reverse.baxia import build_trajectory
from qwen_reverse.browser import is_challenge


def test_chats_new_payload_matches_live_shape():
    p = _build_chat_payload("qwen3.7-plus", "t2t", "guest", 1793131863266)
    assert p["chatId"] == ""
    assert p["models"] == ["qwen3.7-plus"]
    assert p["chat_mode"] == "guest"
    assert "title" not in p


def test_message_payload_matches_live_shape():
    p = _build_message_payload(
        model="qwen3.7-plus",
        conversation_id="cid-1",
        parent_id=None,
        prompt="hi",
        chat_type="t2t",
        chat_mode="guest",
        feature_config=_build_feature_config("qwen3.7-plus", "none", None),
        message_id="fid-1",
        timestamp_ms=1793131863266,
        files=None,
        aspect_ratio=None,
    )
    assert p["chatId"] == p["chat_id"] == "cid-1"
    assert p["parentId"] == ""
    assert p["parent_id"] is None
    msg = p["messages"][0]
    assert msg["fid"] == "fid-1"
    assert msg["parentId"] is None
    assert msg["id"] is None
    assert msg["model"] == ""
    assert isinstance(msg["childrenIds"], list) and msg["childrenIds"]
    assert msg["timestamp"] == 1793131863  # seconds, not ms


def test_events_from_created_frame():
    chunk = {"response.created": {"chat_id": "c", "parent_id": "p", "response_id": "r"}}
    events, usage, rid = _events_from_chunk(chunk)
    assert events == []
    assert usage == {}
    assert rid == "r"


def test_events_from_thinking_summary():
    chunk = {
        "choices": [{"delta": {
            "phase": "thinking_summary", "content": "",
            "extra": {
                "summary_title": {"content": ["title"]},
                "summary_thought": {"content": ["a\n", "b"]},
            },
        }}],
        "response_id": "r1",
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    events, usage, rid = _events_from_chunk(chunk)
    assert events == [("reasoning", "a\nb")]
    assert usage["input_tokens"] == 5
    assert rid == "r1"


def test_events_from_answer():
    chunk = {"choices": [{"delta": {"phase": "answer", "content": "OK"}}], "response_id": "r"}
    events, _, _ = _events_from_chunk(chunk)
    assert events == [("content", "OK")]


def test_legacy_reasoning_still_supported():
    chunk = {"choices": [{"delta": {"reasoning": "thinking"}}]}
    events, _, _ = _events_from_chunk(chunk)
    assert events == [("reasoning", "thinking")]


async def _collect(agen):
    return [e async for e in agen]


async def test_sse_events_from_text_new_format():
    chunks = [
        'data: {"response.created":{"response_id":"r1","parent_id":"p1"}}\n\n',
        'data: {"choices":[{"delta":{"phase":"answer","content":"OK"}}],"response_id":"r1"}\n\n',
    ]

    async def gen():
        for c in chunks:
            yield c

    events = await _collect(_sse_events_from_text(gen()))
    assert len(events) == 2
    assert events[0]["response.created"]["response_id"] == "r1"
    assert events[1]["choices"][0]["delta"]["content"] == "OK"


def test_is_challenge_markers():
    assert is_challenge('{"ret":["FAIL_SYS_USER_VALIDATE"]}')
    assert is_challenge('{"rgv587_flag":"sm"}')
    assert is_challenge("<!doctype html><meta name=aliyun_waf_aa>")
    assert not is_challenge('data: {"choices": []}')


def test_trajectory_is_monotonic_and_bounded():
    class Rng:
        def uniform(self, a, b):
            return (a + b) / 2

    pts = list(build_trajectory(10.0, 50.0, 110.0, 50.0, Rng()))
    xs = [x for x, _, _ in pts]
    assert xs == sorted(xs)
    assert min(xs) >= 10.0
    assert max(xs) <= 110.0
    assert xs[-1] == 110.0
