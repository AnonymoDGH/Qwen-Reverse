import asyncio
import importlib
from unittest.mock import AsyncMock, patch

import pytest

from qwen_reverse import client as client_mod
from qwen_reverse.tts import TTS

tts_mod = importlib.import_module("qwen_reverse.tts")


async def _collect(agen):
    return [e async for e in agen]


def test_tts_public_exports():
    assert callable(TTS)
    assert callable(tts_mod.tts)
    assert callable(tts_mod.fetch_tts_config)


def test_wav_header():
    h = tts_mod._wav_header(10, sample_rate=24000)
    assert h.startswith(b"RIFF")
    assert b"WAVE" in h
    assert h[24:28] == (24000).to_bytes(4, "little")


def test_tts_class_init():
    t = TTS()
    assert t.model == "qwen3.8-max"
    assert t.token is None
    assert t.timeout == 120


@pytest.mark.asyncio
async def test_tts_returns_decoded_audio(monkeypatch):
    done_events = [
        {"type": "content", "data": "hola"},
        {"type": "done", "conversation_id": "chat-1", "parent_id": "resp-1"},
    ]

    async def fake_create_chat(*args, **kwargs):
        for e in done_events:
            yield e

    class FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    sse_chunks = [
        {"choices": [{"delta": {"tts": "aGVsbG8g"}}]},
        {"choices": [{"delta": {"tts": "d29ybGQ="}}]},
    ]

    async def fake_sse(resp):
        for c in sse_chunks:
            yield c

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return FakeResp()

    async def fake_raise_for_status(resp):
        return None

    monkeypatch.setattr(client_mod, "create_chat", fake_create_chat)
    monkeypatch.setattr(tts_mod, "_attach_extra_headers", AsyncMock())
    monkeypatch.setattr(tts_mod, "_sse_stream", fake_sse)
    monkeypatch.setattr(tts_mod, "_raise_for_status", fake_raise_for_status)

    with patch.object(tts_mod, "_get_headers", return_value={}):
        with patch.object(client_mod.aiohttp, "ClientSession", return_value=FakeSession()):
            audio = await tts_mod.tts("hola")
    assert audio == b"hello world"


@pytest.mark.asyncio
async def test_tts_wraps_wav(monkeypatch):
    done_events = [
        {"type": "done", "conversation_id": "chat-1", "parent_id": "resp-1"},
    ]

    async def fake_create_chat(*args, **kwargs):
        for e in done_events:
            yield e

    async def fake_sse(resp):
        yield {"choices": [{"delta": {"tts": "aGVsbG8="}}]}

    class FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return FakeResp()

    monkeypatch.setattr(client_mod, "create_chat", fake_create_chat)
    monkeypatch.setattr(tts_mod, "_attach_extra_headers", AsyncMock())
    monkeypatch.setattr(tts_mod, "_sse_stream", fake_sse)
    monkeypatch.setattr(tts_mod, "_raise_for_status", AsyncMock())

    with patch.object(tts_mod, "_get_headers", return_value={}):
        with patch.object(client_mod.aiohttp, "ClientSession", return_value=FakeSession()):
            audio = await tts_mod.tts("hola", wrap_wav=True)
    assert audio.startswith(b"RIFF")


def test_fetch_tts_config_url():
    assert "tts/config" in tts_mod.TTS_CONFIG_URL


def test_chat_mode_guest_when_no_token():
    assert client_mod._send_message is not None
    assert client_mod.QWEN_URL == "https://chat.qwen.ai"


def test_tts_stream_audio_public():
    assert callable(tts_mod.stream_audio)
    assert callable(TTS().stream)


@pytest.mark.asyncio
async def test_tts_stream_audio_yields_chunks(monkeypatch):
    async def fake_resolve(text, model, token, proxy, timeout, reasoning_effort, kwargs):
        return "chat-1", "resp-1"

    async def fake_chunks(conversation_id, response_id, token, proxy, timeout):
        yield b"hello "
        yield b"world"

    monkeypatch.setattr(tts_mod, "_resolve_voice_target", fake_resolve)
    monkeypatch.setattr(tts_mod, "_tts_audio_chunks", fake_chunks)

    got = []
    async for chunk in tts_mod.stream_audio("hola"):
        got.append(chunk)
    assert got == [b"hello ", b"world"]


@pytest.mark.asyncio
async def test_tts_class_stream(monkeypatch):
    seen = {}

    async def fake_stream(text, **kwargs):
        seen["text"] = text
        seen["token"] = kwargs.get("token")
        yield b"a"
        yield b"b"

    monkeypatch.setattr(tts_mod, "stream_audio", fake_stream)
    t = TTS(token="tok-1", model="qwen3.7-plus", proxy="http://p", timeout=60)
    got = [c async for c in t.stream("hola")]
    assert got == [b"a", b"b"]
    assert seen["text"] == "hola"
    assert seen["token"] == "tok-1"
