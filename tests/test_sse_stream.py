import json
import pytest

from qwen_reverse.client import _sse_stream


class FakeContent:
    def __init__(self, raw_lines):
        # Concatenate into a single byte buffer; readline() splits on the
        # first b"\n" exactly like aiohttp's StreamReader.readline(), so
        # blank-line SSE separators are preserved.
        self._buf = b""
        for l in raw_lines:
            self._buf += l.encode() if isinstance(l, str) else l

    async def readline(self):
        if not self._buf:
            return b""
        idx = self._buf.find(b"\n")
        if idx == -1:
            line = self._buf
            self._buf = b""
        else:
            line = self._buf[: idx + 1]
            self._buf = self._buf[idx + 1 :]
        return line


class FakeResponse:
    def __init__(self, lines):
        self.content = FakeContent(lines)


async def _collect(agen):
    return [e async for e in agen]


async def test_simple_data_line():
    resp = FakeResponse(["data: {\"hello\": 1}\n", "\n"])
    events = await _collect(_sse_stream(resp))
    assert events == [{"hello": 1}]


async def test_data_no_space():
    resp = FakeResponse(["data:{\"hello\": 2}\n", "\n"])
    events = await _collect(_sse_stream(resp))
    assert events == [{"hello": 2}]


async def test_crlf_line_endings():
    resp = FakeResponse(["data: {\"hello\": 3}\r\n", "\r\n"])
    events = await _collect(_sse_stream(resp))
    assert events == [{"hello": 3}]


async def test_multiline_data_concatenated_with_newline():
    resp = FakeResponse([
        "data: {\"part1\": \"a\",\n",
        "data:  \"part2\": \"b\"}\n",
        "\n",
    ])
    events = await _collect(_sse_stream(resp))
    assert events == [{"part1": "a", "part2": "b"}]


async def test_done_terminates_stream():
    resp = FakeResponse([
        "data: {\"x\": 1}\n\n",
        "data: [DONE]\n\n",
        "data: {\"never\": true}\n\n",
    ])
    events = await _collect(_sse_stream(resp))
    assert events == [{"x": 1}]


async def test_invalid_json_is_skipped():
    resp = FakeResponse([
        "data: not json\n\n",
        "data: {\"ok\": true}\n\n",
    ])
    events = await _collect(_sse_stream(resp))
    assert events == [{"ok": True}]


async def test_empty_lines_without_buffer_are_ignored():
    resp = FakeResponse(["\n", "\n", "data: {\"a\": 1}\n\n", "\n"])
    events = await _collect(_sse_stream(resp))
    assert events == [{"a": 1}]


async def test_stream_terminates_when_exhausted():
    resp = FakeResponse(["data: {\"a\": 1}\n\n"])
    events = await _collect(_sse_stream(resp))
    assert events == [{"a": 1}]
