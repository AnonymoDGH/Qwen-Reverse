"""Text-to-speech (TTS) support for chat.qwen.ai.

The web UI's "read aloud" button calls ``POST /api/v2/tts/completions``
with the id of the assistant message to voice. This module mirrors that
flow: it first generates a chat response to obtain the ``chat_id`` /
``response_id`` pair, then streams the synthesized audio from the TTS
endpoint and returns the raw audio bytes.
"""

import base64
import time
from typing import AsyncIterator, Optional

import aiohttp

from .client import (
    QWEN_URL,
    QwenError,
    _attach_extra_headers,
    _get_headers,
    _raise_for_status,
    _sse_stream,
)
from .models import DEFAULT_MODEL

TTS_CONFIG_URL = (
    f"{QWEN_URL}/api/v2/tts/config"
    "?omni_speakers=v1&audio_tts_speakers=v1"
    "&omni_language=v1&audio_tts_language=v1"
)


def _wav_header(data_len: int, sample_rate: int = 24000) -> bytes:
    """Build a minimal RIFF/WAV header for raw 16-bit mono PCM audio."""
    byte_rate = sample_rate * 2
    header = b"RIFF" + (36 + data_len).to_bytes(4, "little") + b"WAVE"
    header += b"fmt " + (16).to_bytes(4, "little")
    header += (1).to_bytes(2, "little")          # PCM
    header += (1).to_bytes(2, "little")          # mono
    header += sample_rate.to_bytes(4, "little")
    header += byte_rate.to_bytes(4, "little")
    header += (2).to_bytes(2, "little")          # block align
    header += (16).to_bytes(2, "little")         # bits per sample
    header += b"data" + data_len.to_bytes(4, "little")
    return header


async def fetch_tts_config(token: Optional[str] = None, proxy: Optional[str] = None) -> dict:
    """Fetch available TTS languages and speaker voices.

    Returns ``{"audio_tts_language": [...], "audio_tts_speakers": [...],
    "omni_language": [...], "omni_speakers": [...]}`` as returned by the API.
    """
    headers = _get_headers(token)
    await _attach_extra_headers(headers, proxy)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.get(TTS_CONFIG_URL, proxy=proxy) as resp:
            resp.raise_for_status()
            data = await resp.json()
    return data.get("data", {})


async def _resolve_voice_target(
    text: str,
    model: str,
    token: Optional[str],
    proxy: Optional[str],
    timeout: int,
    reasoning_effort: Optional[str],
    kwargs: dict,
) -> tuple[str, str]:
    """Run a chat turn to obtain the (chat_id, response_id) pair to voice."""
    from .client import create_chat

    conversation_id: Optional[str] = None
    response_id: Optional[str] = None
    async for evt in create_chat(
        model,
        [{"role": "user", "content": text}],
        token=token,
        proxy=proxy,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        **kwargs,
    ):
        if evt["type"] == "done":
            conversation_id = evt.get("conversation_id") or conversation_id
            response_id = evt.get("parent_id") or response_id
    if not conversation_id or not response_id:
        raise QwenError("TTS: could not determine chat / response id from the chat turn")
    return conversation_id, response_id


async def _tts_audio_chunks(
    conversation_id: str,
    response_id: str,
    token: Optional[str],
    proxy: Optional[str],
    timeout: int,
) -> AsyncIterator[bytes]:
    """Yield raw audio bytes as they stream in from the TTS endpoint."""
    headers = _get_headers(token)
    await _attach_extra_headers(headers, proxy)
    payload = {
        "chat_id": conversation_id,
        "timestamp": int(time.time() * 1000),
        "messages": [
            {"id": response_id, "role": "assistant", "sub_chat_type": "tts"},
        ],
    }
    async with aiohttp.ClientSession(
        headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        async with session.post(
            f"{QWEN_URL}/api/v2/tts/completions?chat_id={conversation_id}",
            json=payload,
            proxy=proxy,
        ) as resp:
            await _raise_for_status(resp)
            async for chunk in _sse_stream(resp):
                error = chunk.get("error", {})
                if error:
                    raise QwenError(f'{error.get("code")}: {error.get("details", error)}')
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                tts_data = delta.get("tts")
                if tts_data:
                    yield base64.b64decode(tts_data)


async def tts(
    text: str,
    model: str = DEFAULT_MODEL,
    token: Optional[str] = None,
    proxy: Optional[str] = None,
    timeout: int = 120,
    reasoning_effort: Optional[str] = "none",
    wrap_wav: bool = False,
    sample_rate: int = 24000,
    **kwargs,
) -> bytes:
    """Synthesize speech for ``text`` and return raw audio bytes.

    The TTS endpoint voices an already-generated assistant message, so this
    helper first runs a normal chat turn with ``text`` to obtain the
    ``chat_id`` / ``response_id`` pair, then requests audio for it.

    By default the raw PCM16 bytes are returned. Pass ``wrap_wav=True`` to
    get a complete RIFF/WAV file (``sample_rate`` defaults to 24000).

    For low-latency playback while synthesis is still in progress, use
    :func:`stream_audio` instead.
    """
    conversation_id, response_id = await _resolve_voice_target(
        text, model, token, proxy, timeout, reasoning_effort, kwargs
    )
    chunks = [c async for c in _tts_audio_chunks(
        conversation_id, response_id, token, proxy, timeout
    )]
    if not chunks:
        raise QwenError("TTS: no audio data returned")
    audio = b"".join(chunks)
    if wrap_wav:
        audio = _wav_header(len(audio), sample_rate) + audio
    return audio


async def stream_audio(
    text: str,
    model: str = DEFAULT_MODEL,
    token: Optional[str] = None,
    proxy: Optional[str] = None,
    timeout: int = 120,
    reasoning_effort: Optional[str] = "none",
    **kwargs,
) -> AsyncIterator[bytes]:
    """Stream raw PCM16 audio chunks as they arrive from the TTS endpoint.

    Yields decoded audio bytes incrementally so callers can play them back
    while synthesis is still running, avoiding the need to wait for the full
    file. Each yield is a base64-decoded SSE chunk (the chunk size depends on
    the server).
    """
    conversation_id, response_id = await _resolve_voice_target(
        text, model, token, proxy, timeout, reasoning_effort, kwargs
    )
    async for chunk in _tts_audio_chunks(
        conversation_id, response_id, token, proxy, timeout
    ):
        yield chunk


class TTS:
    """Text-to-speech client mirroring the web UI's "read aloud" feature.

    Usage::

        tts = TTS()
        audio = await tts.synthesize("Hello world", wrap_wav=True)
        with open("hello.wav", "wb") as fh:
            fh.write(audio)

    Anonymous mode works (guest chat). ``fetch_config()`` lists the
    available speaker voices and languages.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model
        self.token = token
        self.proxy = proxy
        self.timeout = timeout

    async def synthesize(self, text: str, **kwargs) -> bytes:
        return await tts(
            text,
            model=kwargs.pop("model", self.model),
            token=kwargs.pop("token", self.token),
            proxy=kwargs.pop("proxy", self.proxy),
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """Stream raw PCM16 audio chunks as they arrive (see :func:`stream_audio`)."""
        return stream_audio(
            text,
            model=kwargs.pop("model", self.model),
            token=kwargs.pop("token", self.token),
            proxy=kwargs.pop("proxy", self.proxy),
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    async def fetch_config(self) -> dict:
        return await fetch_tts_config(token=self.token, proxy=self.proxy)
