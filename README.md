# Qwen-Reverse

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI version](https://img.shields.io/pypi/v/qwen-reverse)](https://pypi.org/project/qwen-reverse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

<p align="center">
  <img src="https://raw.githubusercontent.com/AnonymoDGH/Qwen-Reverse/main/assets/logo_edited.png" width="280" alt="qwen-reverse logo"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/AnonymoDGH/Qwen-Reverse/main/assets/demo.gif" width="480" alt="live streaming demo"/>
</p>

Reverse-engineered async Python client for [chat.qwen.ai](https://chat.qwen.ai) — text chat, streaming with real-time reasoning, tool calling, image and video generation. No official API key required (works anonymously).

```bash
pip install qwen-reverse
```

## Features

- **Chat** — one-shot, streaming (SSE incremental), multi-turn with conversation memory (`conversation_id` / `parent_id` chaining)
- **Real-time reasoning** — `reasoning` events streamed token-by-token before the answer, in both plain and multi-turn mode
- **Tool calling** — OpenAI-style function definitions; either let the SDK execute them (JSON-stringified follow-up) or handle them yourself with `emit_tool_calls=True`
- **Vision** — image upload + chat about it (pass files/URLs/bytes)
- **Image editing (i2i)** — `chat_type="image_edit"`: edit an uploaded image (add/remove/modify) and get back CDN URLs
- **File upload** — `upload()` uploads local files (or URLs/bytes) into the web API for vision/editing chats
- **Image / video generation** — t2i and t2v returning CDN URLs (`cdn.qwenlm.ai`)
- **No account required** — the web API works without a token; OAuth device-flow login (`chat.qwen.ai` account) is also implemented
- **Optional FastAPI server** — OpenAI-compatible `/v1/chat/completions` and `/v1/models` (see `server/`)

## Quickstart

```python
import asyncio
from qwen_reverse import Generative

async def main():
    gen = Generative()
    print(await gen.generate("Explain what a kernel is in 2 lines"))

asyncio.run(main())
```

### Streaming with real-time thinking

```python
import asyncio
from qwen_reverse import Generative

async def main():
    gen = Generative()
    async for event in gen.stream("Explain in 2 sentences what an OS kernel is"):
        if event["type"] == "reasoning":
            print(f"\r\033[90m{event['data']}\033[0m", end="", flush=True)
        elif event["type"] == "content":
            print(event["data"], end="", flush=True)

asyncio.run(main())
```

### Multi-turn conversation

```python
import asyncio
from qwen_reverse import Conversation

async def main():
    conv = Conversation()
    r1 = await conv.send("My name is Popbob and I work with kernels in C.")
    r2 = await conv.send("What is my name?")  # remembers turn 1
    print(r2)  # "Popbob"
    print(conv.conversation_id, conv.parent_id)

asyncio.run(main())
```

### Tool calling (automatic execution)

```python
import asyncio
from qwen_reverse import Conversation

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

async def main():
    conv = Conversation(tools=[WEATHER_TOOL])
    async for event in conv.stream("What's the weather in Buenos Aires?"):
        if event["type"] == "tool_calls":
            print("[tool_calls]", event["data"])
        elif event["type"] == "content":
            print(event["data"], end="", flush=True)

asyncio.run(main())
```

For raw tool-call JSON instead of execution, pass `emit_tool_calls=True` to `stream()`/`create_chat()`.

### Image generation

```python
import asyncio
from qwen_reverse import Image

async def main():
    img = Image()
    urls = await img.generate(
        "A cyberpunk dragon flying over a neon city, anime style",
        aspect_ratio="16:9",
    )
    print(urls[0])  # https://cdn.qwenlm.ai/output/...

asyncio.run(main())
```

### Upload a file + chat with vision

```python
import asyncio
from qwen_reverse import Conversation

async def main():
    conv = Conversation(model="qwen3-vl-plus")  # a vision-capable model
    reply = await conv.send(
        "What is written on the whiteboard?",
        files=["photo.jpg"],  # path | bytes | URL | already-uploaded dict
    )
    print(reply)

asyncio.run(main())
```

`files` accepts local paths, raw `bytes`, `http(s)` URLs (downloaded then uploaded) or a pre-uploaded file dict. Use `upload()` to upload manually:

```python
import asyncio
from qwen_reverse import upload

async def main():
    file_dict = await upload("diagram.png")        # -> {"id", "url", "type", ...}
    # reuse file_dict in multiple chats without re-uploading
asyncio.run(main())
```

### Edit an image (image-to-image)

```python
import asyncio
from qwen_reverse import ImageEdit

async def main():
    img = ImageEdit()  # chat_type="image_edit"
    urls = await img.edit(
        "Add a bold arrow crossing through the logo from top-left to bottom-right, "
        "keep the original violet symbol exactly the same",
        "logo.png",                # path | bytes | URL | uploaded dict
        aspect_ratio="1:1",
    )
    print(urls[0])  # edited image CDN URL

asyncio.run(main())
```

You can also call `Generative(...).edit_image(prompt, image, aspect_ratio=...)` directly.

## API

| Symbol | Description |
|---|---|
| `Generative(model=..., token=...)` | `.generate()`, `.stream()` (events: `reasoning`, `content`, `usage`, `tool_calls`, `done`) |
| `Conversation(token=..., tools=...)` | `.send()`, `.stream()`, `.reset()` — persists `conversation_id`/`parent_id` across turns |
| `Image(model=...)`, `Video(model=...)`, `ImageEdit(model=...)` | `.generate(prompt, aspect_ratio=...)` → list of CDN URLs; `ImageEdit.edit(prompt, image, aspect_ratio=...)` for image-to-image editing |
| `create_chat(model, messages, ...)` | low-level async generator; `conversation_id`/`parent_id` for multi-turn |
| `fetch_models()` | available models (`qwen3.8-max`, `qwen3.7-plus`, `qwen3-vl-plus`, ...) |
| `upload(data, filename=...)` | upload a local path/bytes (or fetch+upload a URL) → file payload dict; returns the web-API file object |
| `resolve_files(files)` | normalize a list of paths/bytes/URLs/dicts into upload-ready payload dicts |
| `upload_file(...)` | low-level file upload (advanced: pass your own session/headers) |
| `start_device_login()` / `complete_device_login(...)` | OAuth device flow |
| `QwenOAuth2Client`, `SharedTokenManager` | token management |
| `generate_cookies()` / `generate_fingerprint()` / `BXUAGenerator` | anti-bot primitives (cookies `ssxmod_itna`, `bx-umidtoken`, fingerprint) |

### Events emitted by `stream()`

| Event | Data | When |
|---|---|---|
| `reasoning` | `str` (chunk) | during `phase: "think"` — the model's chain of thought |
| `content` | `str` (chunk) | during `phase: "answer"` — the actual reply |
| `tool_calls` | `list[dict]` | when the model requests functions |
| `image` / `image_done` | `{"url", "extra"}` / `None` | image generation progress |
| `usage` | `dict` | token usage (`input_tokens`, `output_tokens`, ...) |
| `done` | — | final event; carries `conversation_id` and `parent_id` |

## Authentication

**Anonymous mode works** — no token required for most features (lower rate limits). For higher limits, log in with a `chat.qwen.ai` account:

```python
import asyncio
from qwen_reverse import start_device_login, complete_device_login

async def main():
    client, data = await start_device_login()
    print(data["verification_uri_complete"])  # open in a logged-in browser
    # after authorizing:
    token = await complete_device_login(client, data)
    print(token)
```

## Models

Defaults to `qwen3.8-max`. Others: `qwen3.7-plus`, `qwen3.7-max`, plus vision / image / video models — see `qwen_reverse/models.py` and `fetch_models()`.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Optional server (OpenAI-compatible)

```bash
pip install "qwen-reverse[server]"
cd server
uvicorn app.main:app --port 8000
# GET  /v1/models
# POST /v1/chat/completions  (OpenAI-style, streaming SSE)
```

## How it works

The client replicates what the official web app does against `chat.qwen.ai`:

1. Create a chat via `POST /api/v2/chats/new` (gets a `chat_id`)
2. Stream the response via `POST /api/v2/chat/completions?chat_id=...` with `version: 2.1`, `incremental_output` and a `feature_config` that enables the reasoning stream
3. Multi-turn chaining uses the server's `response_id` (assistant message fid) as the next turn's `parent_id`
4. Anti-bot headers are regenerated per request: `ssxmod_itna` cookies (custom LZW + custom base64), `bx-umidtoken`, fingerprint

## Disclaimer

This project is for **educational and research purposes**. It is not affiliated with or endorsed by Alibaba/Qwen. Use at your own risk — the endpoints may change or the service may rate-limit or block unofficial clients. MIT licensed; reverse-engineering references based on [g4f](https://github.com/xtekky/gpt4free) (MIT).
