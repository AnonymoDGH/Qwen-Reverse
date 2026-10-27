import json
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from qwen_reverse import (
    Image,
    ImageEdit,
    DEFAULT_MODEL,
    create_chat,
    fetch_models,
    QwenError,
    QwenAuthError,
    QwenRateLimitError,
)

try:
    from .config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("qwen_api")


# Map Anthropic model names to qwen model names so Claude Code
# can use qwen3.8-max without knowing the internal name.
ANTHROPIC_TO_QWEN = {
    "claude-opus-4-6": "qwen3.8-max",
    "claude-sonnet-4-6": "qwen3.8-max",
    "claude-3-5-sonnet-20241022": "qwen3.8-max",
    "claude-3-opus-20240229": "qwen3.8-max",
    "claude-3-haiku-20240307": "qwen3.8-max",
}


SPECIAL_SUFFIX_MAP = {
    "-deep-research": "deep_research",
    "-deepresearch": "deep_research",
    "-web-dev": "web_dev",
    "-webdev": "web_dev",
    "-full-stack": "web_dev",
    "-fullstack": "web_dev",
    "-artifacts": "artifacts",
    "-artifact": "artifacts",
    "-slides": "slides",
    "-slide": "slides",
    "-learn": "learn",
    "-travel": "travel",
    "-search": "search",
    "-image": "t2i",
    "-t2i": "t2i",
    "-video": "t2v",
    "-t2v": "t2v",
}


def _parse_model_suffix(model_name: str) -> tuple[str, str, str | None]:
    """
    Translates Anthropic models and parses specialized model suffixes:
    - -web-dev / -full-stack -> chat_type: "web_dev"
    - -deep-research -> chat_type: "deep_research"
    - -artifacts -> chat_type: "artifacts"
    - -slides -> chat_type: "slides"
    - -learn -> chat_type: "learn"
    - -travel -> chat_type: "travel"
    - -search -> chat_type: "search"
    - -thinking -> reasoning_effort: "high"
    - -image / -t2i -> chat_type: "t2i"
    - -video / -t2v -> chat_type: "t2v"
    Returns: (base_model, chat_type, reasoning_effort)
    """
    chat_type = "t2t"
    reasoning_effort = None
    base = model_name

    while True:
        matched = False
        if base.endswith("-thinking"):
            base = base[:-9]
            reasoning_effort = "high"
            matched = True
            continue

        for suff, mode in SPECIAL_SUFFIX_MAP.items():
            if base.endswith(suff):
                base = base[:-len(suff)]
                chat_type = mode
                matched = True
                break

        if not matched:
            break

    base = ANTHROPIC_TO_QWEN.get(base, base)
    return base, chat_type, reasoning_effort


def _format_search_info(citations: list[dict], mode: str = "table") -> str:
    """Format web search citations as Markdown table or text list."""
    if not citations:
        return ""
    if mode == "table":
        lines = ["\n\n### 🔍 Fuentes Consultadas\n| # | Fuentes / References | URL |\n|---|---|---|"]
        for i, c in enumerate(citations, 1):
            title = c.get("title") or c.get("name") or "Fuente"
            url = c.get("url") or "#"
            lines.append(f"| {i} | [{title}]({url}) | {url} |")
        return "\n".join(lines) + "\n"
    elif mode == "text":
        lines = ["\n\n### 🔍 Fuentes Consultadas:"]
        for i, c in enumerate(citations, 1):
            title = c.get("title") or c.get("name") or "Fuente"
            url = c.get("url") or "#"
            lines.append(f"{i}. [{title}]({url})")
        return "\n".join(lines) + "\n"
    return ""



def _map_model(model: str) -> str:
    base, _, _ = _parse_model_suffix(model)
    return base


def _map_size_to_aspect_ratio(size: str | None) -> str:
    if not size:
        return "1:1"
    s = size.lower().strip()
    if s in ("1:1", "4:3", "3:4", "16:9", "9:16"):
        return s
    if "1024x1792" in s or "9:16" in s:
        return "9:16"
    if "1792x1024" in s or "1280x720" in s or "1920" in s or "16:9" in s:
        return "16:9"
    if "1024x768" in s or "4:3" in s:
        return "4:3"
    if "768x1024" in s or "3:4" in s:
        return "3:4"
    return "1:1"



def _now() -> int:
    return int(time.time())


def _openai_error(status_code: int, message: str, err_type: str = "invalid_request_error", code: Optional[str] = None) -> JSONResponse:
    body = {"error": {"message": message, "type": err_type}}
    if code is not None:
        body["error"]["code"] = code
    return JSONResponse(status_code=status_code, content=body)


class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[dict]
    stream: bool = True
    chat_type: Literal["t2t", "search", "artifacts", "web_dev", "deep_research", "t2i", "image_edit", "t2v", "travel", "learn", "slides"] = "t2t"
    reasoning_effort: str | None = "medium"
    thinking_mode: Literal["Auto", "Thinking", "Fast"] | None = None
    tools: list[dict] | None = None
    token: str | None = None
    conversation_id: str | None = None
    parent_id: str | None = None
    files: list[dict] | None = None
    aspect_ratio: Literal["1:1", "4:3", "3:4", "16:9", "9:16"] | None = None


class EmbeddingsRequest(BaseModel):
    model: str = DEFAULT_MODEL
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    token: str | None = None


class CompletionsRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str | list[str]
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    logit_bias: dict | None = None
    user: str | None = None
    token: str | None = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[dict]
    owned_by: str = "qwen-reverse"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Qwen API starting on {settings.host}:{settings.port}")
    if settings.qwen_token:
        logger.info("Using provided QWEN_TOKEN")
    yield
    logger.info("Qwen API stopped")


app = FastAPI(
    title="Qwen API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    try:
        models = await fetch_models(settings.qwen_token, browser=settings.browser)
        all_models = []
        for m in models:
            all_models.append(m)
            all_models.append(f"{m}-search")
            all_models.append(f"{m}-thinking")
            all_models.append(f"{m}-search-thinking")
            all_models.append(f"{m}-web-dev")
            all_models.append(f"{m}-deep-research")
            all_models.append(f"{m}-artifacts")
            all_models.append(f"{m}-slides")
        data = [{"id": m, "object": "model", "owned_by": "qwen-reverse"} for m in all_models]
        return ModelsResponse(data=data, owned_by="qwen-reverse")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completion(body: ChatRequest):
    token = body.token or settings.qwen_token
    base_model, suffix_chat_type, suffix_reasoning = _parse_model_suffix(body.model)
    effective_model = base_model
    effective_chat_type = body.chat_type if body.chat_type != "t2t" else suffix_chat_type
    effective_reasoning = suffix_reasoning or body.reasoning_effort
    try:
        if body.stream:
            return StreamingResponse(
                _stream_chat(body, token),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        content_buffer = []
        reasoning_buffer = []
        tool_calls = None
        usage = {}
        async for event in create_chat(
            model=effective_model,
            messages=body.messages,
            token=token,
            chat_type=effective_chat_type,
            tools=body.tools,
            stream=True,
            reasoning_effort=effective_reasoning,
            thinking_mode=body.thinking_mode,
            conversation_id=body.conversation_id,
            parent_id=body.parent_id,
            emit_tool_calls=True,
            files=body.files,
            aspect_ratio=body.aspect_ratio,
            browser=settings.browser,
        ):
            if event["type"] == "content":
                content_buffer.append(event["data"])
            elif event["type"] == "reasoning":
                reasoning_buffer.append(event["data"])
            elif event["type"] == "tool_calls":
                tool_calls = event["data"]
            elif event["type"] == "usage":
                usage = event["data"]
            elif event["type"] == "done":
                break
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": _now(),
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content_buffer) or None,
                    "reasoning_content": "".join(reasoning_buffer) or None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": _to_openai_usage(usage),
        }
    except QwenAuthError as e:
        raise _openai_error(401, str(e), "authentication_error")
    except QwenRateLimitError as e:
        raise _openai_error(429, str(e), "rate_limit_error")
    except QwenError as e:
        raise _openai_error(502, str(e), "server_error")


async def _stream_chat(body: ChatRequest, token: str | None):
    base_model, suffix_chat_type, suffix_reasoning = _parse_model_suffix(body.model)
    effective_model = base_model
    effective_chat_type = body.chat_type if body.chat_type != "t2t" else suffix_chat_type
    effective_reasoning = suffix_reasoning or body.reasoning_effort

    full_response_id = f"chatcmpl-{uuid.uuid4()}"
    created = _now()

    # Emit initial OpenAI role chunk required by AI SDK / OpenCode CLI
    role_chunk = {
        "id": full_response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": body.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    async for event in create_chat(
        model=effective_model,
        messages=body.messages,
        token=token,
        chat_type=effective_chat_type,
        tools=body.tools,
        stream=True,
        reasoning_effort=effective_reasoning,
        thinking_mode=body.thinking_mode,
        conversation_id=body.conversation_id,
        parent_id=body.parent_id,
        emit_tool_calls=True,
        files=body.files,
        aspect_ratio=body.aspect_ratio,
        browser=settings.browser,
    ):
        if event["type"] == "content":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": created, "model": body.model, "choices": [{"index": 0, "delta": {"content": event["data"]}, "finish_reason": None}]})}\n\n'
        elif event["type"] == "tool_calls":
            for tc in event["data"]:
                yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": created, "model": body.model, "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}]})}\n\n'
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": created, "model": body.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})}\n\n'
            yield "data: [DONE]\n\n"
            return
        elif event["type"] == "usage":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": created, "model": body.model, "choices": [], "usage": _to_openai_usage(event["data"])})}\n\n'
        elif event["type"] == "reasoning":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": created, "model": body.model, "choices": [{"index": 0, "delta": {"reasoning_content": event["data"]}, "finish_reason": None}]})}\n\n'
        elif event["type"] == "done":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": created, "model": body.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
            yield "data: [DONE]\n\n"
            return


@app.post("/v1/embeddings")
async def embeddings(body: EmbeddingsRequest):
    try:
        # qwen-reverse does not have a dedicated embeddings endpoint;
        # return a placeholder embedding so clients that call /v1/embeddings
        # don't get a 404.  Real embedding support would require a backend
        # change in the qwen API.
        input_texts = [body.input] if isinstance(body.input, str) else body.input
        dummy_vec = [0.0] * (body.dimensions or 1536)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": dummy_vec,
                }
                for i in range(len(input_texts))
            ],
            "model": body.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except QwenError as e:
        raise _openai_error(502, str(e), "server_error")


@app.post("/v1/completions")
async def completions(body: CompletionsRequest):
    """Legacy OpenAI completions endpoint — wraps chat completions."""
    token = body.token or settings.qwen_token
    try:
        base_model, _, _ = _parse_model_suffix(body.model)
        content_buffer = []
        reasoning_buffer = []
        usage = {}
        prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
        async for event in create_chat(
            model=base_model,
            messages=[{"role": "user", "content": prompt}],
            token=token,
            chat_type="t2t",
            stream=True,
            reasoning_effort="none",
            emit_tool_calls=False,
        ):
            if event["type"] == "content":
                content_buffer.append(event["data"])
            elif event["type"] == "reasoning":
                reasoning_buffer.append(event["data"])
            elif event["type"] == "usage":
                usage = event["data"]
            elif event["type"] == "done":
                break
        choice_text = "".join(content_buffer) or ""
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": _now(),
            "model": body.model,
            "choices": [{
                "text": choice_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": _to_openai_usage(usage),
        }
    except QwenAuthError as e:
        raise _openai_error(401, str(e), "authentication_error")
    except QwenRateLimitError as e:
        raise _openai_error(429, str(e), "rate_limit_error")
    except QwenError as e:
        raise _openai_error(502, str(e), "server_error")


def _to_openai_usage(u: dict) -> dict:
    if not u:
        return {}
    return {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


# ── Anthropic Messages API compatibility ──────────────────────────────

class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[dict]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict


class AnthropicRequest(BaseModel):
    model: str = "claude-opus-4-6"
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    stream: bool = False
    system: str | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: Literal["auto", "any", "none"] | dict | None = None
    thinking: dict | None = None
    metadata: dict | None = None


def _anthropic_to_openai(body: AnthropicRequest) -> dict:
    """Translate an Anthropic Messages request into OpenAI Chat Completions params."""
    model, chat_type, reasoning_effort = _parse_model_suffix(body.model)
    messages: list[dict] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    for msg in body.messages:
        content = msg.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    text_parts.append(f"[image: {block.get('source', {}).get('media_type', 'unknown')}]")
            content = "\n".join(text_parts)
        messages.append({"role": msg.role, "content": content})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "token": settings.qwen_token,
        "chat_type": chat_type,
        "reasoning_effort": reasoning_effort or "medium",
        "stream": body.stream,
        "emit_tool_calls": True,
        "browser": settings.browser,
    }
    if body.max_tokens:
        kwargs["max_tokens"] = body.max_tokens
    if body.tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in body.tools
        ]
    return kwargs


async def _anthropic_stream(body: AnthropicRequest):
    """Translate OpenAI SSE stream into Anthropic Messages SSE format."""
    openai_kwargs = _anthropic_to_openai(body)
    response_id = f"msg_{uuid.uuid4()}"
    next_index = 0
    reasoning_block_index: int | None = None
    content_block_index: int | None = None
    has_tool = False
    total_input_tokens = 0
    total_output_tokens = 0

    # Emit message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': response_id, 'model': body.model, 'role': 'assistant', 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    async for event in create_chat(**openai_kwargs):
        if event["type"] == "done":
            break
        if event["type"] == "content":
            if content_block_index is None:
                content_block_index = next_index
                next_index += 1
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': event['data']}})}\n\n"
        elif event["type"] == "reasoning":
            if reasoning_block_index is None:
                reasoning_block_index = next_index
                next_index += 1
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': reasoning_block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': reasoning_block_index, 'delta': {'type': 'thinking_delta', 'thinking': event['data']}})}\n\n"
        elif event["type"] == "tool_calls":
            for tc in event["data"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or tc.get("name", "")
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        input_obj = json.loads(raw_args)
                    except json.JSONDecodeError:
                        input_obj = {}
                elif isinstance(raw_args, dict):
                    input_obj = raw_args
                else:
                    input_obj = tc.get("input") or {}
                tc_index = next_index
                next_index += 1
                has_tool = True
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': tc_index, 'content_block': {'type': 'tool_use', 'id': tc.get('id') or uuid.uuid4().hex, 'name': name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tc_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(input_obj)}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tc_index})}\n\n"
        elif event["type"] == "usage":
            usage = event["data"]
            total_input_tokens = usage.get("input_tokens", 0)
            total_output_tokens = usage.get("output_tokens", 0)

    if content_block_index is not None:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
    if reasoning_block_index is not None:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': reasoning_block_index})}\n\n"

    stop_reason = "tool_use" if has_tool else "end_turn"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'input_tokens': total_input_tokens, 'output_tokens': total_output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/messages")
async def anthropic_messages(body: AnthropicRequest, request: Request):
    """Anthropic Messages API — translated to qwen-reverse OpenAI format."""
    try:
        if body.stream:
            return StreamingResponse(
                _anthropic_stream(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        # Non-streaming: collect events and build Anthropic response
        content_blocks = []
        reasoning_text = ""
        tool_results = []
        usage = {}
        openai_kwargs = _anthropic_to_openai(body)
        openai_kwargs["stream"] = False
        async for event in create_chat(**openai_kwargs):
            if event["type"] == "content":
                content_blocks.append({"type": "text", "text": event["data"]})
            elif event["type"] == "reasoning":
                reasoning_text += event["data"]
            elif event["type"] == "tool_calls":
                tool_results.extend(event["data"])
            elif event["type"] == "usage":
                usage = event["data"]
            elif event["type"] == "done":
                break

        if reasoning_text:
            content_blocks.insert(0, {"type": "thinking", "thinking": reasoning_text})

        stop_reason = "tool_calls" if tool_results else "end_turn"
        return {
            "id": f"msg_{uuid.uuid4()}",
            "type": "message",
            "role": "assistant",
            "model": body.model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    except QwenAuthError as e:
        raise _openai_error(401, str(e), "authentication_error")
    except QwenRateLimitError as e:
        raise _openai_error(429, str(e), "rate_limit_error")
    except QwenError as e:
        raise _openai_error(502, str(e), "server_error")


# ── OpenAI Native Image Endpoints ────────────────────────────────────

class ImageGenerationsRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"
    user: str | None = None
    token: str | None = None


@app.post("/v1/images/generations")
async def generate_images(body: ImageGenerationsRequest):
    token = body.token or settings.qwen_token
    base_model, _, _ = _parse_model_suffix(body.model)
    aspect_ratio = _map_size_to_aspect_ratio(body.size)
    try:
        img = Image(model=base_model, token=token)
        urls = await img.generate(body.prompt, aspect_ratio=aspect_ratio)
        data = [{"url": u} for u in urls]
        return {
            "created": _now(),
            "data": data,
        }
    except QwenAuthError as e:
        raise _openai_error(401, str(e), "authentication_error")
    except QwenRateLimitError as e:
        raise _openai_error(429, str(e), "rate_limit_error")
    except QwenError as e:
        raise _openai_error(502, str(e), "server_error")


@app.post("/v1/images/edits")
async def edit_images(
    request: Request,
    model: str = Form(DEFAULT_MODEL),
    prompt: str = Form(""),
    size: str = Form("1024x1024"),
    image_file: UploadFile | None = File(None, alias="image"),
    token: str | None = Form(None),
):
    token_val = token or settings.qwen_token
    base_model, _, _ = _parse_model_suffix(model)
    aspect_ratio = _map_size_to_aspect_ratio(size)

    image_src = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            json_body = await request.json()
            model = json_body.get("model", model)
            prompt = json_body.get("prompt", prompt)
            size = json_body.get("size", size)
            token_val = json_body.get("token") or token_val
            image_src = json_body.get("image")
            base_model, _, _ = _parse_model_suffix(model)
            aspect_ratio = _map_size_to_aspect_ratio(size)
        except Exception:
            pass

    if not image_src and image_file:
        content = await image_file.read()
        image_src = content

    if not image_src or not prompt:
        raise HTTPException(status_code=400, detail="Both image and prompt are required for image editing.")

    try:
        img_edit = ImageEdit(model=base_model, token=token_val)
        urls = await img_edit.edit(prompt, image_src, aspect_ratio=aspect_ratio)
        data = [{"url": u} for u in urls]
        return {
            "created": _now(),
            "data": data,
        }
    except QwenAuthError as e:
        raise _openai_error(401, str(e), "authentication_error")
    except QwenRateLimitError as e:
        raise _openai_error(429, str(e), "rate_limit_error")
    except QwenError as e:
        raise _openai_error(502, str(e), "server_error")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
