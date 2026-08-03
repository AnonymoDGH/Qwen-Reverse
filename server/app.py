import json
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qwen_reverse import (
    DEFAULT_MODEL,
    create_chat,
    fetch_models,
    QwenError,
    QwenAuthError,
    QwenRateLimitError,
)

from config import settings

logger = logging.getLogger("qwen_api")


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


class ModelsResponse(BaseModel):
    models: list[str]
    default: str


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
        models = await fetch_models(settings.qwen_token)
        return ModelsResponse(models=models, default=models[0] if models else DEFAULT_MODEL)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completion(body: ChatRequest):
    token = body.token or settings.qwen_token
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
            model=body.model,
            messages=body.messages,
            token=token,
            chat_type=body.chat_type,
            tools=body.tools,
            stream=True,
            reasoning_effort=body.reasoning_effort,
            thinking_mode=body.thinking_mode,
            conversation_id=body.conversation_id,
            parent_id=body.parent_id,
            emit_tool_calls=True,
            files=body.files,
            aspect_ratio=body.aspect_ratio,
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
            "id": str(uuid.uuid4()),
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content_buffer) or None,
                    "reasoning": "".join(reasoning_buffer) or None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": _to_openai_usage(usage),
        }
    except QwenAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except QwenRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except QwenError as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _stream_chat(body: ChatRequest, token: str | None):
    full_response_id = str(uuid.uuid4())
    async for event in create_chat(
        model=body.model,
        messages=body.messages,
        token=token,
        chat_type=body.chat_type,
        tools=body.tools,
        stream=True,
        reasoning_effort=body.reasoning_effort,
        thinking_mode=body.thinking_mode,
        conversation_id=body.conversation_id,
        parent_id=body.parent_id,
        emit_tool_calls=True,
        files=body.files,
        aspect_ratio=body.aspect_ratio,
    ):
        if event["type"] == "content":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": 0, "model": body.model, "choices": [{"index": 0, "delta": {"content": event["data"]}, "finish_reason": None}]})}\n\n'
        elif event["type"] == "tool_calls":
            for tc in event["data"]:
                yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": 0, "model": body.model, "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}]})}\n\n'
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": 0, "model": body.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})}\n\n'
            yield "data: [DONE]\n\n"
            return
        elif event["type"] == "usage":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": 0, "model": body.model, "choices": [], "usage": _to_openai_usage(event["data"])})}\n\n'
        elif event["type"] == "reasoning":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": 0, "model": body.model, "choices": [{"index": 0, "delta": {"reasoning": event["data"]}, "finish_reason": None}]})}\n\n'
        elif event["type"] == "done":
            yield f'data: {json.dumps({"id": full_response_id, "object": "chat.completion.chunk", "created": 0, "model": body.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
            yield "data: [DONE]\n\n"
            return


def _to_openai_usage(u: dict) -> dict:
    if not u:
        return {}
    return {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }
