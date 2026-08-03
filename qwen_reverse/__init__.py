import re
from typing import AsyncIterator, Optional

from .client import (
    QWEN_URL,
    DEFAULT_HEADERS,
    QwenError,
    QwenAuthError,
    QwenRateLimitError,
    create_chat,
    fetch_models,
    upload_file,
    build_tool_system_prompt,
    inject_tools,
    parse_tool_response,
    execute_tool_call,
    get_oss_headers,
)
from .auth import (
    QwenOAuthError,
    QwenOAuth2Client,
    SharedTokenManager,
    start_device_login,
    complete_device_login,
)
from .models import (
    DEFAULT_MODEL,
    MODELS_REQUIRE_THINKING,
    CHAT_TYPES,
    ASPECT_RATIOS,
    text_models,
    image_models,
    vision_models,
    models,
)
from .cookies import generate_cookies
from .fingerprint import generate_fingerprint, parse_fingerprint
from .bxua import BXUAGenerator, generate_bx_ua

__version__ = "0.1.2"

_URL_RE = re.compile(r"https?://[^\s)\]]+")


class Generative:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: Optional[str] = None,
        chat_type: str = "t2t",
        proxy: Optional[str] = None,
        timeout: int = 300,
    ):
        self.model = model
        self.token = token
        self.chat_type = chat_type
        self.proxy = proxy
        self.timeout = timeout

    async def generate(self, prompt: str, **kwargs) -> str:
        parts = []
        model = kwargs.pop("model", self.model)
        token = kwargs.pop("token", self.token)
        proxy = kwargs.pop("proxy", self.proxy)
        timeout = kwargs.pop("timeout", self.timeout)
        chat_type = kwargs.pop("chat_type", self.chat_type)
        async for event in create_chat(
            model, [{"role": "user", "content": prompt}],
            token=token, chat_type=chat_type,
            proxy=proxy, timeout=timeout, **kwargs,
        ):
            if event["type"] == "content":
                parts.append(event["data"])
        return "".join(parts)

    def stream(self, prompt: str, **kwargs) -> AsyncIterator[dict]:
        model = kwargs.pop("model", self.model)
        token = kwargs.pop("token", self.token)
        proxy = kwargs.pop("proxy", self.proxy)
        timeout = kwargs.pop("timeout", self.timeout)
        chat_type = kwargs.pop("chat_type", self.chat_type)
        return create_chat(
            model, [{"role": "user", "content": prompt}],
            token=token, chat_type=chat_type,
            proxy=proxy, timeout=timeout, **kwargs,
        )

    async def generate_image(self, prompt: str, aspect_ratio: str = "1:1", **kwargs) -> list[str]:
        urls = []
        model = kwargs.pop("model", self.model)
        token = kwargs.pop("token", self.token)
        proxy = kwargs.pop("proxy", self.proxy)
        timeout = kwargs.pop("timeout", self.timeout)
        aspect_ratio = kwargs.pop("aspect_ratio", aspect_ratio)
        async for event in create_chat(
            model, [{"role": "user", "content": prompt}],
            token=token, chat_type="t2i",
            proxy=proxy, timeout=timeout,
            aspect_ratio=aspect_ratio, **kwargs,
        ):
            if event["type"] == "image":
                urls.append(event["data"]["url"])
            elif event["type"] == "content":
                urls.extend(_URL_RE.findall(event["data"]))
        return list(dict.fromkeys(urls))

    async def generate_video(self, prompt: str, aspect_ratio: str = "16:9", **kwargs) -> list[str]:
        urls = []
        model = kwargs.pop("model", self.model)
        token = kwargs.pop("token", self.token)
        proxy = kwargs.pop("proxy", self.proxy)
        timeout = kwargs.pop("timeout", self.timeout)
        aspect_ratio = kwargs.pop("aspect_ratio", aspect_ratio)
        async for event in create_chat(
            model, [{"role": "user", "content": prompt}],
            token=token, chat_type="t2v",
            proxy=proxy, timeout=timeout,
            aspect_ratio=aspect_ratio, **kwargs,
        ):
            if event["type"] == "content":
                urls.extend(_URL_RE.findall(event["data"]))
        return list(dict.fromkeys(urls))


class Image(Generative):
    def __init__(
        self,
        token: Optional[str] = None,
        proxy: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 600,
    ):
        super().__init__(model=model, token=token, chat_type="t2i", proxy=proxy, timeout=timeout)

    async def generate(self, prompt: str, **kwargs) -> list[str]:
        return await self.generate_image(prompt, **kwargs)


class Video(Generative):
    def __init__(
        self,
        token: Optional[str] = None,
        proxy: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 600,
    ):
        super().__init__(model=model, token=token, chat_type="t2v", proxy=proxy, timeout=timeout)

    async def generate(self, prompt: str, **kwargs) -> list[str]:
        return await self.generate_video(prompt, **kwargs)


class Conversation:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: Optional[str] = None,
        chat_type: str = "t2t",
        proxy: Optional[str] = None,
        timeout: int = 300,
        tools: Optional[list] = None,
    ):
        self.model = model
        self.token = token
        self.chat_type = chat_type
        self.proxy = proxy
        self.timeout = timeout
        self.tools = tools
        self.messages: list[dict] = []
        self.conversation_id: Optional[str] = None
        self.parent_id: Optional[str] = None

    async def send(self, prompt: str, **kwargs) -> str:
        parts = []
        async for event in self.stream(prompt, **kwargs):
            if event["type"] == "content":
                parts.append(event["data"])
        return "".join(parts)

    def stream(self, prompt: str, **kwargs) -> AsyncIterator[dict]:
        return self._stream(prompt, **kwargs)

    async def _stream(self, prompt: str, **kwargs) -> AsyncIterator[dict]:
        self.messages.append({"role": "user", "content": prompt})
        answer: list[str] = []
        model = kwargs.pop("model", self.model)
        token = kwargs.pop("token", self.token)
        chat_type = kwargs.pop("chat_type", self.chat_type)
        proxy = kwargs.pop("proxy", self.proxy)
        timeout = kwargs.pop("timeout", self.timeout)
        tools = kwargs.pop("tools", self.tools)
        async for event in create_chat(
            model, self.messages, token=token, chat_type=chat_type,
            proxy=proxy, timeout=timeout, tools=tools,
            conversation_id=self.conversation_id, parent_id=self.parent_id, **kwargs,
        ):
            yield event
            if event["type"] == "content":
                answer.append(event["data"])
            elif event["type"] == "done":
                self.conversation_id = event.get("conversation_id") or self.conversation_id
                self.parent_id = event.get("parent_id") or self.parent_id
        if answer:
            self.messages.append({"role": "assistant", "content": "".join(answer)})

    def reset(self):
        self.messages = []
        self.conversation_id = None
        self.parent_id = None


__all__ = [
    "__version__",
    "QWEN_URL", "DEFAULT_HEADERS",
    "QwenError", "QwenAuthError", "QwenRateLimitError", "QwenOAuthError",
    "create_chat", "fetch_models", "upload_file",
    "build_tool_system_prompt", "inject_tools", "parse_tool_response", "execute_tool_call",
    "get_oss_headers",
    "QwenOAuth2Client", "SharedTokenManager", "start_device_login", "complete_device_login",
    "DEFAULT_MODEL", "MODELS_REQUIRE_THINKING", "CHAT_TYPES", "ASPECT_RATIOS",
    "text_models", "image_models", "vision_models", "models",
    "generate_cookies", "generate_fingerprint", "parse_fingerprint",
    "BXUAGenerator", "generate_bx_ua",
    "Generative", "Conversation", "Image", "Video",
]
