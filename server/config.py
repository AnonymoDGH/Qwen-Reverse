import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("QWEN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("QWEN_PORT", "8000")))
    qwen_token: str | None = field(default_factory=lambda: os.getenv("QWEN_TOKEN") or None)
    debug: bool = field(default_factory=lambda: _env_bool("QWEN_DEBUG"))
    public_url: str | None = field(default_factory=lambda: os.getenv("QWEN_PUBLIC_URL") or None)


settings = Settings()
