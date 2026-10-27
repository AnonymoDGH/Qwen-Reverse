"""Minimal Chrome DevTools Protocol client (no Playwright dependency).

Used by :mod:`qwen_reverse.browser` to execute Qwen requests inside a real
browser page, where baxiaCommon.js injects a valid ``bx-ua`` /
``bx-umidtoken`` per request and where the Alibaba captcha can be solved
with trusted CDP mouse events.

Requires a Chrome/Chromium running with ``--remote-debugging-port``.  If no
browser is reachable, :meth:`ChromeCDP.ensure_browser` launches one.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
import time
from typing import Any, Optional
from urllib.parse import quote

import aiohttp


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9222


class CDPError(RuntimeError):
    pass


def _chrome_candidates() -> list[str]:
    system = platform.system()
    out: list[str] = []
    if system == "Windows":
        for base in (
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if base:
                out.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
                out.append(os.path.join(base, "Chromium", "Application", "chrome.exe"))
                out.append(os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"))
    elif system == "Darwin":
        out += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            p = shutil.which(name)
            if p:
                out.append(p)
    return [p for p in out if p and os.path.isfile(p)]


class CDPPage:
    """One page target. Commands are sent directly on the target websocket."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, session: aiohttp.ClientSession):
        self._ws = ws
        self._session = session
        self._id = 0
        self._closed = False

    async def call(self, method: str, params: Optional[dict] = None) -> dict:
        if self._closed:
            raise CDPError("page closed")
        self._id += 1
        mid = self._id
        await self._ws.send_json({"id": mid, "method": method, "params": params or {}})
        while True:
            msg = await self._ws.receive()
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise CDPError(f"websocket closed ({msg.type.name})")
            data = json.loads(msg.data)
            if data.get("id") != mid:
                continue
            if "error" in data:
                raise CDPError(f"{method}: {data['error']}")
            return data.get("result", {})

    async def evaluate(self, expression: str, await_promise: bool = True, by_value: bool = True) -> Any:
        result = await self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": by_value,
                "awaitPromise": await_promise,
                "userGesture": True,
            },
        )
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            text = details.get("text") or details.get("exception", {}).get("description")
            raise CDPError(f"JS error: {text}")
        return result.get("result", {}).get("value")

    async def navigate(self, url: str, timeout: float = 45.0) -> None:
        await self.call("Page.navigate", {"url": url})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = await self.evaluate("document.readyState", await_promise=False)
            except CDPError:
                state = None
            if state == "complete":
                return
            await asyncio.sleep(0.25)

    async def cookies(self) -> list[dict]:
        return (await self.call("Network.getAllCookies")).get("cookies", [])

    async def cookie_header(self) -> str:
        jar = {}
        for c in await self.cookies():
            if "qwen.ai" in c.get("domain", ""):
                jar[c["name"]] = c["value"]
        return "; ".join(f"{k}={v}" for k, v in jar.items())

    async def set_cookie(self, name: str, value: str, domain: str = ".qwen.ai", path: str = "/") -> None:
        await self.call(
            "Network.setCookie",
            {"name": name, "value": value, "domain": domain, "path": path},
        )

    async def mouse(self, event_type: str, x: float, y: float, buttons: int = 0) -> None:
        await self.call(
            "Input.dispatchMouseEvent",
            {
                "type": event_type,
                "x": x,
                "y": y,
                "button": "left",
                "buttons": buttons,
                "clickCount": 1,
            },
        )

    async def close(self) -> None:
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            pass


class ChromeCDP:
    """Discovery/launch + page creation."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        headless: bool = False,
        user_data_dir: Optional[str] = None,
        executable: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.executable = executable
        self._session: Optional[aiohttp.ClientSession] = None
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    async def version(self) -> Optional[dict]:
        s = await self._http()
        try:
            async with s.get(f"{self.base_url}/json/version") as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            return None
        return None

    async def ensure_browser(self, timeout: float = 20.0) -> None:
        if await self.version():
            return

        exe = self.executable or next(iter(_chrome_candidates()), None)
        if not exe:
            raise CDPError(
                "Chrome not found. Install Chrome/Chromium or pass executable=... "
                "and launch it with --remote-debugging-port=9222"
            )
        profile = self.user_data_dir or os.path.join(
            os.path.expanduser("~"), ".qwen", "browser-profile"
        )
        os.makedirs(profile, exist_ok=True)
        args = [
            exe,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--disable-background-networking",
        ]
        if self.headless:
            args.append("--headless=new")
        args.append("about:blank")
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.version():
                return
            await asyncio.sleep(0.3)
        raise CDPError("Chrome did not expose the DevTools endpoint in time")

    async def list_targets(self) -> list[dict]:
        s = await self._http()
        async with s.get(f"{self.base_url}/json/list") as r:
            return await r.json()

    async def new_page(self, url: str = "about:blank") -> CDPPage:
        s = await self._http()
        target_url = f"{self.base_url}/json/new?{quote(url, safe=':/?&=%')}"
        async with s.put(target_url) as r:
            if r.status != 200:
                raise CDPError(f"/json/new failed: {r.status}")
            target = await r.json()
        return await self.connect(target["webSocketDebuggerUrl"])

    async def connect(self, ws_url: str) -> CDPPage:
        s = await self._http()
        ws = await s.ws_connect(ws_url, max_msg_size=0)
        return CDPPage(ws, s)

    async def find_page(self, url_contains: str = "chat.qwen.ai") -> Optional[CDPPage]:
        for t in await self.list_targets():
            if t.get("type") == "page" and url_contains in t.get("url", ""):
                return await self.connect(t["webSocketDebuggerUrl"])
        return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
