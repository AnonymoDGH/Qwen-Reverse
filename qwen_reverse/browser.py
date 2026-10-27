"""Browser-backed transport for chat.qwen.ai.

Why this exists
---------------
Since the 2026 WAF rollout, ``/api/v2/*`` is guarded by baxiaCommon.js:
every request must carry a ``bx-ua`` signed with the page's
``performance.now()``, plus the full four-part cookie state
(``acw_tc`` / ``cna`` / ``tfstk`` / ``isg`` / ``ssxmod_itna`` ...).
A browserless aiohttp client gets either the Aliyun WAF JS challenge
(``aliyun_waf_aa``) or ``RGV587_ERROR::SM`` (puzzle captcha).

Running the request inside a real page makes baxia sign it and lets us
solve the captcha with trusted CDP mouse events.  This module keeps the
same event surface as the HTTP transport.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional

from .baxia import QwenCaptchaError, captcha_kind, solve_captcha
from .cdp import CDPPage, ChromeCDP, CDPError

FE_VERSION = "0.3.26"
BX_V = "2.5.37"
QWEN_URL = "https://chat.qwen.ai"

#: Markers that mean "risk control challenged this request".
_CHALLENGE_MARKERS = (
    "FAIL_SYS_USER_VALIDATE",
    "RGV587",
    "rgv587",
    "_____tmd_____",
    "aliyun_waf_aa",
)


def is_challenge(text: str) -> bool:
    head = text[:800]
    return any(m in head for m in _CHALLENGE_MARKERS)


class BrowserTransport:
    def __init__(
        self,
        *,
        cdp: Optional[ChromeCDP] = None,
        headless: bool = False,
        proxy: Optional[str] = None,
    ):
        self.cdp = cdp or ChromeCDP(headless=headless)
        self.headless = headless
        self.proxy = proxy
        self.page: Optional[CDPPage] = None
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._started:
            return
        await self.cdp.ensure_browser()
        page = await self.cdp.find_page("chat.qwen.ai")
        if page is None:
            page = await self.cdp.new_page("about:blank")
            await page.set_cookie("qwen_age_verification", "passed")
            await page.navigate(f"{QWEN_URL}/")
            await self._wait_ready()
            await self._warmup()
        self.page = page
        self._started = True

    async def _warmup(self) -> None:
        """Small human-ish interaction burst before the first API call."""
        import random
        page = self.page
        if page is None:
            return
        try:
            size = await page.evaluate(
                "({w: innerWidth, h: innerHeight})", await_promise=False
            ) or {"w": 1280, "h": 800}
        except CDPError:
            return
        rng = random.Random()
        x, y = size["w"] * 0.5, size["h"] * 0.5
        for _ in range(10):
            x = max(10, min(size["w"] - 10, x + rng.uniform(-120, 120)))
            y = max(10, min(size["h"] - 10, y + rng.uniform(-90, 90)))
            await page.mouse("mouseMoved", x, y, 0)
            await asyncio.sleep(rng.uniform(0.02, 0.06))
        try:
            await page.evaluate("window.scrollBy(0, 260); 1", await_promise=False)
            await asyncio.sleep(0.35)
            await page.evaluate("window.scrollTo(0, 0); 1", await_promise=False)
        except CDPError:
            pass
        await asyncio.sleep(0.8)

    async def _wait_ready(self, timeout: float = 20.0) -> None:
        assert self.page
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = await self.page.evaluate(
                    "document.readyState === 'complete' && typeof window.fetch === 'function'",
                    await_promise=False,
                )
            except CDPError:
                state = False
            if state:
                return
            await asyncio.sleep(0.25)

    async def close(self) -> None:
        if self.page:
            try:
                await self.page.close()
            except Exception:
                pass
            self.page = None
        try:
            await self.cdp.close()
        except Exception:
            pass
        self._started = False

    async def _ensure(self) -> CDPPage:
        if not self._started or self.page is None:
            await self.start()
        assert self.page is not None
        return self.page

    # ── in-page primitives ───────────────────────────────────────────
    async def _start_fetch(self, path: str, payload: Any, method: str = "POST") -> int:
        """Fire an in-page fetch and return its request id (non-blocking)."""
        page = await self._ensure()
        url = f"{QWEN_URL}{path}"
        rid = await page.evaluate(
            "window.__qwenFetchReq=(window.__qwenFetchReq||0)+1;"
            "window.__qwenFetch=window.__qwenFetch||{};"
            "window.__qwenFetchReq",
            await_promise=False,
        )
        body = json.dumps(payload)
        body_js = "null" if method.upper() == "GET" else json.dumps(body)
        script = f"""
        (() => {{
          const rid = {rid};
          const rec = window.__qwenFetch[rid] = {{done: false, status: 0, ct: '', text: '', err: null}};
          fetch({json.dumps(url)}, {{
            method: {json.dumps(method)},
            credentials: 'include',
            headers: {{'content-type': 'application/json', 'source': 'web', 'version': {json.dumps(FE_VERSION)}}},
            body: {body_js}
          }}).then(async (r) => {{
            rec.status = r.status;
            rec.ct = r.headers.get('content-type') || '';
            rec.text = await r.text();
            rec.done = true;
          }}).catch((e) => {{
            rec.err = String(e);
            rec.done = true;
          }});
          return true;
        }})()
        """
        await page.evaluate(script, await_promise=False)
        return rid

    async def _poll_fetch(self, rid: int, timeout: float = 60.0) -> dict:
        page = await self._ensure()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        solved = 0
        while loop.time() < deadline:
            state = await page.evaluate(
                f"window.__qwenFetch[{rid}] || {{done:false,err:null}}", await_promise=False
            )
            if state and state.get("done"):
                return state
            if await captcha_kind(page):
                await solve_captcha(page)
                solved += 1
                if solved > 3:
                    raise QwenCaptchaError(
                        "WAF keeps re-challenging this browser profile. "
                        "Open the persistent profile once, log in (or solve the captcha "
                        "manually), then retry."
                    )
                deadline = max(deadline, loop.time() + 15)
            await asyncio.sleep(0.15)
        raise TimeoutError("in-page fetch timed out")

    async def _fetch_text(self, path: str, payload: Any, method: str = "POST") -> dict:
        rid = await self._start_fetch(path, payload, method=method)
        try:
            res = await self._poll_fetch(rid)
        except TimeoutError:
            return {"status": 0, "ct": "", "text": "", "err": "timeout"}
        if res.get("err"):
            raise CDPError(f"in-page fetch failed: {res['err']}")
        return res

    async def fetch_json(self, path: str, payload: Any, method: str = "POST") -> dict:
        """POST JSON from inside the page; solves a baxia challenge once."""
        for attempt in range(2):
            res = await self._fetch_text(path, payload, method=method)
            text = res.get("text", "")
            if res.get("status") == 200 and not is_challenge(text):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"success": False, "raw": text}
            if is_challenge(text):
                await solve_captcha(await self._ensure())
                continue
            return {"success": False, "status": res.get("status"), "raw": text[:1000]}
        raise QwenCaptchaError("challenge persisted after solving captcha")

    async def create_chat(self, model: str, chat_type: str, chat_mode: str) -> str:
        now = int(time.time() * 1000)
        payload = {
            "chatId": "",
            "models": [model],
            "project_id": "",
            "timestamp": now,
            "chat_type": chat_type,
            "chat_mode": chat_mode,
        }
        data = await self.fetch_json("/api/v2/chats/new", payload)
        if not data.get("success") or not (data.get("data") or {}).get("id"):
            raise CDPError(f"create chat failed: {str(data)[:300]}")
        return data["data"]["id"]

    async def _start_stream(self, chat_id: str, payload: dict) -> int:
        page = await self._ensure()
        url = f"{QWEN_URL}/api/v2/chat/completions?chat_id={chat_id}"
        rid = await page.evaluate(
            "window.__qwenReq=(window.__qwenReq||0)+1;"
            "window.__qwenBufs=window.__qwenBufs||{};"
            "window.__qwenDone=window.__qwenDone||{};"
            "window.__qwenErr=window.__qwenErr||{};"
            "window.__qwenMeta=window.__qwenMeta||{};"
            "window.__qwenAbort=window.__qwenAbort||{};"
            "window.__qwenReq"
        )
        script = f"""
        (() => {{
          const rid = {rid};
          window.__qwenBufs[rid] = [];
          window.__qwenMeta[rid] = {{}};
          fetch({json.dumps(url)}, {{
            method: 'POST',
            credentials: 'include',
            headers: {{'content-type': 'application/json', 'source': 'web', 'version': {json.dumps(FE_VERSION)}}},
            body: {json.dumps(json.dumps(payload))}
          }}).then(async (r) => {{
            const m = window.__qwenMeta[rid];
            m.status = r.status;
            m.ct = r.headers.get('content-type') || '';
            const reader = r.body.getReader();
            const dec = new TextDecoder();
            while (true) {{
              if (window.__qwenAbort[rid]) {{ try {{ await reader.cancel(); }} catch (e) {{}} break; }}
              const chunk = await reader.read();
              if (chunk.done) break;
              window.__qwenBufs[rid].push(dec.decode(chunk.value, {{stream: true}}));
            }}
            window.__qwenDone[rid] = true;
          }}).catch((e) => {{
            window.__qwenErr[rid] = String(e);
            window.__qwenDone[rid] = true;
          }});
          return true;
        }})()
        """
        await page.evaluate(script, await_promise=False)
        return rid

    async def _drain(self, rid: int) -> str:
        page = await self._ensure()
        return await page.evaluate(
            f"window.__qwenBufs[{rid}].splice(0).join('')", await_promise=False
        ) or ""

    async def _done(self, rid: int) -> bool:
        page = await self._ensure()
        return bool(
            await page.evaluate(f"window.__qwenDone[{rid}]===true", await_promise=False)
        )

    async def _error(self, rid: int) -> Optional[str]:
        page = await self._ensure()
        return await page.evaluate(
            f"window.__qwenErr[{rid}]||null", await_promise=False
        )

    async def stream_completion(self, chat_id: str, payload: dict) -> AsyncIterator[str]:
        """Yield raw SSE text chunks, solving the captcha if challenged."""
        page = await self._ensure()
        for attempt in range(2):
            rid = await self._start_stream(chat_id, payload)
            prefix = ""
            challenged = False
            solved_here = False
            idle = 0
            while True:
                batch = await self._drain(rid)
                if batch:
                    idle = 0
                    if len(prefix) < 800:
                        prefix += batch
                        if is_challenge(prefix):
                            challenged = True
                            await page.evaluate(
                                f"window.__qwenAbort[{rid}]=true", await_promise=False
                            )
                            break
                    yield batch
                err = await self._error(rid)
                if err:
                    raise CDPError(f"in-page fetch failed: {err}")
                if await self._done(rid):
                    tail = await self._drain(rid)
                    if tail:
                        yield tail
                    break

                if not batch:
                    idle += 1
                    # Poll the challenge twice a second while the fetch is parked.
                    if idle % 5 == 0 and await captcha_kind(page):
                        await solve_captcha(page)
                        solved_here = True
                        idle = 0
                    # 15s after a solve / 60s cold: the parked fetch will not
                    # resume; abort and retry once.
                    if idle > (150 if solved_here else 600):
                        await page.evaluate(
                            f"window.__qwenAbort[{rid}]=true", await_promise=False
                        )
                        challenged = True
                        break
                await asyncio.sleep(0.1)
            if not challenged:
                return
            if not solved_here:
                await solve_captcha(page)
        raise QwenCaptchaError("completion challenged after captcha solve")
