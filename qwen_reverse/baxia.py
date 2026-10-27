"""Aliyun / baxia "Access Verification" puzzle-captcha solver.

The WAF mounts a puzzle captcha (``#waf_nc_block`` -> ``#aliyunCaptcha-*``)
when a request is risk-flagged (``rgv587_flag`` / ``FAIL_SYS_USER_VALIDATE``).
It is *not* a plain slider: the user must drag the puzzle piece into the
matching hole.

Strategy (matches the V5 spec):
  1. Measure the hole and the piece from the two base64 images inside the
     page (canvas pixel analysis, no extra Python deps).
  2. Build a strictly monotonic, clamped trajectory: brisk acceleration to
     88% of the distance, then a micro-stepped deceleration to 100%. Never
     overshoot the slider track.
  3. Dispatch trusted CDP mouse events (``Input.dispatchMouseEvent``).
  4. Verify the block disappeared; refresh and retry with jitter otherwise.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Optional

from .cdp import CDPPage, CDPError


class QwenCaptchaError(RuntimeError):
    """Raised when the baxia captcha cannot be solved."""


# Reads both captcha images and returns geometry in CSS pixels.
_MEASURE_JS = r"""
(async () => {
  const bg = document.getElementById('aliyunCaptcha-img');
  const pc = document.getElementById('aliyunCaptcha-puzzle');
  const sl = document.getElementById('aliyunCaptcha-sliding-slider');
  if (!bg || !pc || !sl) return null;

  const load = async (src) => {
    const r = await fetch(src, { credentials: 'omit' });
    const b = await r.blob();
    return await createImageBitmap(b);
  };
  const [bi, pi] = await Promise.all([load(bg.src), load(pc.src)]);

  const pixels = (img) => {
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, c.width, c.height).data;
  };

  // 1) Piece silhouette from the alpha channel.
  const pd = pixels(pi);
  let pieceMin = 1e9, pieceMax = -1, pieceTop = 1e9, pieceBottom = -1;
  for (let y = 0; y < pi.height; y++) {
    for (let x = 0; x < pi.width; x++) {
      const a = pd[(y * pi.width + x) * 4 + 3];
      if (a > 40) {
        if (x < pieceMin) pieceMin = x;
        if (x > pieceMax) pieceMax = x;
        if (y < pieceTop) pieceTop = y;
        if (y > pieceBottom) pieceBottom = y;
      }
    }
  }

  // 2) Hole: a translucent white puzzle shape. Slide the piece silhouette
  //    across the background and pick the offset maximizing min(r,g,b)
  //    under the opaque pixels (the hole is the brightest neutral region).
  const bd = pixels(bi);
  const pw = pi.width, ph = pi.height;
  const scores = [];
  let bestScore = -1;
  for (let off = 0; off <= bi.width - pw; off++) {
    let score = 0;
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        if (pd[(y * pw + x) * 4 + 3] > 40) {
          const o = (y * bi.width + (off + x)) * 4;
          score += Math.min(bd[o], bd[o+1], bd[o+2]);
        }
      }
    }
    scores.push(score);
    if (score > bestScore) bestScore = score;
  }

  // The whiteness plateau is flat; use its midpoint, not its first sample.
  const tolerance = Math.max(1, bestScore * 0.0025);
  let lo = -1, hi = -1;
  for (let off = 0; off < scores.length; off++) {
    if (scores[off] >= bestScore - tolerance) {
      if (lo < 0) lo = off;
      hi = off;
    }
  }
  const bestOffset = lo >= 0 ? Math.round((lo + hi) / 2) : 0;

  const br = bg.getBoundingClientRect();
  const sr = sl.getBoundingClientRect();
  const currentLeft = parseFloat(pc.style.left || '0') || 0;
  return {
    naturalWidth: bi.width,
    displayWidth: br.width,
    holeMin: bestOffset, holeMax: bestOffset + pw,
    holeCount: pw,
    pieceMin, pieceMax, pieceTop, pieceBottom,
    bestOffset, bestScore,
    currentLeft,
    slider: { x: sr.x, y: sr.y, w: sr.width, h: sr.height },
    image: { x: br.x, y: br.y, w: br.width, h: br.height },
  };
})()
"""

_VISIBLE_JS = r"""
(() => {
  const img = document.getElementById('aliyunCaptcha-img');
  if (img && img.offsetParent !== null) return 'puzzle';
  const frame = document.getElementById('baxia-dialog-content');
  if (frame && frame.offsetParent !== null) return 'slider';
  const block = document.getElementById('waf_nc_block');
  if (block && block.offsetParent !== null &&
      getComputedStyle(block).display !== 'none' &&
      getComputedStyle(block).visibility !== 'hidden') {
    return 'block';
  }
  return false;
})()
"""

# Simple "slide to the end" captcha lives in a same-origin punish iframe.
_MEASURE_SLIDER_JS = r"""
(() => {
  const frame = document.getElementById('baxia-dialog-content');
  if (!frame || frame.offsetParent === null) return null;
  let doc = null;
  try { doc = frame.contentDocument; } catch (e) { return null; }
  if (!doc) return null;
  const handle = doc.getElementById('nc_1_n1z');
  const track = doc.getElementById('nc_1_n1t') || doc.getElementById('nc_1_wrapper');
  if (!handle || !track) return null;
  const fr = frame.getBoundingClientRect();
  const hr = handle.getBoundingClientRect();
  const tr = track.getBoundingClientRect();
  return {
    handle: { x: fr.x + hr.x, y: fr.y + hr.y, w: hr.width, h: hr.height },
    track: { x: fr.x + tr.x, y: fr.y + tr.y, w: tr.width, h: tr.height },
  };
})()
"""


async def captcha_kind(page: CDPPage) -> Optional[str]:
    try:
        value = await page.evaluate(_VISIBLE_JS, await_promise=False)
    except CDPError:
        return None
    return value or None


async def captcha_visible(page: CDPPage) -> bool:
    return bool(await captcha_kind(page))


async def wait_for_captcha(page: CDPPage, timeout: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await captcha_visible(page):
            return True
        await asyncio.sleep(0.1)
    return False


def build_trajectory(start_x: float, start_y: float, end_x: float, end_y: float, rng: random.Random):
    """Yield (x, y, delay_s) samples strictly inside [start_x, end_x]."""
    distance = max(0.0, end_x - start_x)
    if distance <= 0:
        yield end_x, end_y, 0.012
        return

    fast_steps = max(22, min(38, round(distance / 7)))
    settle_steps = max(6, round(fast_steps * 0.25))
    knee_x = start_x + distance * 0.88
    prev = start_x

    for step in range(1, fast_steps + 1):
        progress = step / fast_steps
        eased = 3.2 * progress * progress if progress < 0.45 else 1 - (1 - progress) ** 2.4
        x = start_x + (knee_x - start_x) * eased
        x = min(knee_x, max(prev, x))
        prev = x
        delay = 0.065 if abs(progress - 0.55) < (1 / fast_steps) else rng.uniform(0.008, 0.020)
        yield round(x, 1), round(start_y + rng.uniform(-1.1, 1.1), 1), delay

    for step in range(1, settle_steps + 1):
        progress = step / settle_steps
        eased = 1 - (1 - progress) ** 2.2
        x = knee_x + (end_x - knee_x) * eased
        x = min(end_x, max(prev, x))
        prev = x
        yield round(x, 1), round(end_y + rng.uniform(-0.5, 0.5), 1), rng.uniform(0.014, 0.026)

    yield round(end_x, 1), round(end_y, 1), 0.024


async def _drag(page: CDPPage, start_x: float, start_y: float, dx: float, seed: int) -> None:
    rng = random.Random(seed)
    end_x = start_x + dx
    await page.mouse("mousePressed", start_x, start_y, 1)
    for x, y, delay in build_trajectory(start_x, start_y, end_x, start_y, rng):
        await page.mouse("mouseMoved", x, y, 1)
        await asyncio.sleep(delay)
    await page.mouse("mouseMoved", end_x, start_y, 1)
    await asyncio.sleep(0.05)
    await page.mouse("mouseReleased", end_x, start_y, 0)


async def _drag_puzzle(page: CDPPage, info: dict, target_left: float, seed: int) -> None:
    """Place the puzzle piece exactly at ``target_left``.

    baxia maps the mouse delta to the piece's ``style.left`` with a
    quadratic curve (measured live: ``left = a*dx^2 + b*dx``, normalised so
    ``left(max_travel) == max_travel``).  We sample two points while holding
    the handle, solve for ``a``/``b`` including the origin, then invert the
    curve and drag once to the exact offset before releasing.
    """
    rng = random.Random(seed)
    start_x = info["slider"]["x"] + info["slider"]["w"] / 2
    start_y = info["slider"]["y"] + info["slider"]["h"] / 2
    max_travel = max(0.0, info["image"]["w"] - info["slider"]["w"])
    target_left = max(0.0, min(max_travel, target_left))

    async def read_left() -> float:
        return await page.evaluate(
            "parseFloat(document.getElementById('aliyunCaptcha-puzzle')?.style.left || '0') || 0",
            await_promise=False,
        )

    await page.mouse("mousePressed", start_x, start_y, 1)
    await asyncio.sleep(0.05)

    samples: list[tuple[float, float]] = [(0.0, 0.0)]
    for dx in (20.0, 50.0):
        for step in range(1, 5):
            await page.mouse("mouseMoved", start_x + dx * step / 4, start_y, 1)
            await asyncio.sleep(0.012)
        await asyncio.sleep(0.12)
        samples.append((dx, await read_left()))

    # Solve left = a*dx^2 + b*dx from the two non-zero samples.
    (x1, y1), (x2, y2) = samples[1], samples[2]
    a = b = 0.0
    if x1 > 0 and x2 > x1:
        a = (y1 / x1 - y2 / x2) / (x1 - x2)
        b = y1 / x1 - a * x1

    dx_target = target_left
    if a > 1e-9 and b >= 0:
        disc = b * b + 4 * a * target_left
        dx_target = (-b + math.sqrt(disc)) / (2 * a)
    elif b > 1e-9:
        dx_target = target_left / b
    dx_target = max(50.0, min(max_travel, dx_target))

    end_x = start_x + dx_target
    cur_x = start_x + 50.0 if dx_target >= 50.0 else start_x
    steps = max(16, min(34, round(abs(end_x - cur_x) / 7)))
    if end_x > cur_x:
        for step in range(1, steps + 1):
            progress = step / steps
            eased = 3.2 * progress * progress if progress < 0.45 else 1 - (1 - progress) ** 2.4
            x = cur_x + (end_x - cur_x) * eased
            await page.mouse("mouseMoved", x, start_y + rng.uniform(-0.8, 0.8), 1)
            await asyncio.sleep(rng.uniform(0.010, 0.022))

    # Final in-flight correction: release only when the piece is at target.
    for _ in range(3):
        await page.mouse("mouseMoved", end_x, start_y, 1)
        await asyncio.sleep(0.10)
        left_now = await read_left()
        err = target_left - left_now
        if abs(err) <= 0.8:
            break
        dx_now = end_x - start_x
        slope = (2 * a * dx_now + b) if a > 1e-9 else (b if b > 1e-9 else 0.8)
        if slope < 0.15:
            slope = 0.5
        step = max(-6.0, min(14.0, (err / slope) * 0.8))
        end_x = max(start_x, min(start_x + max_travel, end_x + step))
        await page.mouse("mouseMoved", end_x, start_y, 1)
        await asyncio.sleep(0.12)

    await page.mouse("mouseMoved", end_x, start_y, 1)
    await asyncio.sleep(0.06)
    await page.mouse("mouseReleased", end_x, start_y, 0)





async def _solve_slider(page: CDPPage, *, max_attempts: int = 3) -> bool:
    for attempt in range(1, max_attempts + 1):
        info = await page.evaluate(_MEASURE_SLIDER_JS, await_promise=False)
        if not info:
            return True
        dx = max(0.0, info["track"]["w"] - info["handle"]["w"])
        start_x = info["handle"]["x"] + info["handle"]["w"] / 2
        start_y = info["handle"]["y"] + info["handle"]["h"] / 2
        await _drag(page, start_x, start_y, dx, seed=9000 + attempt)
        await asyncio.sleep(2.0)
        if not await captcha_visible(page):
            return True
        await asyncio.sleep(0.8)
    return not await captcha_visible(page)


async def _solve_puzzle(page: CDPPage, *, max_attempts: int = 1, timeout: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            info = await page.evaluate(_MEASURE_JS)
        except CDPError:
            info = None
        if not info:
            return True
        if info["holeCount"] < 50 or info["pieceMax"] < 0:
            await asyncio.sleep(0.4)
            continue

        attempt += 1
        if attempt > max_attempts:
            break
        scale = info["displayWidth"] / info["naturalWidth"] if info["naturalWidth"] else 1.0
        target_left = info.get("bestOffset", info["holeMin"] - info["pieceMin"]) * scale
        await _drag_puzzle(page, info, max(0.0, target_left), seed=1337 + attempt)
        await asyncio.sleep(2.0)
        return not await captcha_visible(page)
    return not await captcha_visible(page)


async def solve_captcha(
    page: CDPPage,
    *,
    max_attempts: int = 1,
    timeout: float = 25.0,
) -> bool:
    """Solve whichever baxia captcha is mounted in ``page``.

    The WAF often chains challenges (slide-to-end -> puzzle).  Each widget is
    attempted **once**: a wrong release returns Aliyun ``F015`` and two rapid
    failures trip ``F011`` (same-device rate limit), after which even a perfect
    manual solve is rejected.  On failure we stop and surface a clear error.

    Safe to call when no captcha is present (returns True immediately).
    """
    for _round in range(2):
        kind = await captcha_kind(page)
        if kind is None:
            # Give baxia up to 1.2s to mount the modal asynchronously.
            if not await wait_for_captcha(page, 1.2):
                return True
            kind = await captcha_kind(page) or "block"

        if kind == "slider":
            solved = await _solve_slider(page, max_attempts=1)
        elif kind == "puzzle":
            solved = await _solve_puzzle(page, max_attempts=1, timeout=timeout)
        else:
            # Block without a known widget: wait for the widget to mount.
            await asyncio.sleep(0.6)
            continue

        if not solved:
            raise QwenCaptchaError(
                "captcha interaction rejected (Aliyun F015). The device is now "
                "rate-limited (F011); wait a few minutes before retrying, or "
                "solve it once manually in the persistent profile."
            )
        await asyncio.sleep(0.4)

    if not await captcha_visible(page):
        return True
    raise QwenCaptchaError("baxia captcha could not be solved")
