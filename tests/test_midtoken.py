import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from qwen_reverse import client as client_mod


@pytest.fixture(autouse=True)
def reset_midtoken():
    client_mod._midtoken = None
    client_mod._midtoken_uses = 0
    yield
    client_mod._midtoken = None
    client_mod._midtoken_uses = 0


async def test_concurrent_attach_does_not_double_fetch():
    fetch_calls = {"n": 0}

    async def fake_fetch(proxy=None):
        fetch_calls["n"] += 1
        await asyncio.sleep(0.05)
        return "token-xyz"

    with patch.object(client_mod, "_fetch_midtoken", side_effect=fake_fetch):
        async def call():
            h = {}
            await client_mod._attach_extra_headers(h, None)
            return h

        results = await asyncio.gather(*(call() for _ in range(5)))

    assert fetch_calls["n"] == 1, "Concurrent callers triggered duplicate midtoken fetches"
    for r in results:
        assert r["bx-umidtoken"] == "token-xyz"


async def test_midtoken_rotates_after_100_uses():
    async def fake_fetch(proxy=None):
        return "fresh-token"

    with patch.object(client_mod, "_fetch_midtoken", side_effect=fake_fetch):
        async def call():
            h = {}
            await client_mod._attach_extra_headers(h, None)
            return h.get("bx-umidtoken")

        first = await call()
        for _ in range(99):
            assert await call() == first

        second = await call()
        assert second == "fresh-token"
        assert second != first or first == "fresh-token"


async def test_headers_always_get_bx_v_and_request_id():
    async def fake_fetch(proxy=None):
        return "tok"

    with patch.object(client_mod, "_fetch_midtoken", side_effect=fake_fetch):
        h = {}
        await client_mod._attach_extra_headers(h, None)
        assert h["bx-v"] == "2.5.37"
        assert h["x-request-id"]
