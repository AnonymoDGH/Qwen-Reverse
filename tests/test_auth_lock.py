import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from qwen_reverse.auth import SharedTokenManager, QwenOAuthError, LOCK_TIMEOUT_MS


@pytest.fixture(autouse=True)
def reset_singleton():
    SharedTokenManager._instance = None
    yield
    SharedTokenManager._instance = None


@pytest.fixture
def tmp_creds(tmp_path, monkeypatch):
    creds_dir = tmp_path / ".qwen"
    creds_dir.mkdir()
    creds_file = creds_dir / "oauth_creds.json"
    lock_file = creds_dir / "oauth_creds.lock"
    monkeypatch.setattr(SharedTokenManager, "_credential_file", lambda self: creds_file)
    monkeypatch.setattr(SharedTokenManager, "_lock_file", lambda self: lock_file)
    return creds_file, lock_file


def _write_valid_creds(path: Path):
    import json
    path.write_text(json.dumps({
        "access_token": "tok",
        "refresh_token": "ref",
        "token_type": "Bearer",
        "expiry_date": int(time.time() * 1000) + 3600_000,
    }))


async def test_lock_blocks_concurrent_acquirers(tmp_creds, monkeypatch):
    _, lock_file = tmp_creds
    manager = SharedTokenManager.get_instance()

    in_critical = {"count": 0, "max": 0}

    async def hold_then_release():
        await manager._acquire_lock(lock_file)
        in_critical["count"] += 1
        in_critical["max"] = max(in_critical["max"], in_critical["count"])
        await asyncio.sleep(0.1)
        in_critical["count"] -= 1
        await manager._release_lock(lock_file)

    await asyncio.gather(*(hold_then_release() for _ in range(5)))

    assert in_critical["max"] == 1, "Lock failed to serialize concurrent acquirers"
    assert not lock_file.exists()


async def test_lock_recovers_from_stale_file(tmp_creds):
    _, lock_file = tmp_path = tmp_creds
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    old_mtime = time.time() - (LOCK_TIMEOUT_MS / 1000) - 10
    lock_file.write_text("stale")
    os.utime(lock_file, (old_mtime, old_mtime))

    manager = SharedTokenManager.get_instance()
    await manager._acquire_lock(lock_file)
    await manager._release_lock(lock_file)


async def test_lock_raises_when_unrecoverable(tmp_creds, monkeypatch):
    _, lock_file = tmp_creds
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(FileExistsError("x")))
    monkeypatch.setattr(Path, "unlink", lambda self, **k: None)
    monkeypatch.setattr(Path, "stat", lambda self: type("S", (), {"st_mtime": time.time()})())

    manager = SharedTokenManager.get_instance()
    with pytest.raises(QwenOAuthError, match="Failed to acquire lock"):
        await manager._acquire_lock(lock_file)
