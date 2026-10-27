import asyncio
import base64
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import aiohttp

QWEN_OAUTH_BASE_URL = "https://chat.qwen.ai"
QWEN_OAUTH_DEVICE_CODE_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/device/code"
QWEN_OAUTH_TOKEN_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/token"

QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_SCOPE = "openid profile email model.completion"
QWEN_OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

QWEN_DIR = ".qwen"
QWEN_CREDENTIAL_FILENAME = "oauth_creds.json"
QWEN_LOCK_FILENAME = "oauth_creds.lock"

TOKEN_REFRESH_BUFFER_MS = 30 * 1000
LOCK_TIMEOUT_MS = 10 * 1000
CACHE_CHECK_INTERVAL_MS = 1000


class QwenOAuthError(Exception):
    pass


def generate_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")


def generate_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def generate_pkce_pair() -> dict:
    verifier = generate_code_verifier()
    return {"code_verifier": verifier, "code_challenge": generate_code_challenge(verifier)}


def _urlencode(data: dict) -> str:
    return "&".join(f"{k}={v}" for k, v in data.items())


class QwenOAuth2Client:
    def __init__(self):
        self.credentials: dict = {}
        self.manager = SharedTokenManager.get_instance()

    def set_credentials(self, credentials: dict):
        self.credentials = credentials

    def get_credentials(self) -> dict:
        return self.credentials

    async def get_access_token(self) -> Optional[str]:
        try:
            credentials = await self.manager.get_valid_credentials(self)
            return credentials.get("access_token")
        except Exception:
            if self.credentials.get("access_token") and self.is_token_valid(self.credentials):
                return self.credentials["access_token"]
            return None

    async def request_device_authorization(self, code_challenge: str) -> dict:
        body = _urlencode({
            "client_id": QWEN_OAUTH_CLIENT_ID,
            "scope": QWEN_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        })
        async with aiohttp.ClientSession(headers={"user-agent": ""}) as session:
            async with session.post(
                QWEN_OAUTH_DEVICE_CODE_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "x-request-id": str(uuid.uuid4()),
                },
                data=body,
            ) as resp:
                resp_json = await resp.json()
                if resp.status != 200 or "device_code" not in resp_json:
                    raise QwenOAuthError(f"Device authorization failed {resp.status}: {resp_json}")
                return resp_json

    async def poll_device_token(self, device_code: str, code_verifier: str) -> dict:
        body = _urlencode({
            "grant_type": QWEN_OAUTH_GRANT_TYPE,
            "client_id": QWEN_OAUTH_CLIENT_ID,
            "device_code": device_code,
            "code_verifier": code_verifier,
        })
        async with aiohttp.ClientSession(headers={"user-agent": ""}) as session:
            async with session.post(
                QWEN_OAUTH_TOKEN_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data=body,
            ) as resp:
                resp_json = await resp.json()
                if resp.status == 400:
                    error = resp_json.get("error")
                    if error == "authorization_pending":
                        return {"status": "pending"}
                    if error == "slow_down":
                        return {"status": "pending", "slow_down": True}
                if resp.status != 200:
                    raise QwenOAuthError(f"Token poll failed {resp.status}: {resp_json}")
                return resp_json

    async def refresh_access_token(self) -> dict:
        if not self.credentials.get("refresh_token"):
            raise QwenOAuthError("No refresh token")
        body = _urlencode({
            "grant_type": "refresh_token",
            "refresh_token": self.credentials["refresh_token"],
            "client_id": QWEN_OAUTH_CLIENT_ID,
        })
        async with aiohttp.ClientSession(headers={"user-agent": ""}) as session:
            async with session.post(
                QWEN_OAUTH_TOKEN_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data=body,
            ) as resp:
                resp_json = await resp.json()
                if resp.status == 400:
                    self.credentials = {}
                    raise QwenOAuthError("Refresh token expired or invalid")
                if resp.status != 200:
                    raise QwenOAuthError(f"Token refresh failed {resp.status}: {resp_json}")
                return resp_json

    def is_token_valid(self, credentials: dict) -> bool:
        expiry = credentials.get("expiry_date")
        if not expiry:
            return False
        return time.time() * 1000 < expiry - TOKEN_REFRESH_BUFFER_MS


class SharedTokenManager:
    _instance: Optional["SharedTokenManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.memory_cache = {"credentials": None, "file_mod_time": 0, "last_check": 0}
        self.refresh_promise = None

    @classmethod
    def get_instance(cls) -> "SharedTokenManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _credential_file(self) -> Path:
        return Path(os.path.expanduser(f"~/{QWEN_DIR}/{QWEN_CREDENTIAL_FILENAME}"))

    def _lock_file(self) -> Path:
        return Path(os.path.expanduser(f"~/{QWEN_DIR}/{QWEN_LOCK_FILENAME}"))

    async def get_valid_credentials(self, client: QwenOAuth2Client, force_refresh: bool = False) -> dict:
        self._check_reload()
        if (
            self.memory_cache["credentials"]
            and not force_refresh
            and self.is_token_valid(self.memory_cache["credentials"])
        ):
            return self.memory_cache["credentials"]
        if self.refresh_promise:
            return await self.refresh_promise
        self.refresh_promise = asyncio.create_task(self._perform_token_refresh(client, force_refresh))
        try:
            return await self.refresh_promise
        finally:
            self.refresh_promise = None

    def _check_reload(self):
        now = int(time.time() * 1000)
        if now - self.memory_cache["last_check"] < CACHE_CHECK_INTERVAL_MS:
            return
        self.memory_cache["last_check"] = now
        try:
            stat = self._credential_file().stat()
            file_mod_time = int(stat.st_mtime * 1000)
            if file_mod_time > self.memory_cache["file_mod_time"]:
                self._reload_credentials()
                self.memory_cache["file_mod_time"] = file_mod_time
        except FileNotFoundError:
            self.memory_cache["file_mod_time"] = 0

    def _reload_credentials(self):
        with open(self._credential_file(), "r") as f:
            data = json.load(f)
        for field in ("access_token", "refresh_token", "token_type", "expiry_date"):
            if field not in data:
                raise QwenOAuthError(f"Invalid credentials: missing {field}")
        self.memory_cache["credentials"] = data

    async def _perform_token_refresh(self, client: QwenOAuth2Client, force_refresh: bool) -> dict:
        lock_path = self._lock_file()
        acquired = False
        try:
            if self.memory_cache["credentials"] is None:
                self._reload_credentials()
            client.set_credentials(self.memory_cache["credentials"])
            if not client.get_credentials().get("refresh_token"):
                raise QwenOAuthError("No refresh token")
            await self._acquire_lock(lock_path)
            acquired = True

            self._check_reload()
            if (
                not force_refresh
                and self.memory_cache["credentials"]
                and self.is_token_valid(self.memory_cache["credentials"])
            ):
                client.set_credentials(self.memory_cache["credentials"])
                return self.memory_cache["credentials"]

            response = await client.refresh_access_token()
            if "access_token" not in response:
                raise QwenOAuthError("No access_token returned")
            current = client.get_credentials()
            credentials = {
                "access_token": response["access_token"],
                "token_type": response["token_type"],
                "refresh_token": response.get("refresh_token", current.get("refresh_token")),
                "resource_url": response.get("resource_url"),
                "expiry_date": int(time.time() * 1000) + response.get("expires_in", 0) * 1000,
            }
            self.memory_cache["credentials"] = credentials
            client.set_credentials(credentials)
            await self._save_credentials(credentials)
            return credentials
        finally:
            if acquired:
                await self._release_lock(lock_path)

    async def _acquire_lock(self, lock_path: Path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(50):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age_ms = int(time.time() * 1000) - int(lock_path.stat().st_mtime * 1000)
                    if age_ms > LOCK_TIMEOUT_MS:
                        lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
                await asyncio.sleep(0.2)
                continue
            except OSError:
                await asyncio.sleep(0.2)
                continue
            try:
                os.write(fd, str(uuid.uuid4()).encode())
            finally:
                os.close(fd)
            return
        raise QwenOAuthError("Failed to acquire lock")

    async def _release_lock(self, lock_path: Path):
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def _save_credentials(self, credentials: dict):
        path = self._credential_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(credentials, f, indent=2)
        self.memory_cache["file_mod_time"] = int(path.stat().st_mtime * 1000)

    def is_token_valid(self, credentials: dict) -> bool:
        expiry = credentials.get("expiry_date")
        if not expiry:
            return False
        return time.time() * 1000 < expiry - TOKEN_REFRESH_BUFFER_MS


async def start_device_login() -> tuple[QwenOAuth2Client, dict]:
    client = QwenOAuth2Client()
    pkce = generate_pkce_pair()
    data = await client.request_device_authorization(pkce["code_challenge"])
    data["code_verifier"] = pkce["code_verifier"]
    return client, data


async def complete_device_login(client: QwenOAuth2Client, data: dict) -> str:
    interval = data.get("interval", 5)
    while True:
        result = await client.poll_device_token(data["device_code"], data["code_verifier"])
        if result.get("status") == "pending":
            if result.get("slow_down"):
                interval += 5
            await asyncio.sleep(interval)
            continue
        credentials = {
            "access_token": result["access_token"],
            "token_type": result.get("token_type", "Bearer"),
            "refresh_token": result.get("refresh_token", ""),
            "resource_url": result.get("resource_url"),
            "expiry_date": int(time.time() * 1000) + result.get("expires_in", 0) * 1000,
        }
        await client.manager._save_credentials(credentials)
        return credentials["access_token"]
