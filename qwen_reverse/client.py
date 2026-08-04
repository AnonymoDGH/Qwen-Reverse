import json
import os
import re
import uuid
import asyncio
import time
import hashlib
import hmac
import datetime
from typing import Any, AsyncIterator, Optional, Literal
from urllib.parse import quote

import aiohttp

from .cookies import generate_cookies
from .bxua import generate_bx_ua
from .models import MODELS_REQUIRE_THINKING, REASONING_EFFORT_LEVELS

QWEN_URL = "https://chat.qwen.ai"

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Origin': QWEN_URL,
    'Referer': f'{QWEN_URL}/',
    'Content-Type': 'application/json',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Connection': 'keep-alive',
    'X-Requested-With': 'XMLHttpRequest',
    'source': 'web',
    'version': '0.2.63',
    'X-Accel-Buffering': 'no',
}

class QwenError(Exception):
    pass

class QwenAuthError(QwenError):
    pass

class QwenRateLimitError(QwenError):
    pass

_midtoken: str | None = None
_midtoken_uses: int = 0
_midtoken_lock: asyncio.Lock = asyncio.Lock()

def _get_headers(token: Optional[str] = None):
    data = generate_cookies()
    headers = {**DEFAULT_HEADERS}
    headers['Cookie'] = f'ssxmod_itna={data["ssxmod_itna"]};ssxmod_itna2={data["ssxmod_itna2"]}'
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers

async def _fetch_midtoken(proxy: Optional[str] = None) -> str:
    async with aiohttp.ClientSession() as s:
        async with s.get('https://sg-wum.alibaba.com/w/wu.json', proxy=proxy) as r:
            r.raise_for_status()
            text = await r.text()
            match = re.search(r"(?:umx\.wu|__fycb)\('([^']+)'\)", text)
            if not match:
                raise QwenError("Failed to extract bx-umidtoken")
            return match.group(1)

async def _attach_extra_headers(headers: dict, proxy: Optional[str] = None):
    global _midtoken, _midtoken_uses
    async with _midtoken_lock:
        if _midtoken is None or _midtoken_uses >= 100:
            _midtoken = None
            for _ in range(2):
                try:
                    _midtoken = await _fetch_midtoken(proxy)
                    _midtoken_uses = 0
                    break
                except Exception:
                    await asyncio.sleep(1)
        if _midtoken is not None:
            _midtoken_uses += 1
            headers['bx-umidtoken'] = _midtoken
    headers['bx-v'] = '2.5.36'
    headers['x-request-id'] = str(uuid.uuid4())
    # Regenerated per request, matching the web client's anti-bot behaviour.
    headers['bx-ua'] = generate_bx_ua()

async def _sse_stream(response: aiohttp.ClientResponse) -> AsyncIterator[dict]:
    # Read line-by-line (not chunk-by-chunk): aiohttp's StreamReader yields
    # arbitrary byte chunks via `async for`, which can split or merge SSE
    # lines and silently drop/corrupt events. readline() guarantees one full
    # line per call, including lines split across TCP segments.
    buffer: list[str] = []
    while True:
        raw_line = await response.content.readline()
        if not raw_line:
            if buffer:
                data = "\n".join(buffer)
                buffer.clear()
                if data != "[DONE]":
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        pass
            return
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if buffer:
                data = "\n".join(buffer)
                buffer.clear()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    pass
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""

def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```\s*$", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*)$", text)
    if m:
        return m.group(1).strip()
    return text

def _parse_json_maybe(s: str):
    if not s:
        return None
    s = _strip_code_fences(s).strip()
    if s.startswith("{") and not s.endswith("}"):
        s += "}"
    try:
        return json.loads(s)
    except Exception:
        pass
    m = None
    if "{" in s and "}" in s:
        m = re.search(r"\{[\s\S]*\}", s)
    if m is None and "[" in s and "]" in s:
        m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            return None

def _stringify_tool_calls(tool_calls: list) -> str:
    parts = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if tc.get("type") == "function" else tc
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or tc.get("name") or "unknown"
        args = fn.get("arguments")
        if isinstance(args, str):
            args_str = args
        else:
            try:
                args_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=True)
            except Exception:
                args_str = "{}"
        call_id = tc.get("id", "")
        header = f"[Tool call: {name}]"
        if call_id:
            header += f" (id={call_id})"
        parts.append(f"{header}\nArguments: {args_str}")
    return "\n\n".join(parts)

def _stringify_tool_response(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)
    if content is None:
        content = ""
    tool_call_id = message.get("tool_call_id", "")
    name = message.get("name", "")
    header = "[Tool response"
    if name:
        header += f": {name}"
    if tool_call_id:
        header += f" (id={tool_call_id})"
    header += "]"
    return f"{header}\n{content}"

_TOOL_CALL_HEADER_RE = re.compile(
    r"\[\s*Tool\s*call\s*:\s*([^\]]+?)\s*\](?:\s*\(id=([^\)]*)\))?\s*\n\s*Arguments\s*:\s*",
    re.IGNORECASE,
)

def _extract_balanced_json(s: str, start: int) -> tuple[str, int]:
    if start >= len(s):
        return "", start
    open_ch = s[start]
    if open_ch not in "{[":
        end = s.find("\n", start)
        if end == -1:
            end = len(s)
        return s[start:end].strip(), end
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    i = start
    while i < len(s):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1], i + 1
        i += 1
    return s[start:].strip(), len(s)

def _parse_stringified_tool_calls(text: str) -> list[dict]:
    calls = []
    if not text:
        return calls
    for m in _TOOL_CALL_HEADER_RE.finditer(text):
        name = (m.group(1) or "").strip()
        call_id = (m.group(2) or "").strip()
        args_str, _ = _extract_balanced_json(text, m.end())
        if not name:
            continue
        arguments = None
        if args_str:
            try:
                arguments = json.loads(args_str)
            except Exception:
                arguments = args_str
        else:
            arguments = {}
        calls.append({"name": name, "id": call_id or "", "arguments": arguments})
    return calls

def _preprocess_tool_messages(messages: list[dict]) -> list[dict]:
    processed = []
    pending_system = []
    for msg in messages:
        if not isinstance(msg, dict):
            if pending_system:
                processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
                pending_system = []
            processed.append(msg)
            continue
        role = msg.get("role")
        tool_calls = msg.get("tool_calls")
        if role == "system":
            text = _extract_text(msg.get("content"))
            if text:
                pending_system.append(text)
            continue
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            text_parts = []
            existing_content = new_msg.get("content")
            if isinstance(existing_content, str) and existing_content.strip():
                text_parts.append(existing_content)
            elif isinstance(existing_content, list):
                for part in existing_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if t:
                            text_parts.append(t)
                    elif isinstance(part, str) and part:
                        text_parts.append(part)
            rendered = _stringify_tool_calls(tool_calls)
            if rendered:
                text_parts.append(rendered)
            new_msg["content"] = "\n\n".join(text_parts) if text_parts else rendered
            if pending_system:
                processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
                pending_system = []
            processed.append(new_msg)
        elif role == "tool":
            rendered = _stringify_tool_response(msg)
            if pending_system:
                rendered = "[System]\n" + "\n".join(pending_system) + "\n\n" + rendered
                pending_system = []
            processed.append({"role": "user", "content": rendered})
        elif role == "user":
            new_msg = dict(msg)
            if pending_system:
                existing = _extract_text(new_msg.get("content"))
                new_msg["content"] = "[System]\n" + "\n".join(pending_system) + (
                    ("\n\n" + existing) if existing else ""
                )
                pending_system = []
            processed.append(new_msg)
        else:
            if pending_system:
                processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
                pending_system = []
            processed.append(msg)
    if pending_system:
        processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
    return processed

def _merge_messages_to_single_user(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(msg.get("content"))
        if not text:
            continue
        header = "User" if role == "user" else "Assistant"
        parts.append(f"[{header}]\n{text}")
    if not parts:
        return messages
    return [{"role": "user", "content": "\n\n".join(parts)}]

def build_tool_system_prompt(tools: list[dict]) -> str:
    lines = [
        "Eres un agente en modo SIMULACIÓN de herramientas.",
        "Las herramientas REALES no las ejecutas tú: el usuario humano actúa como intermediario y las ejecuta por ti.",
        "Cuando una tarea requiera una herramienta, SIMULA la llamada respondiendo SOLO con un objeto JSON válido "
        "(sin markdown, sin texto adicional) en este formato exacto:",
        '{"tool_calls": [{"name": "NOMBRE_HERRAMIENTA", "arguments": {"param": "valor"}}]}',
        "Puedes incluir varias tool calls en el array. El valor de `arguments` DEBE ser un objeto JSON "
        "que coincida con el esquema de parámetros de la herramienta.",
        "IMPORTANTE: no digas que no tienes acceso a las herramientas. Simúlalas: emite el JSON y espera "
        "a que el intermediario te devuelva el resultado.",
        "NO inventes herramientas: usa únicamente las que aparecen en la lista disponible.",
        "Si ninguna herramienta es necesaria, responde en texto plano en español.",
    ]
    tool_names = []
    tool_schemas = {}
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            tool_names.append(name)
            params = fn.get("parameters")
            if isinstance(params, dict):
                tool_schemas[name] = params
    if tool_names:
        lines.append(f"Herramientas disponibles: {', '.join(tool_names)}")
        if tool_schemas:
            lines.append(f"Esquemas de parámetros (JSON Schema): {json.dumps(tool_schemas, ensure_ascii=True)}")
    return "\n".join(lines)

def inject_tools(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    if not tools:
        return messages
    prompt = (
        "Eres un agente con acceso a herramientas vía un intermediario humano.\n"
        + build_tool_system_prompt(tools)
        + "\n\nResuelve la tarea. Si hace falta una herramienta, SIMULA la llamada con el JSON y "
        "espera el resultado del intermediario (máximo 2 llamadas por turno). "
        "Cuando tengas la respuesta final, escríbela en texto plano, en español, sin JSON."
    )
    for m in messages:
        if m.get("role") in ("system", "developer"):
            m["content"] = prompt + "\n\n" + (m["content"] or "")
            return messages
    return [{"role": "system", "content": prompt}] + messages

def parse_tool_response(content: str, tool_names: list[str]) -> list[dict] | None:
    payload = content.strip()
    obj = _parse_json_maybe(payload)
    calls = None
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        calls = obj.get("tool_calls")
    elif isinstance(obj, dict) and ("name" in obj or "tool" in obj):
        calls = [obj]
    elif isinstance(obj, list):
        calls = obj
    openai_calls = []
    if isinstance(calls, list):
        idx = -1
        for c in calls:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("tool")
            if not isinstance(name, str) or not name or (tool_names and name not in tool_names):
                continue
            args = c.get("arguments")
            if isinstance(args, str):
                arguments_str = args
            else:
                try:
                    arguments_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=True)
                except Exception:
                    arguments_str = "{}"
            idx += 1
            openai_calls.append({
                "index": idx,
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": arguments_str},
            })
    if not openai_calls:
        stringified = _parse_stringified_tool_calls(content)
        if stringified:
            idx = -1
            for c in stringified:
                name = c.get("name")
                if not isinstance(name, str) or not name or (tool_names and name not in tool_names):
                    continue
                args = c.get("arguments")
                if isinstance(args, str):
                    arguments_str = args
                else:
                    try:
                        arguments_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=True)
                    except Exception:
                        arguments_str = "{}"
                idx += 1
                openai_calls.append({
                    "index": idx,
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_str},
                })
    return openai_calls if openai_calls else None

async def execute_tool_call(tc: dict, tools: list[dict], session: aiohttp.ClientSession | None = None) -> str:
    name = tc["function"]["name"]
    try:
        args = json.loads(tc["function"]["arguments"])
    except Exception:
        args = {}
    for tool in (tools or []):
        func = tool.get("function", {})
        if func.get("name") == name:
            url = func.get("url", "")
            if url and session:
                method = func.get("method", "POST").upper()
                try:
                    async with session.request(method, url, json=args, timeout=aiohttp.ClientTimeout(total=30)) as r:
                        return await r.text()
                except Exception as e:
                    return json.dumps({"error": str(e)})
            return json.dumps({"result": f"Tool '{name}' executed", "arguments": args})
    return json.dumps({"error": f"Unknown tool: {name}"})

def get_oss_headers(method: str, date_str: str, sts_data: dict, content_type: str) -> dict[str, str]:
    bucket_name = sts_data.get('bucketname', 'qwen-webui-prod')
    file_path = sts_data.get('file_path', '')
    access_key_id = sts_data.get('access_key_id')
    access_key_secret = sts_data.get('access_key_secret')
    security_token = sts_data.get('security_token')
    headers = {
        'Content-Type': content_type,
        'x-oss-content-sha256': 'UNSIGNED-PAYLOAD',
        'x-oss-date': date_str,
        'x-oss-security-token': security_token,
        'x-oss-user-agent': 'aliyun-sdk-js/6.23.0 Chrome 132.0.0.0 on Windows 10 64-bit'
    }
    headers_lower = {k.lower(): v for k, v in headers.items()}
    canonical_headers_list = []
    signed_headers_list = []
    required_headers = ['content-md5', 'content-type', 'x-oss-content-sha256', 'x-oss-date', 'x-oss-security-token',
                        'x-oss-user-agent']
    for header_name in sorted(required_headers):
        if header_name in headers_lower:
            canonical_headers_list.append(f"{header_name}:{headers_lower[header_name]}")
            signed_headers_list.append(header_name)
    canonical_headers = '\n'.join(canonical_headers_list) + '\n'
    canonical_uri = f"/{bucket_name}/{quote(file_path, safe='/')}"
    canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n\nUNSIGNED-PAYLOAD"
    date_parts = date_str.split('T')
    date_scope = f"{date_parts[0]}/ap-southeast-1/oss/aliyun_v4_request"
    string_to_sign = f"OSS4-HMAC-SHA256\n{date_str}\n{date_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

    def sign(key, msg):
        return hmac.new(key, msg.encode() if isinstance(msg, str) else msg, hashlib.sha256).digest()

    date_key = sign(f"aliyun_v4{access_key_secret}".encode(), date_parts[0])
    region_key = sign(date_key, "ap-southeast-1")
    service_key = sign(region_key, "oss")
    signing_key = sign(service_key, "aliyun_v4_request")
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    headers['authorization'] = f"OSS4-HMAC-SHA256 Credential={access_key_id}/{date_scope},Signature={signature}"
    return headers

def _detect_file_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b'\xff\xd8\xff'):
        return ".jpg", "image/jpeg"
    if data.startswith(b'\x89PNG'):
        return ".png", "image/png"
    if data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
        return ".gif", "image/gif"
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return ".webp", "image/webp"
    if data.startswith(b'%PDF'):
        return ".pdf", "application/pdf"
    if data.startswith(b'\x1aE\xdf\xa3'):
        return ".webm", "video/webm"
    if data.startswith(b'\x00\x00\x00\x18ftyp'):
        return ".mp4", "video/mp4"
    return ".bin", "application/octet-stream"

async def upload_file(
    data: bytes,
    filename: str,
    headers: dict,
    session: aiohttp.ClientSession,
    proxy: Optional[str] = None,
) -> dict:
    ext, file_type = _detect_file_type(data)
    filename = filename or f"file-{len(data)}{ext}"
    async with session.post(
        f'{QWEN_URL}/api/v2/files/getstsToken',
        json={"filename": filename, "filesize": len(data), "filetype": file_type},
        proxy=proxy,
    ) as r:
        await _raise_for_status(r)
        res = await r.json()
        data_res = res.get("data")
        if res.get("success") is False:
            code = data_res.get("code") if isinstance(data_res, dict) else None
            details = data_res.get("details") if isinstance(data_res, dict) else None
            raise QwenError(f"{code}:{details}")
        file_url = data_res.get("file_url")
        file_id = data_res.get("file_id")
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    oss_headers = get_oss_headers('PUT', date_str, data_res, file_type)
    async with session.put(file_url.split("?")[0], data=data, headers=oss_headers, proxy=proxy) as resp:
        await _raise_for_status(resp)
    if "image" in file_type:
        file_class, _type, show_type = "vision", "image", "image"
    elif "video" in file_type:
        file_class, _type, show_type = "video", "video", "video"
    elif "audio" in file_type:
        file_class, _type, show_type = "audio", "audio", "audio"
    else:
        file_class, _type, show_type = "document", "file", "file"
    return {
        "type": _type,
        "file": {
            "created_at": int(time.time() * 1000),
            "data": {},
            "filename": filename,
            "hash": None,
            "id": file_id,
            "meta": {"name": filename, "size": len(data), "content_type": file_type},
            "update_at": int(time.time() * 1000),
        },
        "id": file_id,
        "url": file_url,
        "name": filename,
        "collection_name": "",
        "progress": 0,
        "status": "uploaded",
        "greenNet": "success",
        "size": len(data),
        "error": "",
        "itemId": str(uuid.uuid4()),
        "file_type": file_type,
        "showType": show_type,
        "file_class": file_class,
        "uploadTaskId": str(uuid.uuid4()),
    }

async def upload(
    file: str | os.PathLike | bytes,
    filename: Optional[str] = None,
    token: Optional[str] = None,
    proxy: Optional[str] = None,
) -> dict:
    """Upload a local file (path or raw bytes) and return the file payload dict.

    High-level wrapper around :func:`upload_file` that builds headers and
    a session internally, so callers only pass the file itself.
    """
    if isinstance(file, (str, os.PathLike)):
        path = os.fspath(file)
        with open(path, "rb") as fh:
            data = fh.read()
        filename = filename or os.path.basename(path)
    else:
        data = bytes(file)
    headers = _get_headers(token)
    await _attach_extra_headers(headers, proxy)
    async with aiohttp.ClientSession(headers=headers) as session:
        return await upload_file(data, filename or f"file-{len(data)}", headers, session, proxy=proxy)

async def resolve_files(
    files: Optional[list],
    token: Optional[str] = None,
    proxy: Optional[str] = None,
) -> list[dict]:
    """Normalize mixed file inputs into upload-ready file payload dicts.

    Accepted per item:
      - dict            -> used as-is (already an upload payload)
      - local path      -> uploaded via :func:`upload`
      - http(s) URL     -> downloaded then uploaded
      - bytes           -> uploaded with a generated filename
    """
    out: list[dict] = []
    for item in files or []:
        if isinstance(item, dict):
            out.append(item)
            continue
        if isinstance(item, str) and item.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(item, proxy=proxy) as resp:
                    await _raise_for_status(resp)
                    data = await resp.read()
            name = item.split("?")[0].rsplit("/", 1)[-1] or f"remote-{len(data)}"
            out.append(await upload(data, name, token=token, proxy=proxy))
            continue
        out.append(await upload(item, token=token, proxy=proxy))
    return out

async def _send_message(
    model: str,
    conversation_id: str | None,
    parent_id: str | None,
    prompt: str,
    chat_type: str,
    feature_config: dict,
    proxy: str | None,
    timeout: int,
    stream: bool,
    token: str | None = None,
    files: list | None = None,
    aspect_ratio: str | None = None,
    _state: dict | None = None,
) -> AsyncIterator[dict]:
    message_id = str(uuid.uuid4())
    now = int(time.time() * 1000)

    headers = _get_headers(token)
    await _attach_extra_headers(headers, proxy)

    async with aiohttp.ClientSession(headers=headers) as session:
        if not conversation_id:
            chat_payload = {
                "title": "New Chat",
                "models": [model],
                "chat_mode": "normal",
                "chat_type": chat_type,
                "timestamp": now,
                "project_id": "",
            }
            async with session.post(f'{QWEN_URL}/api/v2/chats/new', json=chat_payload, proxy=proxy) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    if text.strip().startswith('<'):
                        raise QwenError("WAF blocked chat creation")
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        raise QwenError(f"Non-JSON chat response: {text[:100]}")
                else:
                    data = await resp.json()
                if not (data.get('success') and data['data'].get('id')):
                    raise QwenError(f"Failed to create chat: {data}")
                conversation_id = data['data']['id']

        # The upstream API rejects stream=false with Bad_Request (verified
        # live), so the payload always requests SSE.  The `stream` flag only
        # affects how callers consume the yielded events.
        msg_payload = {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": conversation_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": parent_id,
            "messages": [
                {
                    "fid": message_id,
                    "parentId": parent_id,
                    "childrenIds": [],
                    "role": "user",
                    "content": prompt,
                    "user_action": "chat",
                    "files": files or [],
                    "timestamp": now,
                    "models": [model],
                    "chat_type": chat_type,
                    "feature_config": feature_config,
                    "extra": {"meta": {"subChatType": chat_type}},
                    "sub_chat_type": chat_type,
                }
            ],
            "timestamp": now,
        }
        if aspect_ratio:
            msg_payload["size"] = aspect_ratio

        async with session.post(
            f'{QWEN_URL}/api/v2/chat/completions?chat_id={conversation_id}',
            json=msg_payload,
            proxy=proxy,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            await _raise_for_status(resp)
            accumulated = ""
            usage: dict = {}
            response_id = None
            async for chunk in _sse_stream(resp):
                error = chunk.get("error", {})
                if error:
                    raise QwenError(f'{error.get("code")}: {error.get("details", error)}')
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("response_id"):
                    response_id = chunk["response_id"]
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                phase = delta.get("phase")
                content = delta.get("content")
                reasoning = delta.get("reasoning")
                status = delta.get("status")
                extra = delta.get("extra", {})
                if phase == "image_gen" and status == "typing":
                    yield {"type": "image", "data": {"url": content, "extra": extra}}
                    continue
                elif phase == "image_gen" and status == "finished":
                    yield {"type": "image_done"}
                    continue
                if reasoning:
                    yield {"type": "reasoning", "data": reasoning}
                elif content and phase == "answer":
                    accumulated += content
                    yield {"type": "content", "data": content}
                elif content and phase == "think":
                    yield {"type": "reasoning", "data": content}
            if _state is not None:
                _state["conv_id"] = conversation_id
                _state["parent_id_out"] = response_id or message_id
                _state["content"] = accumulated
                _state["usage"] = usage
            yield {"type": "usage", "data": usage}
            yield {"type": "done"}

async def fetch_models(token: Optional[str] = None, proxy: Optional[str] = None) -> list[str]:
    headers = _get_headers(token)
    await _attach_extra_headers(headers, proxy)
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f'{QWEN_URL}/api/models', proxy=proxy) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]

def _build_feature_config(
    model: str,
    reasoning_effort: Optional[str],
    thinking_mode: Optional[str],
) -> dict:
    """Build the ``feature_config`` payload sent to chat.qwen.ai.

    ``reasoning_effort`` must be one of ``REASONING_EFFORT_LEVELS``
    (``none``, ``low``, ``medium``, ``high``).  An explicit ``"none"``
    disables thinking for *all* models, including those in
    ``MODELS_REQUIRE_THINKING``: the upstream API accepts
    ``thinking_enabled: false`` everywhere (verified live).

    For ``low``/``medium``/``high`` the SDK differentiates the
    ``thinking_mode`` and ``thinking_budget`` so the API can tune
    reasoning depth per level.
    """
    if reasoning_effort is not None and reasoning_effort.lower() not in REASONING_EFFORT_LEVELS:
        raise QwenError(
            f"Invalid reasoning_effort '{reasoning_effort}'. "
            f"Must be one of: {', '.join(REASONING_EFFORT_LEVELS)}"
        )

    if not reasoning_effort or reasoning_effort.lower() == "none":
        return {
            "thinking_enabled": False,
            "output_schema": "phase",
            "thinking_budget": 81920,
        }

    effort = reasoning_effort.lower()
    _thinking_mode = thinking_mode or ("Fast" if effort == "low" else "Auto")
    auto_thinking = _thinking_mode == "Auto"
    thinking_budget = {"low": 40960, "medium": 81920, "high": 163840}[effort]

    return {
        "auto_thinking": auto_thinking,
        "thinking_mode": _thinking_mode,
        "thinking_enabled": True,
        "output_schema": "phase",
        "research_mode": "normal",
        "auto_search": True,
        "thinking_budget": thinking_budget,
    }

async def create_chat(
    model: str,
    messages: list[dict],
    token: Optional[str] = None,
    chat_type: str = "t2t",
    stream: bool = True,
    tools: Optional[list[dict]] = None,
    reasoning_effort: Optional[str] = "none",
    thinking_mode: Optional[str] = None,
    conversation_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    proxy: Optional[str] = None,
    timeout: int = 300,
    emit_tool_calls: bool = False,
    files: Optional[list[dict]] = None,
    aspect_ratio: Optional[Literal["1:1", "4:3", "3:4", "16:9", "9:16"]] = None,
    **kwargs: Any,
) -> AsyncIterator[dict]:
    feature_config = _build_feature_config(model, reasoning_effort, thinking_mode)

    tool_names = []
    if tools:
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function":
                fn = t.get("function", {})
                if isinstance(fn, dict) and fn.get("name"):
                    tool_names.append(fn["name"])

    messages = inject_tools(messages, tools)
    messages = _preprocess_tool_messages(messages)
    if conversation_id is None:
        messages = _merge_messages_to_single_user(messages)

    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt = content.strip()
            elif isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        prompt = part["text"].strip()
                        break
            if prompt:
                break

    _content_buffer: list[str] = []
    _yielded_content = False

    for attempt in range(3):
        try:
            state: dict = {}
            async for event in _send_message(
                model, conversation_id, parent_id, prompt, chat_type, feature_config,
                proxy, timeout, stream, token, files, aspect_ratio, _state=state,
            ):
                if event["type"] == "content":
                    _content_buffer.append(event["data"])
                    if not tool_names:
                        yield event
                        _yielded_content = True
                elif event["type"] in ("reasoning", "usage", "image", "image_done"):
                    yield event
                elif event["type"] == "done":
                    content = "".join(_content_buffer)
                    if tool_names:
                        tool_calls = parse_tool_response(content, tool_names)
                        if tool_calls:
                            conv_id = state.get("conv_id") or conversation_id or ""
                            parent_id_out = state.get("parent_id_out") or parent_id or ""
                            if emit_tool_calls:
                                yield {"type": "usage", "data": state.get("usage", {})}
                                yield {"type": "tool_calls", "data": tool_calls}
                                yield {"type": "done", "conversation_id": conv_id, "parent_id": parent_id_out}
                                return
                            async for evt in _run_tool_calls(
                                tool_calls, tools, model, conv_id, parent_id_out,
                                chat_type, feature_config, proxy, timeout, stream, token,
                            ):
                                yield evt
                            return
                    if not _yielded_content:
                        yield {"type": "content", "data": content}
                    yield {"type": "done", "conversation_id": state.get("conv_id") or conversation_id or "", "parent_id": state.get("parent_id_out") or parent_id or ""}
                    return
            break
        except (aiohttp.ClientResponseError, QwenRateLimitError) as e:
            if "429" in str(e) or "RateLimited" in str(e):
                conversation_id = None
                _content_buffer.clear()
                _yielded_content = False
                await asyncio.sleep(2)
                continue
            raise
    else:
        raise QwenRateLimitError("Rate limit exceeded after 3 attempts")

async def _run_tool_calls(
    tool_calls: list[dict],
    tools: list[dict] | None,
    model: str,
    conv_id: str,
    parent_id_out: str,
    chat_type: str,
    feature_config: dict,
    proxy: str | None,
    timeout: int,
    stream: bool,
    token: str | None,
) -> AsyncIterator[dict]:
    async with aiohttp.ClientSession() as tool_session:
        for tc in tool_calls:
            # Pass a live session so URL-based tools are actually invoked
            # (execute_tool_call needs a session to make the HTTP request).
            tool_result = await execute_tool_call(tc, tools, tool_session)
            for retry in range(3):
                try:
                    state2: dict = {}
                    async for evt in _send_message(
                        model, conv_id, parent_id_out,
                        f"Tool result: {tool_result}",
                        chat_type, feature_config, proxy, timeout, stream, token,
                        _state=state2,
                    ):
                        yield evt
                        if evt["type"] == "done":
                            break
                    yield {
                        "type": "done",
                        "conversation_id": state2.get("conv_id") or conv_id or "",
                        "parent_id": state2.get("parent_id_out") or parent_id_out or "",
                    }
                    break
                except (aiohttp.ClientResponseError, QwenRateLimitError) as e:
                    if "429" in str(e) or "RateLimited" in str(e):
                        await asyncio.sleep(2)
                        continue
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise QwenError(f"Follow-up failed: {e}")
    return


async def _raise_for_status(response: aiohttp.ClientResponse):
    status = response.status
    if status == 429:
        raise QwenRateLimitError(f"Rate limited (status {status})")
    if status == 401:
        raise QwenAuthError(f"Unauthorized (status {status})")
    if 200 <= status < 300:
        return
    text = await response.text()
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html"):
        raise QwenError(f"WAF blocked (status {status})")
    try:
        data = json.loads(text)
        msg = data.get("message", data.get("error", text))
    except json.JSONDecodeError:
        msg = text
    raise QwenError(f"HTTP {status}: {msg[:200]}")
