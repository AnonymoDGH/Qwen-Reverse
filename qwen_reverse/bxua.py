import random
import time
import base64
import hashlib
from typing import Dict, List, Any, Optional
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import json

DEFAULT_TEMPLATE: Dict[str, Any] = {
    "deviceId": "84985177a19a010dea49",
    "sdkVersion": "websdk-2.3.15d",
    "initTimestamp": "1765348410850",
    "field3": "91",
    "field4": "1|15",
    "language": "zh-CN",
    "timezoneOffset": "-480",
    "colorDepth": "16705151|12791",
    "screenInfo": "1470|956|283|797|158|0|1470|956|1470|798|0|0",
    "field9": "5",
    "platform": "MacIntel",
    "field11": "10",
    "webglRenderer": (
        "ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified Version)"
        "|Google Inc. (Apple)"
    ),
    "field13": "30|30",
    "field14": "0",
    "field15": "28",
    "pluginCount": "5",
    "vendor": "Google Inc.",
    "field29": "8",
    "touchInfo": "-1|0|0|0|0",
    "field32": "11",
    "field35": "0",
    "mode": "P",
}


class BXUAGenerator:
    """Generates the `bx-ua` header value used by the Qwen WAF."""

    def __init__(self):
        self.version = "231"
        self.aes_key = None
        self.aes_iv = None

    def _generate_key_iv(self, seed_data: str) -> tuple:
        seed_hash = hashlib.sha256(seed_data.encode()).digest()
        return seed_hash[:16], seed_hash[16:32]

    def _encrypt_aes_cbc(self, data: bytes, key: bytes, iv: bytes) -> bytes:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(pad(data, AES.block_size))

    def _create_payload(self, fingerprint: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        fields = fingerprint.split("^")
        payload = {
            "v": self.version,
            "ts": timestamp,
            "fp": fingerprint,
            "d": {
                "deviceId": fields[0],
                "sdkVer": fields[1],
                "lang": fields[5],
                "tz": fields[6],
                "platform": fields[10],
                "renderer": fields[12],
                "mode": fields[23],
                "vendor": fields[28],
            },
            "rnd": random.randint(1000, 9999),
            "seq": 1,
        }
        checksum_str = f"{fingerprint}{timestamp}{payload['rnd']}"
        payload["cs"] = hashlib.md5(checksum_str.encode()).hexdigest()[:8]
        return payload

    def generate(self, fingerprint: str, options: Optional[Dict[str, Any]] = None) -> str:
        if options is None:
            options = {}
        timestamp = options.get("timestamp") or int(time.time() * 1000)
        payload = self._create_payload(fingerprint, timestamp)
        payload_json = json.dumps(payload, separators=(',', ':'))
        seed = options.get("seed", fingerprint)
        key, iv = self._generate_key_iv(seed)
        encrypted = self._encrypt_aes_cbc(payload_json.encode(), key, iv)
        encrypted_b64 = base64.b64encode(encrypted).decode()
        return f"{self.version}!{encrypted_b64}"


def generate_device_id() -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(20))


def generate_hash() -> int:
    return random.randint(0, 0xFFFFFFFF)


def generate_fingerprint(options: Dict[str, Any] = None) -> str:
    if options is None:
        options = {}
    config = DEFAULT_TEMPLATE.copy()
    if "custom" in options and isinstance(options["custom"], dict):
        config.update(options["custom"])
    device_id = options.get("deviceId") or generate_device_id()
    current_timestamp = int(time.time() * 1000)
    plugin_hash = generate_hash()
    canvas_hash = generate_hash()
    ua_hash1 = generate_hash()
    ua_hash2 = generate_hash()
    url_hash = generate_hash()
    doc_hash = random.randint(10, 100)
    fields: List[Any] = [
        device_id, config["sdkVersion"], config["initTimestamp"], config["field3"],
        config["field4"], config["language"], config["timezoneOffset"], config["colorDepth"],
        config["screenInfo"], config["field9"], config["platform"], config["field11"],
        config["webglRenderer"], config["field13"], config["field14"], config["field15"],
        f'{config["pluginCount"]}|{plugin_hash}', canvas_hash, ua_hash1,
        "1", "0", "1", "0", config["mode"], "0", "0", "0", "416",
        config["vendor"], config["field29"], config["touchInfo"], ua_hash2,
        config["field32"], current_timestamp, url_hash, config["field35"], doc_hash,
    ]
    return "^".join(map(str, fields))


def generate_bx_ua(options: Dict[str, Any] = None) -> str:
    """One-shot bx-ua generation for a fresh fingerprint."""
    fp = generate_fingerprint(options)
    return BXUAGenerator().generate(fp, {"timestamp": int(time.time() * 1000)})
