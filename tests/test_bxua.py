import base64
import hashlib
import json
from unittest.mock import patch

from qwen_reverse.bxua import BXUAGenerator, generate_bx_ua, generate_fingerprint
from qwen_reverse.fingerprint import generate_device_id


def test_bx_ua_shape():
    ua = generate_bx_ua()
    assert ua.startswith("231!")
    payload_b64 = ua.split("!", 1)[1]
    assert len(payload_b64) > 50


def test_bx_ua_decrypts_to_valid_json_payload():
    gen = BXUAGenerator()
    fp = generate_fingerprint({"deviceId": "deadbeefdeadbeefdead"})
    ua = gen.generate(fp)

    version, b64 = ua.split("!", 1)
    assert version == "231"

    seed_hash = hashlib.sha256(fp.encode()).digest()
    key, iv = seed_hash[:16], seed_hash[16:32]
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(base64.b64decode(b64)), AES.block_size)
    payload = json.loads(decrypted.decode())

    assert payload["v"] == "231"
    assert payload["fp"] == fp
    assert "ts" in payload and isinstance(payload["ts"], int)
    assert "d" in payload and isinstance(payload["d"], dict)
    assert payload["d"]["deviceId"] == "deadbeefdeadbeefdead"
    expected_cs = hashlib.md5(f"{fp}{payload['ts']}{payload['rnd']}".encode()).hexdigest()[:8]
    assert payload["cs"] == expected_cs


def test_bx_ua_deterministic_with_fixed_timestamp_and_random():
    fp = generate_fingerprint({"deviceId": "0" * 20})
    gen = BXUAGenerator()
    ts = 1700000000000

    with patch("qwen_reverse.bxua.random.randint", return_value=4242):
        a = gen.generate(fp, {"timestamp": ts})
        b = gen.generate(fp, {"timestamp": ts})

    assert a == b


def test_bx_ua_differs_across_fingerprints():
    fp1 = generate_fingerprint({"deviceId": "a" * 20})
    fp2 = generate_fingerprint({"deviceId": "b" * 20})
    gen = BXUAGenerator()
    ts = 1700000000000
    with patch("qwen_reverse.bxua.random.randint", return_value=1):
        ua1 = gen.generate(fp1, {"timestamp": ts})
        ua2 = gen.generate(fp2, {"timestamp": ts})
    assert ua1 != ua2


def test_generate_accepts_valid_fingerprint():
    gen = BXUAGenerator()
    ua = gen.generate(generate_fingerprint())
    assert ua.startswith("231!")


def test_generate_device_id_is_hex_of_length_20():
    dev = generate_device_id()
    assert len(dev) == 20
    int(dev, 16)
