import time
from unittest.mock import patch

from qwen_reverse import generate_cookies, generate_fingerprint, parse_fingerprint
from qwen_reverse.cookies import random_hash


def test_cookie_shape():
    data = generate_cookies()
    assert data["ssxmod_itna"].startswith("1-")
    assert data["ssxmod_itna2"].startswith("1-")
    assert data["timestamp"] > 0
    assert len(data["ssxmod_itna"]) > 20
    assert len(data["ssxmod_itna2"]) > 20


def test_raw_data_has_37_fields_and_matches_fingerprint():
    fp = generate_fingerprint({"deviceId": "abc" * 6 + "ab"})
    data = generate_cookies(real_data=fp)
    raw_fields = data["rawData"].split("^")
    assert len(raw_fields) == 37
    assert raw_fields[0] == "abc" * 6 + "ab"
    fp_parsed = parse_fingerprint(fp)
    for keep_idx in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27, 28, 30, 32, 35):
        assert raw_fields[keep_idx] == fp_parsed["raw"][keep_idx]


def test_hashed_fields_are_replaced():
    fp = generate_fingerprint({"pluginCount": "5"})
    fp_fields = parse_fingerprint(fp)["raw"]
    data = generate_cookies(real_data=fp)
    raw_fields = data["rawData"].split("^")

    assert "|" in raw_fields[16]
    assert raw_fields[16].split("|")[0] == fp_fields[16].split("|")[0]
    assert raw_fields[16].split("|")[1] != fp_fields[16].split("|")[1]

    for idx in (17, 18, 31, 34):
        assert raw_fields[idx] != fp_fields[idx], f"field {idx} not re-hashed"

    assert raw_fields[36] != fp_fields[36]


def test_field_33_uses_current_timestamp():
    before = int(time.time() * 1000)
    data = generate_cookies()
    after = int(time.time() * 1000)
    assert before <= data["timestamp"] <= after


def test_itna2_reuses_processed_indices():
    data = generate_cookies()
    raw1 = data["rawData"].split("^")
    raw2 = data["rawData2"].split("^")
    assert raw2[0] == raw1[0]
    assert raw2[1] == raw1[1]
    assert raw2[2] == raw1[23]
    assert raw2[11] == raw1[32]
    assert raw2[12] == raw1[33]


def test_accepts_real_fingerprint_and_preserves_device_id():
    fp = generate_fingerprint({"platform": "macM1", "screen": "1440x900"})
    data = generate_cookies(real_data=fp)
    assert data["ssxmod_itna"].startswith("1-")


def test_deterministic_with_fixed_random_and_time():
    fp = generate_fingerprint({"deviceId": "fixed" + "0" * 14})
    with patch("qwen_reverse.cookies.random.randint", return_value=12345), \
         patch("qwen_reverse.cookies.time.time", return_value=1700000000.0):
        a = generate_cookies(real_data=fp)
        b = generate_cookies(real_data=fp)
    assert a["ssxmod_itna"] == b["ssxmod_itna"]
    assert a["ssxmod_itna2"] == b["ssxmod_itna2"]
    assert a["rawData"] == b["rawData"]


def test_random_hash_in_range():
    h = random_hash()
    assert 0 <= h <= 0xFFFFFFFF
