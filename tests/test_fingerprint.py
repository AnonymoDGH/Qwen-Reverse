from qwen_reverse import generate_fingerprint, parse_fingerprint


def test_fingerprint_field_count():
    fp = generate_fingerprint()
    assert len(fp.split("^")) == 37


def test_fingerprint_platform_preset():
    fp = generate_fingerprint({"platform": "macos"})
    assert parse_fingerprint(fp)["platform"] == "MacIntel"


def test_fingerprint_custom_device_id():
    fp = generate_fingerprint({"deviceId": "0" * 20})
    assert parse_fingerprint(fp)["deviceId"] == "0" * 20


def test_fingerprint_roundtrip_parse():
    fp = generate_fingerprint({"platform": "linux"})
    parsed = parse_fingerprint(fp)
    assert len(parsed["raw"]) == 37
    assert parsed["timestamp"] == fp.split("^")[33]
