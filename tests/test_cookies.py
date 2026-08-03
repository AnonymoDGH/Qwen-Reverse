from qwen_reverse import generate_cookies, generate_fingerprint, parse_fingerprint


def test_cookie_shape():
    data = generate_cookies()
    assert data["ssxmod_itna"].startswith("1-")
    assert data["ssxmod_itna2"].startswith("1-")
    assert data["timestamp"] > 0
    assert len(data["ssxmod_itna"]) > 20


def test_cookies_accept_real_fingerprint():
    fp = generate_fingerprint({"platform": "windows", "screen": "1920x1080"})
    data = generate_cookies(real_data=fp)
    assert data["ssxmod_itna"].startswith("1-")


def test_cookies_vary_between_calls():
    a = generate_cookies()
    b = generate_cookies()
    assert a["ssxmod_itna"] != b["ssxmod_itna"]
