from qwen_reverse.bxua import BXUAGenerator, generate_bx_ua, generate_fingerprint


def test_bx_ua_shape():
    ua = generate_bx_ua()
    assert ua.startswith("231!")
    assert len(ua) > 100


def test_bx_ua_roundtrip_same_fingerprint():
    fp = generate_fingerprint()
    a = BXUAGenerator().generate(fp)
    b = BXUAGenerator().generate(fp)
    assert a.startswith("231!")
    assert b.startswith("231!")


def test_generate_accepts_valid_fingerprint():
    gen = BXUAGenerator()
    ua = gen.generate(generate_fingerprint())
    assert ua.startswith("231!")
