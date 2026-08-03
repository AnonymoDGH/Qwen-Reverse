import qwen_reverse


def test_version():
    assert qwen_reverse.__version__ == "0.1.0"


def test_public_exports_exist():
    for name in qwen_reverse.__all__:
        assert hasattr(qwen_reverse, name), f"missing export: {name}"


def test_default_model_in_fallback_list():
    assert qwen_reverse.DEFAULT_MODEL in qwen_reverse.models


def test_chat_types():
    for t in ("t2t", "t2i", "t2v", "search", "artifacts"):
        assert t in qwen_reverse.CHAT_TYPES


def test_aspect_ratios():
    for r in ("1:1", "4:3", "3:4", "16:9", "9:16"):
        assert r in qwen_reverse.ASPECT_RATIOS


def test_classes_instantiate():
    conv = qwen_reverse.Conversation()
    assert conv.model == qwen_reverse.DEFAULT_MODEL
    assert conv.chat_type == "t2t"
    assert conv.conversation_id is None
    img = qwen_reverse.Image()
    assert img.chat_type == "t2i"
    vid = qwen_reverse.Video()
    assert vid.chat_type == "t2v"


def test_conversation_reset():
    conv = qwen_reverse.Conversation()
    conv.messages.append({"role": "user", "content": "x"})
    conv.conversation_id = "abc"
    conv.reset()
    assert conv.messages == []
    assert conv.conversation_id is None
