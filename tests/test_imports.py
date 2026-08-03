import qwen_reverse


def test_version():
    assert qwen_reverse.__version__ == "0.1.2"


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


def test_generate_model_override(monkeypatch):
    seen = {}

    async def fake_create_chat(model, messages, **kwargs):
        seen["model"] = model
        seen["kwargs_model"] = kwargs.get("model", "<absent>")
        yield {"type": "content", "data": "hi"}

    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    gen = qwen_reverse.Generative(model="qwen3.7-plus")

    async def run():
        return await gen.generate("x", model="qwen3.8-max")

    import asyncio
    result = asyncio.run(run())
    assert result == "hi"
    assert seen["model"] == "qwen3.8-max"
    assert seen["kwargs_model"] == "<absent>"


def test_stream_model_override(monkeypatch):
    seen = {}

    async def fake_create_chat(model, messages, **kwargs):
        seen["model"] = model
        yield {"type": "content", "data": "a"}
        yield {"type": "content", "data": "b"}

    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    gen = qwen_reverse.Generative(model="qwen3.7-plus")

    import asyncio
    events = asyncio.run(_collect(gen.stream("x", model="qwen3.7-max")))
    assert [e["data"] for e in events] == ["a", "b"]
    assert seen["model"] == "qwen3.7-max"


async def _collect(agen):
    return [e async for e in agen]


def test_conversation_model_override(monkeypatch):
    seen = {}

    async def fake_create_chat(model, messages, **kwargs):
        seen["model"] = model
        yield {"type": "content", "data": "ok"}
        yield {"type": "done", "conversation_id": "c1", "parent_id": "p1"}

    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    conv = qwen_reverse.Conversation(model="qwen3.7-plus")

    import asyncio
    result = asyncio.run(conv.send("x", model="qwen3.8-max"))
    assert result == "ok"
    assert seen["model"] == "qwen3.8-max"
    assert conv.conversation_id == "c1"
