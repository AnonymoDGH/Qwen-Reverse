import qwen_reverse


def test_version():
    assert qwen_reverse.__version__ == "0.2.0"


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


def test_image_edit_class():
    ie = qwen_reverse.ImageEdit()
    assert ie.chat_type == "image_edit"
    assert ie.model == qwen_reverse.DEFAULT_MODEL


def test_edit_image_chat_type_and_files(monkeypatch):
    seen = {}

    async def fake_resolve_files(files, token=None, proxy=None):
        return [{"file": "fake_payload"}]

    async def fake_create_chat(model, messages, **kwargs):
        seen["chat_type"] = kwargs.get("chat_type")
        seen["files"] = kwargs.get("files")
        seen["model"] = model
        seen["aspect_ratio"] = kwargs.get("aspect_ratio")
        yield {"type": "image", "data": {"url": "https://cdn.qwenlm.ai/x.png"}}
        yield {"type": "image_done"}

    monkeypatch.setattr(qwen_reverse, "resolve_files", fake_resolve_files)
    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    gen = qwen_reverse.Generative(model="qwen3.7-plus")

    import asyncio

    urls = asyncio.run(gen.edit_image("draw an arrow", "logo.png", aspect_ratio="4:3", model="qwen3.8-max"))
    assert urls == ["https://cdn.qwenlm.ai/x.png"]
    assert seen["chat_type"] == "image_edit"
    assert seen["files"] == [{"file": "fake_payload"}]
    assert seen["model"] == "qwen3.8-max"
    assert seen["aspect_ratio"] == "4:3"


def test_image_edit_class_edit_maps_to_edit_image(monkeypatch):
    seen = {}

    async def fake_resolve_files(files, token=None, proxy=None):
        return [{"file": "p"}]

    async def fake_create_chat(model, messages, **kwargs):
        seen["chat_type"] = kwargs.get("chat_type")
        yield {"type": "image", "data": {"url": "u"}}

    monkeypatch.setattr(qwen_reverse, "resolve_files", fake_resolve_files)
    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    ie = qwen_reverse.ImageEdit()

    import asyncio

    asyncio.run(ie.edit("arrow", "logo.png"))
    assert seen["chat_type"] == "image_edit"


def test_conversation_files_passed(monkeypatch):
    seen = {}

    async def fake_resolve_files(files, token=None, proxy=None):
        seen["files_in"] = files
        return [{"file": "resolved"}]

    async def fake_create_chat(model, messages, **kwargs):
        seen["files"] = kwargs.get("files")
        seen["model"] = model
        yield {"type": "content", "data": "ok"}
        yield {"type": "done", "conversation_id": "c1", "parent_id": "p1"}

    monkeypatch.setattr(qwen_reverse, "resolve_files", fake_resolve_files)
    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    conv = qwen_reverse.Conversation(model="qwen3.7-plus")

    import asyncio

    result = asyncio.run(conv.send("look at this", files=["a.png", b"raw"], model="qwen3.8-max"))
    assert result == "ok"
    assert seen["files_in"] == ["a.png", b"raw"]
    assert seen["files"] == [{"file": "resolved"}]
    assert seen["model"] == "qwen3.8-max"


def test_upload_and_resolve_files_public():
    assert callable(qwen_reverse.upload)
    assert callable(qwen_reverse.resolve_files)


def test_alist_helper():
    import asyncio

    async def gen():
        yield 1
        yield 2
        yield 3

    assert asyncio.run(qwen_reverse.alist(gen())) == [1, 2, 3]


def test_conversation_async_context_manager():
    import asyncio

    async def run():
        async with qwen_reverse.Conversation() as conv:
            conv.messages.append({"role": "user", "content": "x"})
            conv.conversation_id = "abc"
            assert conv.messages == [{"role": "user", "content": "x"}]
        # __aexit__ resets state
        assert conv.messages == []
        assert conv.conversation_id is None

    asyncio.run(run())


def test_generative_reasoning_effort_instance_defaults():
    gen = qwen_reverse.Generative(reasoning_effort="medium")
    assert gen.reasoning_effort == "medium"
    gen2 = qwen_reverse.Generative()
    assert gen2.reasoning_effort is None


def test_generative_generate_passes_reasoning_effort(monkeypatch):
    import asyncio

    seen = {}

    async def fake_create_chat(model, messages, **kwargs):
        seen["reasoning_effort"] = kwargs.get("reasoning_effort")
        seen["thinking_mode"] = kwargs.get("thinking_mode")
        yield {"type": "content", "data": "hi"}

    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    gen = qwen_reverse.Generative(reasoning_effort="high", thinking_mode="Thinking")

    result = asyncio.run(gen.generate("x"))
    assert result == "hi"
    assert seen["reasoning_effort"] == "high"
    assert seen["thinking_mode"] == "Thinking"


def test_conversation_reasoning_effort_passed(monkeypatch):
    import asyncio

    seen = {}

    async def fake_create_chat(model, messages, **kwargs):
        seen["reasoning_effort"] = kwargs.get("reasoning_effort")
        seen["thinking_mode"] = kwargs.get("thinking_mode")
        yield {"type": "content", "data": "ok"}
        yield {"type": "done", "conversation_id": "c1", "parent_id": "p1"}

    monkeypatch.setattr(qwen_reverse, "create_chat", fake_create_chat)
    conv = qwen_reverse.Conversation(reasoning_effort="low", thinking_mode="Fast")

    result = asyncio.run(conv.send("x"))
    assert result == "ok"
    assert seen["reasoning_effort"] == "low"
    assert seen["thinking_mode"] == "Fast"
