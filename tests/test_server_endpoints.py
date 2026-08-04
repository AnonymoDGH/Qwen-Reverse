
import sys
sys.path.insert(0, r"C:\Users\USUARIO\Qwen-Reverse")
sys.path.insert(0, r"C:\Users\USUARIO\Qwen-Reverse\server")

from server.app import _parse_model_suffix, _map_size_to_aspect_ratio

def test_parse_model_suffix():
    assert _parse_model_suffix("qwen3.8-max") == ("qwen3.8-max", "t2t", None)
    assert _parse_model_suffix("qwen3.8-max-search") == ("qwen3.8-max", "search", None)
    assert _parse_model_suffix("qwen3.8-max-thinking") == ("qwen3.8-max", "t2t", "high")
    assert _parse_model_suffix("qwen3.8-max-search-thinking") == ("qwen3.8-max", "search", "high")
    assert _parse_model_suffix("qwen3.8-max-image") == ("qwen3.8-max", "t2i", None)
    assert _parse_model_suffix("claude-opus-4-6-search") == ("qwen3.8-max", "search", None)

def test_map_size_to_aspect_ratio():
    assert _map_size_to_aspect_ratio("1024x1024") == "1:1"
    assert _map_size_to_aspect_ratio("1792x1024") == "16:9"
    assert _map_size_to_aspect_ratio("1024x1792") == "9:16"
    assert _map_size_to_aspect_ratio("1024x768") == "4:3"
