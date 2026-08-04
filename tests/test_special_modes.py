
import sys
sys.path.insert(0, r"C:\Users\USUARIO\Qwen-Reverse")
sys.path.insert(0, r"C:\Users\USUARIO\Qwen-Reverse\server")

from server.app import _parse_model_suffix, _format_search_info

def test_special_suffixes():
    assert _parse_model_suffix("qwen3.8-max-web-dev") == ("qwen3.8-max", "web_dev", None)
    assert _parse_model_suffix("qwen3.8-max-deep-research") == ("qwen3.8-max", "deep_research", None)
    assert _parse_model_suffix("qwen3.8-max-artifacts") == ("qwen3.8-max", "artifacts", None)
    assert _parse_model_suffix("qwen3.8-max-slides") == ("qwen3.8-max", "slides", None)
    assert _parse_model_suffix("qwen3.8-max-web-dev-thinking") == ("qwen3.8-max", "web_dev", "high")

def test_format_search_info():
    citations = [
        {"title": "NASA Official", "url": "https://nasa.gov", "snippet": "NASA news"},
        {"title": "SpaceX Launch", "url": "https://spacex.com", "snippet": "Starship info"}
    ]
    table_out = _format_search_info(citations, mode="table")
    assert "| Fuentes / References | URL |" in table_out
    assert "[NASA Official](https://nasa.gov)" in table_out
    
    text_out = _format_search_info(citations, mode="text")
    assert "1. [NASA Official](https://nasa.gov)" in text_out
