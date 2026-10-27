"""qwen-reverse CLI — start the API server and launch AI coding agents.

Usage:
    qwen-reverse                    # start server, choose agent interactively
    qwen-reverse --port 8090        # specific port
    qwen-reverse --claude           # start server + launch Claude Code
    qwen-reverse --gemini           # start server + launch Gemini CLI
    qwen-reverse --cursor           # start server + launch Cursor
    qwen-reverse --qwen             # start server + launch Qwen Code
    qwen-reverse --openai           # start server + show OpenAI setup
"""

import argparse
import os
import shutil
import socket
import subprocess
import threading
import time
from .server.app import _parse_model_suffix


# ── Agent definitions ─────────────────────────────────────────────────
# Each agent knows: how to find it on PATH, what env vars it needs,
# and which API format it speaks (anthropic vs openai).

AGENTS = {
    "claude": {
        "cmd": "claude",
        "name": "Claude Code",
        "format": "anthropic",
        "install": "npm install -g @anthropic-ai/claude-code",
        "env": {
            "ANTHROPIC_BASE_URL": "{base_url}",
            "ANTHROPIC_API_KEY": "sk-ant-api03-qwen-reverse-dummy-token-00000000000000000000000000000000000000000000000",
            "ANTHROPIC_MODEL": "claude-opus-4-6",
            "CLAUDE_CODE_USE_BEDROCK": "0",
            "CLAUDE_CODE_USE_VERTEX": "0",
            "AWS_PROFILE": "",
            "AWS_ACCESS_KEY_ID": "",
            "AWS_SECRET_ACCESS_KEY": "",
            "AWS_SESSION_TOKEN": "",
            "AWS_REGION": "",
        },
    },
    "gemini": {
        "cmd": "gemini",
        "name": "Gemini CLI",
        "format": "openai",
        "install": "npm install -g @google/gemini-cli",
        "env": {
            "OPENAI_BASE_URL": "{base_url}",
            "OPENAI_API_KEY": "qwen-reverse",
            "OPENAI_MODEL": "qwen3.8-max",
        },
    },
    "cursor": {
        "cmd": "cursor",
        "name": "Cursor",
        "format": "openai",
        "install": "https://cursor.com/download",
        "env": {
            "OPENAI_BASE_URL": "{base_url}",
            "OPENAI_API_KEY": "qwen-reverse",
            "OPENAI_MODEL": "qwen3.8-max",
        },
    },
    "qwen": {
        "cmd": "qwen",
        "name": "Qwen Code",
        "format": "openai",
        "install": "pip install qwen-code",
        "env": {
            "OPENAI_BASE_URL": "{base_url}",
            "OPENAI_API_KEY": "qwen-reverse",
            "OPENAI_MODEL": "qwen3.8-max",
        },
    },
    "opencode": {
        "cmd": "opencode-cli",
        "name": "OpenCode CLI",
        "format": "openai",
        "install": "npm install -g opencode-ai",
        "env": {
            "OPENAI_BASE_URL": "{base_url}/v1",
            "OPENAI_API_BASE": "{base_url}/v1",
            "OPENAI_API_KEY": "qwen-reverse",
            "OPENAI_MODEL": "qwen3.8-max",
        },
    },
    "aider": {
        "cmd": "aider",
        "name": "Aider",
        "format": "openai",
        "install": "pip install aider-chat",
        "env": {
            "OPENAI_API_BASE": "{base_url}/v1",
            "OPENAI_API_KEY": "qwen-reverse",
            "OPENAI_MODEL": "qwen3.8-max",
        },
    },
}


def find_free_port(start: int = 8000, end: int = 8100) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{end}")


def wait_for_server(url: str, timeout: int = 15) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def run_uvicorn(port: int, host: str):
    import sys
    import os
    import uvicorn
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_dir = os.path.join(repo_root, "qwen_reverse", "server")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    uvicorn.run(
        "qwen_reverse.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


def find_agent_cmd(name: str) -> str | None:
    """Find the agent executable on PATH (handles .cmd/.bat on Windows)."""
    cmd = AGENTS[name]["cmd"]
    # On Windows, shutil.which finds .cmd/.bat/.exe automatically
    path = shutil.which(cmd)
    return path


def detect_available_agents() -> list[str]:
    """Return list of agent keys that are installed and available."""
    available = []
    for key in AGENTS:
        if find_agent_cmd(key):
            available.append(key)
    return available


def _map_to_agent_model(agent_key: str, model_override: str | None) -> str | None:
    if not model_override:
        return None
    if agent_key == "claude":
        if model_override.startswith("claude-"):
            return model_override
        
        # Parse suffixes from qwen model name
        base, chat_type, effort = _parse_model_suffix(model_name=model_override)
        suffix = ""
        if chat_type == "search":
            suffix += "-search"
        elif chat_type == "web_dev":
            suffix += "-web-dev"
        elif chat_type == "deep_research":
            suffix += "-deep-research"
        elif chat_type == "artifacts":
            suffix += "-artifacts"
        elif chat_type == "slides":
            suffix += "-slides"
            
        if effort == "high":
            suffix += "-thinking"
            
        return f"claude-opus-4-6{suffix}"
    return model_override


def build_env(base_url: str, agent_key: str, model_override: str | None = None) -> dict:
    """Build env vars for a specific agent, filling in the base_url and model override."""
    env = os.environ.copy()
    agent = AGENTS[agent_key]
    
    if agent_key == "claude":
        env["CLAUDE_CODE_USE_BEDROCK"] = "0"
        env["CLAUDE_CODE_USE_VERTEX"] = "0"
        env["AWS_PROFILE"] = ""
        env["AWS_ACCESS_KEY_ID"] = ""
        env["AWS_SECRET_ACCESS_KEY"] = ""
        env["AWS_SESSION_TOKEN"] = ""
        env["AWS_REGION"] = ""

    for key, val in agent["env"].items():
        env[key] = val.format(base_url=base_url) if "{base_url}" in val else val

    if model_override:
        mapped_model = _map_to_agent_model(agent_key, model_override)
        if agent_key == "claude":
            env["ANTHROPIC_MODEL"] = mapped_model
        else:
            env["OPENAI_MODEL"] = mapped_model
            env["OPENAI_API_MODEL"] = mapped_model

    return env


def launch_agent(base_url: str, agent_key: str, model_override: str | None = None):
    """Launch an AI coding agent with the right env vars."""
    agent = AGENTS[agent_key]
    cmd_path = find_agent_cmd(agent_key)
    env = build_env(base_url, agent_key, model_override=model_override)

    print(f"\n[{agent_key}] Launching {agent['name']} with base_url={base_url}\n")

    try:
        proc = subprocess.Popen([cmd_path], env=env)
        proc.wait()
    except FileNotFoundError:
        print(f"[{agent_key}] '{agent['cmd']}' not found. Install with:")
        print(f"  {agent['install']}")
        print("\nOr set env vars manually:")
        for k, v in agent["env"].items():
            print(f"  {k}={v.format(base_url=base_url)}")
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass


def interactive_choose(available: list[str]) -> str | None:
    """Let the user choose an agent interactively."""
    if not available:
        return None

    print("\nDetected agents:")
    for i, key in enumerate(available, 1):
        agent = AGENTS[key]
        print(f"  [{i}] {agent['name']:<16} ({agent['format']})")

    print("  [0] Just start server (no agent)")

    try:
        choice = input("\nChoose an agent to launch: ").strip()
        idx = int(choice)
        if idx == 0:
            return None
        if 1 <= idx <= len(available):
            return available[idx - 1]
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="qwen-reverse",
        description="Start the qwen-reverse API server (OpenAI + Anthropic compatible)",
    )
    parser.add_argument("--port", type=int, default=0, help="Port (0 = auto-find)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    # Agent shortcuts
    for key, agent in AGENTS.items():
        parser.add_argument(
            f"--{key}",
            action="store_true",
            help=f"Launch {agent['name']} after starting server",
        )
    parser.add_argument("--openai", action="store_true", help="Show OpenAI-compatible setup")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--no-server", action="store_true", help="Don't start server (just launch agent)")
    args = parser.parse_args()

    # Find which agent was requested via flags
    requested_agent = None
    for key in AGENTS:
        if getattr(args, key, False):
            requested_agent = key
            break

    if args.no_server:
        # Just launch agent with existing server
        base_url = f"http://{args.host}:{args.port}" if args.port else "http://127.0.0.1:8090"
        if requested_agent:
            launch_agent(base_url, requested_agent, model_override=args.model)
        else:
            available = detect_available_agents()
            chosen = interactive_choose(available)
            if chosen:
                launch_agent(base_url, chosen, model_override=args.model)
        return

    port = args.port or find_free_port()
    host = args.host
    base_url = f"http://{host}:{port}"

    print(f"""
=========================================
 qwen-reverse API server
 Base URL:  {base_url}
 Model:     qwen3.8-max
=========================================
""")

    print("Endpoints:")
    print(f"  OpenAI:       {base_url}/v1/chat/completions")
    print(f"  Anthropic:    {base_url}/v1/messages")
    print(f"  Models list:  {base_url}/v1/models")
    print()

    if args.openai:
        print("OpenAI setup:")
        print(f"  OPENAI_BASE_URL={base_url}")
        print("  OPENAI_API_KEY=qwen-reverse")
        print("  OPENAI_MODEL=qwen3.8-max")
        print()

    # If no agent requested via flag, detect and ask
    if not requested_agent:
        available = detect_available_agents()
        if available:
            chosen = interactive_choose(available)
            if chosen:
                requested_agent = chosen
            else:
                requested_agent = None  # just server

    if requested_agent:
        agent = AGENTS[requested_agent]
        print(f"{agent['name']} will launch automatically after server starts.\n")
        # Start server in a background thread
        server_thread = threading.Thread(
            target=run_uvicorn, args=(port, host), daemon=True
        )
        server_thread.start()

        # Wait for server to be ready
        if wait_for_server(f"{base_url}/health"):
            print(f"[server] Ready at {base_url}\n")
            launch_agent(base_url, requested_agent, model_override=args.model)
        else:
            print("[server] Failed to start within 10s, check logs above.")
        return

    # Default: just run the server
    try:
        run_uvicorn(port, host)
    except KeyboardInterrupt:
        print("\n[server] Stopped.")


if __name__ == "__main__":
    main()
