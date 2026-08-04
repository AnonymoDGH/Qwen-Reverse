#!/usr/bin/env python
"""Start the qwen-reverse API server and launch AI coding agents in a single command.

Usage:
    python run.py                                    # starts server
    python run.py --claude                          # starts server + launches Claude Code
    python run.py --claude --model qwen3.8-max-thinking  # single command with model
"""

import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "server"))

from qwen_reverse.cli import main

if __name__ == "__main__":
    main()
