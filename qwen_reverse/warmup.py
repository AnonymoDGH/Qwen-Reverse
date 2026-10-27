"""One-time browser warm-up for the persistent Qwen profile.

A brand-new Chrome profile has no WAF history, so Aliyun can rate-limit it
(``F011``) after a single imperfect captcha interaction and reject even
perfect manual solves.  Run this once:

    python -m qwen_reverse.warmup

It opens Chrome **without** the DevTools port, so you can log in / solve the
captcha like a normal user.  The profile is the same one the SDK uses
(``~/.qwen/browser-profile``), so the resulting ``x5sec`` / ``acw_*`` cookies
persist and automated requests start working.
"""

from __future__ import annotations

import os
import subprocess
import sys

from .cdp import _chrome_candidates


def profile_dir() -> str:
    return os.environ.get(
        "QWEN_BROWSER_PROFILE",
        os.path.join(os.path.expanduser("~"), ".qwen", "browser-profile"),
    )


def main() -> int:
    exe = os.environ.get("QWEN_CHROME") or next(iter(_chrome_candidates()), None)
    if not exe:
        print("Chrome/Chromium not found. Set QWEN_CHROME=/path/to/chrome")
        return 1

    profile = profile_dir()
    os.makedirs(profile, exist_ok=True)
    print(f"Profile: {profile}")
    print("Opening Chrome without remote debugging...")
    print("1) Log in to chat.qwen.ai (recommended) or solve the captcha once.")
    print("2) Send one message successfully.")
    print("3) Close Chrome and return here.")

    proc = subprocess.Popen(
        [exe, f"--user-data-dir={profile}", "--no-first-run",
         "--no-default-browser-check", "https://chat.qwen.ai/"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        input("\nPress Enter once Chrome is closed (or Ctrl+C to keep it open)... ")
    except KeyboardInterrupt:
        print("\nLeft Chrome running.")
        return 130
    if proc.poll() is None:
        print("Note: Chrome is still running. Close it before using browser=True.")
    else:
        print("Done. The profile is warm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
