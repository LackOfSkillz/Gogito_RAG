"""
Cogito GUI Launcher (v0.7.x)
----------------------------
- Verifies LM Studio (OpenAI-compatible) at /v1/models
- Verifies/installs split packages: langchain-openai, langchain-chroma
- Starts Streamlit on a fixed port (default 8501)
- Waits for readiness, then opens the browser ONCE (no tab loop)
- Writes launch timing to logs/cogito_timing.log

Usage:
  python run_cogito_gui.py
  python run_cogito_gui.py --port 8501 --no-browser
  python run_cogito_gui.py --install-deps
"""

from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except Exception:
    print("requests is required for launcher. Installing now...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "requests"], check=False)
    import requests  # retry

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "settings.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "cogito_timing.log"

DEFAULTS = {
    "lm_studio_url": "http://localhost:1234",
    "default_model": "",
    "temperature": 0.3,
    "top_k_max": 16,
    "max_tokens": 768,
    "streaming": False,
}

def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {msg}"
    print(line)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        s = DEFAULTS.copy()
        s.update(data or {})
        return s
    except Exception:
        return DEFAULTS.copy()

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0

def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return 200 <= r.status_code < 500  # Streamlit root can redirect; treat as OK
    except Exception:
        return False

def wait_until_ready(url: str, total_timeout: float = 90.0, step: float = 1.5) -> bool:
    t0 = time.time()
    while time.time() - t0 < total_timeout:
        if http_ok(url):
            return True
        time.sleep(step)
    return False

def ensure_deps(install: bool = False) -> None:
    """
    Make sure split packages exist; optionally auto-install them.
    """
    missing = []
    try:
        import langchain_openai  # noqa: F401
    except Exception:
        missing.append("langchain-openai")
    try:
        import langchain_chroma  # noqa: F401
    except Exception:
        missing.append("langchain-chroma")

    if not missing:
        return

    if install:
        log(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", *missing], check=False)
        return

    log("⚠️ Missing packages detected: " + ", ".join(missing))
    log("   Run again with --install-deps to auto-install, or:")
    log(f"   {sys.executable} -m pip install -U " + " ".join(missing))

def check_lm_studio(base_url: str) -> Tuple[bool, Optional[float]]:
    """
    Probe LM Studio's /v1/models; return (ok, latency_seconds or None).
    """
    url = base_url.rstrip("/") + "/v1/models"
    t0 = time.time()
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        _ = [m.get("id") for m in (data.get("data") or [])]
        return True, round(time.time() - t0, 2)
    except Exception as e:
        log(f"❌ LM Studio check failed at {url}: {e}")
        return False, None

def start_streamlit(port: int, no_browser: bool = True) -> subprocess.Popen:
    """
    Start Streamlit pointing to cogito_gui.py without auto-browser (we open once ourselves).
    """
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"
    cmd = [
        "streamlit", "run", str(ROOT / "cogito_gui.py"),
        "--server.port", str(port),
        "--server.headless", "true" if no_browser else "false",
    ]
    log("🚀 Starting Streamlit server…")
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )

def tail_until_ready(proc: subprocess.Popen, url: str, soft_timeout: float = 60.0) -> float:
    """
    Consume some logs while waiting for readiness, returning latency seconds.
    """
    t0 = time.time()
    ready = False
    while True:
        if proc.poll() is not None:
            log("❌ Streamlit process exited early.")
            break
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            # echo a few key lines to console log
            if "You can now view your Streamlit app" in line or "Running on local URL" in line:
                ready = True
        if ready and http_ok(url):
            break
        if time.time() - t0 > soft_timeout:
            break
    return round(time.time() - t0, 2)

def main():
    parser = argparse.ArgumentParser(description="Cogito GUI Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the Streamlit server on")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    parser.add_argument("--install-deps", action="store_true", help="Auto-install missing split packages")
    args = parser.parse_args()

    settings = load_settings()
    lm_url = settings.get("lm_studio_url", "http://localhost:1234").rstrip("/")
    port = int(args.port)
    url = f"http://localhost:{port}"

    log("\n🧠 Launching Cogito Knowledge Assistant…\n")

    # 1) Ensure split deps (langchain-openai, langchain-chroma)
    ensure_deps(install=args.install_deps)

    # 2) Check Streamlit availability
    try:
        subprocess.run(["streamlit", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        log("❌ Streamlit not found on PATH. Install with: pip install -U streamlit")
        sys.exit(1)

    # 3) Check LM Studio
    ok, lm_latency = check_lm_studio(lm_url)
    if ok:
        log(f"✅ LM Studio ready at {lm_url} (models endpoint OK, {lm_latency}s)")
    else:
        log("⚠️ LM Studio not reachable. You can still start the GUI, but chat will fail until LM Studio is up.")

    # 4) If port already in use, assume Streamlit is running; just open or print URL
    if is_port_in_use(port):
        log(f"ℹ️ Port {port} already in use — assuming Streamlit is running.")
        if not args.no_browser:
            webbrowser.open(url)
        print(f"\n✅ Cogito is (likely) running at: {url}\n")
        return

    # 5) Start Streamlit (headless) and wait for readiness
    proc = start_streamlit(port=port, no_browser=True)
    st_latency = tail_until_ready(proc, url, soft_timeout=60.0)

    # Fallback: explicit HTTP poll if logs didn’t show readiness
    if not http_ok(url):
        log("⏳ Waiting for HTTP readiness…")
        if not wait_until_ready(url, total_timeout=90.0, step=1.5):
            log("⚠️ Streamlit readiness not confirmed within timeout. You can still try the URL.")

    # 6) Open browser once (unless suppressed)
    if not args.no_browser:
        log("🌐 Opening browser…")
        webbrowser.open(url)
        log(f"✅ Browser opened at {url}")

    # 7) Print summary
    log("\n--- Launch Summary ---")
    if lm_latency is not None:
        log(f"LM Studio check: {lm_latency}s")
    log(f"Streamlit start: {st_latency}s")
    log(f"Dashboard URL : {url}")
    log("----------------------\n")

    print("\n✅ Cogito launcher done. Close this window to stop the server.\n")

if __name__ == "__main__":
    main()
