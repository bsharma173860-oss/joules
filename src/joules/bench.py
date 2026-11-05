"""Benchmarks a local model server and counts the tokens it produced.

Targets:
  ollama:<model>                  Ollama on http://localhost:11434 (or --url)
  openai:<model>                  any OpenAI-compatible server: vLLM, llama.cpp, LM Studio, TGI
                                  (default --url http://localhost:8000/v1)
"""
from __future__ import annotations

import http.client
import json
import os
import re
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_PROMPT = ("Explain, in about 200 words, why data centres measure power usage effectiveness "
                  "and what a good value looks like.")


@dataclass
class Target:
    kind: str     # ollama | openai
    model: str
    url: str


def parse_target(spec: str, url: str | None) -> Target:
    kind, sep, model = spec.partition(":")
    if not sep or kind not in ("ollama", "openai") or not model:
        raise ValueError("target must be ollama:<model> or openai:<model>, e.g. ollama:llama3.1:8b")
    default = "http://localhost:11434" if kind == "ollama" else "http://localhost:8000/v1"
    return Target(kind, model, (url or default).rstrip("/"))


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} answered HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url} ({e.reason}). Is the model server running?") from None
    except (OSError, http.client.HTTPException) as e:  # timeouts, dropped connections
        raise RuntimeError(f"{url} did not answer ({e or type(e).__name__}). Is the model server busy or stuck?") from None
    except ValueError:
        raise RuntimeError(f"{url} did not answer with JSON. Is that the right server address?") from None


_STARTED: list = []  # the Ollama servers this process started; they keep running after joules exits


def _ollama_app() -> str | None:
    """The Ollama app on a Mac (it runs the server in the background when opened)."""
    if platform.system() != "Darwin":
        return None
    for d in ("/Applications", os.path.expanduser("~/Applications")):
        if os.path.isdir(os.path.join(d, "Ollama.app")):
            return os.path.join(d, "Ollama.app")
    return None


def install_hint(then: str = ", then run this again") -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        how = "download it from https://ollama.com/download (or: brew install ollama)"
    elif sysname == "Windows":
        how = "download it from https://ollama.com/download (or: winget install Ollama.Ollama)"
    else:
        how = "run: curl -fsSL https://ollama.com/install.sh | sh"
    return f"Ollama is not installed. To install it, {how}{then}."


def _up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=2) as r:
            return r.status == 200
    except (OSError, ValueError, http.client.HTTPException):
        return False


def start_ollama(t: Target, log=None, wait: float = 45) -> bool:
    """Ollama installed but not running on this machine: start it, as the user would. False when it is not installed
    (or the server is on another machine)."""
    host = urlparse(t.url).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        return False
    app, cli = _ollama_app(), shutil.which("ollama")
    if not app and not cli:
        return False
    if log:
        log("starting Ollama…")
    if app:
        subprocess.run(["open", "-g", "-a", app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        env = dict(os.environ)
        port = urlparse(t.url).port
        if port and port != 11434:
            env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        kw = {"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" else {"start_new_session": True}  # detached
        _STARTED.append(subprocess.Popen([cli, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=env, **kw))
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if _up(t.url):
            return True
        time.sleep(0.5)
    raise RuntimeError(f"started Ollama, but it did not answer at {t.url} within {wait:.0f} s. Open the Ollama app "
                       "(or run: ollama serve), then run this again.")


def ensure_model(t: Target, log=None) -> bool:
    """Ollama: download the model first when the server does not have it. Returns True when it downloaded."""
    if t.kind != "ollama":
        return False
    for attempt in (1, 2):
        try:
            _post(f"{t.url}/api/show", {"model": t.model}, 30)
            return False
        except RuntimeError as e:
            if "cannot reach" in str(e):
                if attempt == 1 and start_ollama(t, log):
                    continue
                local = (urlparse(t.url).hostname or "") in ("localhost", "127.0.0.1", "::1")
                raise RuntimeError(install_hint() if local else f"Ollama is not running at {t.url}. Start it there "
                                   "(open the Ollama app, or run: ollama serve), then run this again.") from None
            if "HTTP 404" not in str(e):
                raise
            break
    if log:
        log(f"downloading {t.model} with Ollama (once; a few GB for most models)…")
    req = urllib.request.Request(f"{t.url}/api/pull", data=json.dumps({"model": t.model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = -1
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise RuntimeError(f"Ollama could not download {t.model}: {ev['error']}. Check the name at https://ollama.com/library")
                if log and ev.get("total") and ev.get("completed") is not None:
                    pct = int(100 * ev["completed"] / ev["total"])
                    if pct >= last + 10:
                        last = pct
                        log(f"  {ev.get('status', 'downloading')}: {pct}%")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama could not download {t.model} (HTTP {e.code}). Check the name at https://ollama.com/library") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"lost the connection to Ollama while downloading {t.model} ({e.reason})") from None
    except (OSError, http.client.HTTPException) as e:
        raise RuntimeError(f"lost the connection to Ollama while downloading {t.model} ({e or type(e).__name__}); run this again to resume") from None
    return True


def model_params(t: Target) -> float | None:
    """The model's size in billions of parameters, as Ollama reports it, so the Landauer gap needs no --params.
    None for other servers or when Ollama does not say."""
    if t.kind != "ollama":
        return None
    try:
        info = _post(f"{t.url}/api/show", {"model": t.model}, 30)
    except RuntimeError:
        return None
    n = (info.get("model_info") or {}).get("general.parameter_count")
    if isinstance(n, (int, float)) and n > 0:
        return n / 1e9
    m = re.fullmatch(r"([\d.]+)\s*([BM])", str((info.get("details") or {}).get("parameter_size") or "").strip(), re.I)
    return float(m.group(1)) * (1 if m.group(2).upper() == "B" else 1e-3) if m else None


def generate(t: Target, prompt: str, max_tokens: int, timeout: float = 600) -> dict:
    """One request. Returns output tokens, prompt tokens and the server-reported generation time."""
    if t.kind == "ollama":
        try:
            r = _post(f"{t.url}/api/generate", {"model": t.model, "prompt": prompt, "stream": False,
                                                "options": {"num_predict": max_tokens, "temperature": 0}}, timeout)
        except RuntimeError as e:
            if "cannot reach" in str(e):
                raise RuntimeError(f"Ollama is not running at {t.url}. Install it from https://ollama.com/download and open it "
                                   f"(or run: ollama serve), then download the model: ollama pull {t.model}") from None
            if "HTTP 404" in str(e) and "not found" in str(e):
                raise RuntimeError(f"Ollama does not have {t.model} yet. Download it first: ollama pull {t.model}") from None
            raise
        return {"output_tokens": int(r.get("eval_count", 0)), "prompt_tokens": int(r.get("prompt_eval_count", 0)),
                "gen_s": r.get("eval_duration", 0) / 1e9 or None}
    t0 = time.monotonic()
    r = _post(f"{t.url}/chat/completions", {"model": t.model, "messages": [{"role": "user", "content": prompt}],
                                            "max_tokens": max_tokens, "temperature": 0}, timeout)
    u = r.get("usage") or {}
    return {"output_tokens": int(u.get("completion_tokens", 0)), "prompt_tokens": int(u.get("prompt_tokens", 0)),
            "gen_s": time.monotonic() - t0}
