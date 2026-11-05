"""joules ci: an energy check for pull requests.

Runs a command on this checkout and on the base branch (in a temporary git worktree), several times,
alternating so drift hits both sides equally, and compares the medians. Writes a Markdown summary
(the GitHub job summary when it runs in Actions), can post or update one pull request comment, and
fails when energy grows by more than a budget.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .meters import Discovery
from .sampler import Sampler

MARKER = "<!-- landauer-gap-energy-check -->"


def measure(d: Discovery, cmd: list[str], cwd: str, interval: float) -> dict:
    with Sampler(d.meters, interval) as s:
        p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL)
        for m in d.meters:
            if hasattr(m, "attach"):
                m.attach(p)
        p.wait()
    return {"energy_j": sum(m.energy_j for m in d.meters), "duration_s": s.duration_s, "exit_code": p.returncode,
            "devices": {m.name: m.energy_j for m in d.meters}}


def git(*args: str, cwd: str | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout.strip()


@contextlib.contextmanager
def worktree(ref: str):
    """A clean checkout of ref next to this one, removed afterwards."""
    tmp = tempfile.mkdtemp(prefix="joules-base-")
    path = os.path.join(tmp, "base")
    git("-c", "advice.detachedHead=false", "worktree", "add", "--detach", path, ref)
    try:
        yield path
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", path], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def default_base() -> str | None:
    b = os.environ.get("GITHUB_BASE_REF")
    return f"origin/{b}" if b else None


def summarize(head: list[dict], base: list[dict] | None, budget: float | None) -> dict:
    h = statistics.median(r["energy_j"] for r in head)
    out = {"head_j": h, "head_s": statistics.median(r["duration_s"] for r in head),
           "head_range": [min(r["energy_j"] for r in head), max(r["energy_j"] for r in head)], "runs": len(head)}
    if base:
        b = statistics.median(r["energy_j"] for r in base)
        out.update(base_j=b, base_s=statistics.median(r["duration_s"] for r in base),
                   base_range=[min(r["energy_j"] for r in base), max(r["energy_j"] for r in base)])
        out["change_pct"] = (h - b) / b * 100 if b > 0 else None
        # The runs overlap: the difference is not bigger than the run-to-run spread.
        out["within_noise"] = out["head_range"][0] <= out["base_range"][1] and out["base_range"][0] <= out["head_range"][1]
        out["over_budget"] = bool(budget is not None and out["change_pct"] is not None and out["change_pct"] > budget
                                  and not out["within_noise"])
    out["budget_pct"] = budget
    return out


def _j(x: float) -> str:
    for unit, k in (("MJ", 1e6), ("kJ", 1e3)):
        if abs(x) >= k:
            return f"{x / k:.3g} {unit}"
    return f"{x:.3g} J"


def markdown(s: dict, *, head_name: str, base_name: str | None, devices: list[str], estimated: bool,
             command: str, track_link: str | None = None) -> str:
    lines = [MARKER, "### ⚡ Energy check", ""]
    if "base_j" in s:
        c = s["change_pct"]
        if c is None:
            verdict = "no comparison (the base used no energy)"
        elif s["over_budget"]:
            verdict = f"❌ **{c:+.1f}%**, over the {s['budget_pct']:g}% budget"
        elif s["within_noise"]:
            verdict = f"≈ **{c:+.1f}%**, within run-to-run noise"
        elif c > 0:
            verdict = f"🔺 **{c:+.1f}%** more energy"
        else:
            verdict = f"✅ **{c:+.1f}%**, less energy"
        lines += [f"{verdict}", "",
                  f"| | energy (median of {s['runs']}) | range | time |", "|---|---|---|---|",
                  f"| this change `{head_name}` | **{_j(s['head_j'])}** | {_j(s['head_range'][0])} – {_j(s['head_range'][1])} | {s['head_s']:.2f} s |",
                  f"| base `{base_name}` | {_j(s['base_j'])} | {_j(s['base_range'][0])} – {_j(s['base_range'][1])} | {s['base_s']:.2f} s |"]
    else:
        lines += [f"| | energy (median of {s['runs']}) | range | time |", "|---|---|---|---|",
                  f"| `{head_name}` | **{_j(s['head_j'])}** | {_j(s['head_range'][0])} – {_j(s['head_range'][1])} | {s['head_s']:.2f} s |"]
    lines += ["", f"Command: `{command.replace('`', chr(39))[:200]}`  ", f"Measured with: {', '.join(devices)}"]
    if estimated:
        lines.append("> Estimated from CPU time: this runner has no energy counters. Good for spotting changes; "
                     "run on a self-hosted runner for measured joules.")
    if track_link:
        lines.append(f"\n[History of this check on WattOps]({track_link})")
    lines.append("\n<sub>[WattOps](https://wattops.dev) · `joules ci`</sub>")
    return "\n".join(lines) + "\n"


def _api(method: str, url: str, token: str, body: dict | None = None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "joules-ci",
        "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode() or "null")


def post_comment(md: str, env: dict | None = None) -> str:
    """Creates or updates this check's comment on the pull request. Returns what happened."""
    env = env if env is not None else os.environ
    token, repo = env.get("GITHUB_TOKEN"), env.get("GITHUB_REPOSITORY")
    api = (env.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")
    try:
        with open(env.get("GITHUB_EVENT_PATH") or "") as f:
            pr = (json.load(f).get("pull_request") or {}).get("number")
    except (OSError, ValueError):
        pr = None
    if not (token and repo and pr):
        return "not a pull request (or no GITHUB_TOKEN): comment skipped"
    try:
        existing = None
        for page in range(1, 11):
            items = _api("GET", f"{api}/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}", token) or []
            existing = next((c for c in items if MARKER in (c.get("body") or "")), None)
            if existing or len(items) < 100:
                break
        if existing:
            _api("PATCH", f"{api}/repos/{repo}/issues/comments/{existing['id']}", token, {"body": md})
            return f"updated the energy comment on #{pr}"
        _api("POST", f"{api}/repos/{repo}/issues/{pr}/comments", token, {"body": md})
        return f"commented on #{pr}"
    except urllib.error.HTTPError as e:
        hint = " (pull requests from forks get a read-only token; give the job `permissions: pull-requests: write`)" if e.code in (403, 404) else ""
        return f"could not comment: HTTP {e.code}{hint}"
    except urllib.error.URLError as e:
        return f"could not comment: {e.reason}"
    except (OSError, http.client.HTTPException, ValueError) as e:
        return f"could not comment: {e or type(e).__name__}"


def write_outputs(s: dict, env: dict | None = None) -> None:
    env = env if env is not None else os.environ
    out = env.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"energy-j={s['head_j']:.6g}\n")
        if "base_j" in s:
            f.write(f"base-energy-j={s['base_j']:.6g}\n")
            f.write(f"change-pct={'' if s['change_pct'] is None else format(s['change_pct'], '.4g')}\n")
        f.write(f"over-budget={'true' if s.get('over_budget') else 'false'}\n")


def run_setup(cmd: str | None, cwd: str) -> None:
    if cmd:
        p = subprocess.run(cmd, shell=True, cwd=cwd)
        if p.returncode != 0:
            raise RuntimeError(f"setup failed in {cwd} (exit {p.returncode}): {cmd}")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
