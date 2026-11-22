"""joules run --from/--until: the window's line matching against a plain reference, over random patterns, texts and
ways the output is split into reads. The reference cleans the whole output, splits it into lines and runs re.search
on each line; joules must find the same --from line and the same --until line after it, whatever the split.
Fixed seeds: a failure is repeatable (the message has the case)."""
import random
import re
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from joules import cli  # noqa: E402

ATOMS = ["a", "b", "A", "B", "k", "K", "s", "S", "i", "I", "ı", "İ", "ſ", "K", "x", " ", "\\n", "\\r", "\\x0b",
         "\\x85", ".", "\\w", "\\W", "\\d", "\\D", "\\s", "\\S", "\\b", "\\B", "^", "$", "\\A", "\\Z", "[a-c]", "[^a]",
         "[\\x00-\\x7f]", "[\\w]", "[A-Z]", "[i]", "[s]", "\\1", "ab", "ss", "ki", " ", "\\u2028", "[\\x0b-\\x0d]",
         "START", "END"]
TEXT = ["START", "END", "start", "end", "\x1b[31m", "\x1b[0m", "\r\n", "\r", "\n", "\n", "x", " ", "ı", "İ", "ſ",
        "K", "\x85", "é", "1", "a", "b", "s", "S", "i", "I", "\x0b", "\x0c", "\x1c", " ", "̇", "\t", "ab", "ss"]


class Clock:
    def __init__(self):
        self.n = 0

    def snapshot(self):
        self.n += 1
        return {"t": float(self.n), "j": {}}


def pattern(r: random.Random, d: int = 0) -> str:
    out = []
    for _ in range(r.randint(1, 4)):
        x = r.random()
        if d < 3 and x < 0.12:
            a = "(" + pattern(r, d + 1) + ")"
        elif d < 3 and x < 0.17:
            a = "(?:" + pattern(r, d + 1) + "|" + pattern(r, d + 1) + ")"
        elif d < 3 and x < 0.2:
            a = r.choice(["(?=", "(?!", "(?<=", "(?<!", "(?>", "(?i:", "(?-i:", "(?m:", "(?s:"]) + pattern(r, d + 1) + ")"
        elif d < 3 and x < 0.22:
            a = "(?(1)" + pattern(r, d + 1) + "|" + pattern(r, d + 1) + ")"
        else:
            a = r.choice(ATOMS)
        if r.random() < 0.25:
            a += r.choice(["*", "+", "?", "{2}", "{0,2}", "*?", "++", "*+", "{1,}"])
        out.append(a)
    s = "".join(out)
    if d == 0 and r.random() < 0.2:
        s += "|" + pattern(r, 1)
    if d == 0 and r.random() < 0.15:
        s = r.choice([".*", ".+", "^.*", "^.+", ".*?"]) + s
    if d == 0 and r.random() < 0.3:
        s = r.choice(["(?i)", "(?a)", "(?x)", "(?ia)", "(?s)", "(?m)"]) + s
    return s


def reference(data: bytes, rf, ru):
    full = cli.Window.clean(data.decode("utf-8", "replace"))
    lines = full.split("\n")
    if lines and lines[-1] == "":   # (no last line: the output ended with a line end, or with escape codes only)
        lines.pop()
    b = e = None
    for i, ln in enumerate(lines):
        if b is None:
            if rf.search(ln):
                b = i
        elif ru.search(ln):
            e = i
            break
    return (lines[b].strip()[:200] if b is not None else None, lines[e].strip()[:200] if e is not None else None)


def run_joules(data: bytes, pf: str, pu: str, r: random.Random):
    w = cli.Window(Clock(), 0.0, [], pf, pu)
    for rx in list(getattr(w, "block", {})):
        w.block_size[rx] = r.choice([1, 2, 256])   # (tiny pieces: block searches cut at every edge)
    lines = cli._Lines()
    pos = 0
    while pos < len(data):
        n = r.randint(1, 12)
        done = lines.add(data[pos:pos + n], w, 1)
        if done is not None and w.pending():
            w.feed([done], [1])
        pos += n
    lines.end(w, 1)
    return (w.begin["line"] if w.begin else None, w.end["line"] if w.end else None)


class MatchingFuzz(unittest.TestCase):
    def check(self, seed: int, n: int):
        warnings.simplefilter("ignore")
        r = random.Random(seed)
        tried = 0
        for _ in range(n):
            pf = pattern(r) if r.random() < 0.7 else r.choice(["START", "(?i)start", "^START$", "S.*T", "\\bSTART\\b"])
            pu = pattern(r) if r.random() < 0.7 else r.choice(["END", "(?i)end", "^END$", "E.D", "END$"])
            try:
                rf, ru = re.compile(pf), re.compile(pu)
                cli.Window(Clock(), 0.0, [], pf, pu)
            except Exception:
                continue
            data = "".join(r.choice(TEXT) for _ in range(r.randint(0, 80))).encode()
            if r.random() < 0.2:
                data += b"\xff\xfe"
            tried += 1
            got, want = run_joules(data, pf, pu, r), reference(data, rf, ru)
            self.assertEqual(got, want, f"--from {pf!r} --until {pu!r} output {data!r}")
        self.assertGreater(tried, n // 3)

    def test_seed_1(self):
        self.check(1, 15000)

    def test_seed_2(self):
        self.check(2, 15000)

    def test_seed_3(self):
        self.check(3, 15000)


if __name__ == "__main__":
    unittest.main()
