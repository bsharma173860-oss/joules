"""CLI regression: common slips get a clear message and exit code, not a Python traceback.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ["JOULES_NO_PROMPT"] = "1"

from joules import cli  # noqa: E402

SIM = {"JOULES_SIMULATE": "gpu:H100 SXM:300", "DO_NOT_TRACK": "1", "JOULES_NO_TRACK": "1"}
MISSING = "joules_no_such_command_xyz"


def run_cli(argv, cwd=None):
    err, out = io.StringIO(), io.StringIO()
    old = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        with mock.patch.dict(os.environ, SIM), contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            try:
                code = cli.main(argv)
            except SystemExit as e:  # argparse errors
                code = e.code
    finally:
        os.chdir(old)
    return code, out.getvalue(), err.getvalue()


class CliInputErrors(unittest.TestCase):
    def test_compare_command_not_found(self):
        code, out, err = run_cli(["compare", MISSING, "true", "--runs", "2", "--warmup", "0"])
        self.assertEqual(code, 127)
        self.assertIn(f"command not found: {MISSING}", err)
        self.assertNotIn("Traceback", err)

    def test_ci_command_not_found(self):
        with tempfile.TemporaryDirectory() as t:
            code, out, err = run_cli(["ci", "--runs", "1", "--warmup", "0", "--", MISSING], cwd=t)
        self.assertEqual(code, 127)
        self.assertIn(f"command not found: {MISSING}", err)

    def test_bad_gpus(self):
        for bad in ("a", "0,", "1,x", "-1"):
            code, out, err = run_cli(["run", "--gpus", bad, "--", sys.executable, "-c", "pass"])
            self.assertEqual(code, 2, bad)
            self.assertIn("argument --gpus", err)
            self.assertNotIn("Traceback", err)
        self.assertEqual(cli._gpu_list(cli._gpus_arg(" 0, 1 ")), [0, 1])

    def test_interval_must_be_positive(self):
        for bad in ("0", "-1", "nan", "abc"):
            code, out, err = run_cli(["run", "--interval", bad, "--", sys.executable, "-c", "pass"])
            self.assertEqual(code, 2, bad)
            self.assertIn("argument --interval", err)
        self.assertEqual(cli._interval_arg("0.05"), 0.05)

    def test_one_run_compare_has_no_empty_parentheses(self):
        code, out, err = run_cli(["compare", f"{sys.executable} -c pass", f"{sys.executable} -c pass", "--runs", "1", "--warmup", "0"])
        self.assertEqual(code, 0, err)
        line = next(ln for ln in err.splitlines() if " vs " in ln)
        self.assertNotIn("()", line)
        self.assertNotIn("1 runs", err)
        from joules import compare
        r = {"runs": 1, "order": "alternating", "tool": "joules 0", "devices": ["x"], "names": [], "commands": [], "blocks": [], "results": [], "notes": []}
        self.assertIn("(1 run each,", compare.markdown(r).splitlines()[0])


if __name__ == "__main__":
    unittest.main()
