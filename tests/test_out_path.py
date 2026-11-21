"""--out regression: a path that cannot be written must never lose the result of a measured run.
Run: PYTHONPATH=src python3 -m unittest discover -s tests -v"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["JOULES_NO_PROMPT"] = "1"

from joules import cli  # noqa: E402

SIM = {"JOULES_SIMULATE": "gpu:H100 SXM:300", "DO_NOT_TRACK": "1", "JOULES_NO_TRACK": "1"}


def run_cli(argv):
    err, out = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, SIM), contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class OutPath(unittest.TestCase):
    def test_missing_folder_is_refused_before_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            marker = Path(t) / "ran"
            code, out, err = run_cli(["run", "--out", str(Path(t) / "nope" / "x.json"), "--",
                                      sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
            self.assertEqual(code, 2)
            self.assertIn("cannot write --out", err)
            self.assertIn("does not exist", err)
            self.assertNotIn("Traceback", err)
            self.assertFalse(marker.exists())  # nothing was run (and measured) for nothing

    def test_write_failing_after_the_run_keeps_the_printed_result(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t) / "results"
            folder.mkdir()
            code, out, err = run_cli(["run", "--json", "--out", str(folder / "x.json"), "--",
                                      sys.executable, "-c", f"import os, time; time.sleep(0.2); os.rmdir({str(folder)!r})"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["kind"], "run")  # the result was printed
            self.assertIn("could not write", err)
            self.assertNotIn("Traceback", err)

    def test_good_path_still_written(self):
        with tempfile.TemporaryDirectory() as t:
            code, out, err = run_cli(["run", "--out", str(Path(t) / "x.json"), "--", sys.executable, "-c", "pass"])
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads((Path(t) / "x.json").read_text())["kind"], "run")


if __name__ == "__main__":
    unittest.main()
