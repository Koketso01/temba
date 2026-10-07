"""The pipeline's own unit tests (temba.tests), run under pytest."""
import subprocess
import sys


def test_temba_unit_suite(tmp_path):
    r = subprocess.run([sys.executable, "-m", "temba.tests"], cwd=tmp_path,
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert " 0 failed" in r.stdout
