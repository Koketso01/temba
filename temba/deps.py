"""temba.deps -- check, and where possible install, what TEMBA depends on.

Python packages come with ``pip install``. Two scientific programs are
external: SoFiA-2 (a C program) and 3D-Barolo, used through its Python
interface pyBBarolo. ``temba check`` reports all of them;
``temba install-sofia`` downloads and compiles SoFiA-2.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY_PACKAGES = ("numpy", "scipy", "astropy", "matplotlib", "yaml")
# SoFiA-2 is developed on GitLab; its GitHub repository holds only a pointer there
SOFIA_REPOS = ("https://gitlab.com/SoFiA-Admin/SoFiA-2.git",)


def python_packages():
    out = []
    for name in PY_PACKAGES:
        try:
            m = importlib.import_module(name)
            out.append((name, True, getattr(m, "__version__", "")))
        except Exception as e:
            out.append((name, False, str(e)))
    return out


def find_sofia(spec="sofia"):
    """Path of the SoFiA-2 executable, from a path or a name on PATH."""
    if spec and Path(spec).expanduser().is_file():
        return str(Path(spec).expanduser())
    return shutil.which(spec or "sofia")


def sofia_version(exe, env=None):
    """SoFiA-2 prints its banner (with version) when run without arguments."""
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=30, env=env)
        for line in (r.stdout + r.stderr).splitlines():
            if "SoFiA" in line and any(ch.isdigit() for ch in line):
                return line.strip()
        return "found"
    except Exception as e:
        return f"could not run: {e}"


def pybbarolo(pythonpath=None):
    """Whether pyBBarolo (with SimulatedGalaxyCube) imports, optionally from a source tree."""
    if pythonpath and pythonpath not in sys.path:
        sys.path.insert(0, pythonpath)
    try:
        from pyBBarolo.utils import SimulatedGalaxyCube  # noqa: F401
        import pyBBarolo
        return True, getattr(pyBBarolo, "__version__", "found")
    except Exception as e:
        return False, str(e)


def env_with(tools):
    e = os.environ.copy()
    for key, var in (("ld_library_path", "LD_LIBRARY_PATH"), ("pythonpath", "PYTHONPATH")):
        v = (tools or {}).get(key)
        if v:
            e[var] = str(v) + (":" + e[var] if e.get(var) else "")
    return e


def report(tools):
    """[(component, ok, detail)] for everything TEMBA needs."""
    rows = [(f"python: {n}", ok, d) for n, ok, d in python_packages()]
    exe = find_sofia(tools.get("sofia"))
    rows.append(("SoFiA-2", bool(exe), sofia_version(exe, env_with(tools)) if exe else
                 "not found: install it with `temba install-sofia`, or set tools.sofia"))
    ok, detail = pybbarolo(tools.get("pythonpath"))
    rows.append(("pyBBarolo (3D-Barolo)", ok, detail if ok else
                 "not importable: `pip install pyBBarolo`, or set tools.pythonpath"))
    return rows


def install_sofia(prefix, openmp=True):
    """Clone and compile SoFiA-2 into <prefix>/SoFiA-2; return the executable path.

    SoFiA-2 needs a C compiler and wcslib (``conda install -c conda-forge wcslib``
    or the system package). Its own compile.sh does the build.
    """
    prefix = Path(prefix).expanduser().resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    src = prefix / "SoFiA-2"
    if not (src / "compile.sh").exists():
        if src.exists():
            import shutil as _sh
            _sh.rmtree(src)
        for url in SOFIA_REPOS:
            r = subprocess.run(["git", "clone", "--depth", "1", url, str(src)])
            if r.returncode == 0 and (src / "compile.sh").exists():
                break
        else:
            raise SystemExit("could not get SoFiA-2 from GitLab "
                             "(https://gitlab.com/SoFiA-Admin/SoFiA-2); install it by hand "
                             "and set tools.sofia")
    cmd = ["sh", "compile.sh"] + (["-fopenmp"] if openmp else [])
    r = subprocess.run(cmd, cwd=src)
    exe = src / "sofia"
    if r.returncode != 0 or not exe.exists():
        raise SystemExit("SoFiA-2 did not compile; it needs a C compiler and wcslib "
                         "(conda install -c conda-forge wcslib)")
    return str(exe)
