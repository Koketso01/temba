"""temba.demo -- a tiny synthetic survey for checking an installation.

``temba demo <dir>`` writes a 160 x 160 x 96 cube of Gaussian noise with two
"catalogued" sources, their mask, a SoFiA-2 parameter file and a survey file
that injects 12 galaxies in one realisation. The whole run takes a minute or
two, and exercises every step: substrate, population, 3D-Barolo models,
injection, SoFiA-2, matching and fitting.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

PAR = """# SoFiA-2 settings for the TEMBA demo (TEMBA rewrites the paths)
input.data          = demo_cube.fits
scaleNoise.enable   = false
scfind.enable       = true
scfind.kernelsXY    = 0, 3, 6
scfind.kernelsZ     = 0, 3, 7
scfind.threshold    = 4.0
linker.radiusXY     = 2
linker.radiusZ      = 2
linker.minSizeXY    = 5
linker.minSizeZ     = 3
reliability.enable  = false
dilation.enable     = false
parameter.enable    = true
parameter.wcs       = true
parameter.physical  = true
output.filename     = demo
output.writeCatXML  = true
output.writeMask    = true
"""

SURVEY = """# TEMBA demo survey: one realisation of 12 galaxies in a small synthetic cube.
survey: demo
workdir: work
tools:
{tools}injection:
  sources_per_realisation: 12
  realisations: 1
  stamps: 2
run:
  null_run: false
  cleanup: false
analysis:
  bootstrap: 20
fields:
  - name: Demo
    cube: demo_cube.fits
    mask: demo_mask.fits
    sofia_par: demo_sofia.par
    z: 0.02
"""


def write(outdir, seed=1, sofia_exe=None, tools=None):
    """Write the demo cube, mask, SoFiA-2 parameters and survey file into outdir."""
    from astropy.io import fits
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nz, ny, nx = 96, 160, 160
    h = fits.Header()
    for i, (ct, crv, cd, crp, cu) in enumerate((
            ("RA---SIN", 150.0, -6 / 3600, nx / 2, "deg"),
            ("DEC--SIN", -30.0, 6 / 3600, ny / 2, "deg"),
            ("VRAD", 4.0e6, 44100.0, 1, "m/s")), 1):
        h[f"CTYPE{i}"], h[f"CRVAL{i}"], h[f"CDELT{i}"], h[f"CRPIX{i}"], h[f"CUNIT{i}"] = ct, crv, cd, crp, cu
    h["BMAJ"], h["BMIN"], h["BPA"] = 30 / 3600, 30 / 3600, 0.0
    h["BUNIT"], h["SPECSYS"], h["RESTFRQ"] = "JY/BEAM", "BARYCENT", 1.420405752e9
    sig = 3e-4
    cube = rng.normal(0, sig, (nz, ny, nx)).astype("f4")
    mask = np.zeros((nz, ny, nx), "i4")
    yy, xx = np.mgrid[:ny, :nx]
    for k, (y, x, z) in enumerate(((40, 50, 30), (110, 120, 60)), 1):
        blob = np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / 4.0 ** 2)
        for dz in range(-3, 4):
            cube[z + dz] += (6 * sig * blob).astype("f4")
            mask[z + dz][blob > 0.05] = k
    fits.PrimaryHDU(cube, h).writeto(out / "demo_cube.fits", overwrite=True)
    fits.PrimaryHDU(mask, h).writeto(out / "demo_mask.fits", overwrite=True)
    (out / "demo_sofia.par").write_text(PAR)
    import yaml
    t = dict(tools or {})
    t["sofia"] = find_sofia_exe(sofia_exe or t.get("sofia")) or t.get("sofia") or "sofia"
    t.setdefault("ld_library_path", conda_libs())
    t = {k: v for k, v in t.items() if v}
    block = "".join("  " + line + "\n" for line in yaml.safe_dump(t, sort_keys=False).splitlines())
    (out / "survey.yaml").write_text(SURVEY.format(tools=block))
    return out / "survey.yaml"


def find_sofia_exe(given=None):
    """SoFiA-2: as given, on PATH, or where `temba install-sofia` and common builds put it."""
    import os
    for c in (given, shutil.which("sofia"),
              os.path.expanduser("~/.temba/SoFiA-2/sofia"), os.path.expanduser("~/SoFiA-2/sofia")):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(Path(c).resolve())
    return None


def conda_libs():
    """The active conda environment's lib/ (where conda-forge puts wcslib for SoFiA-2)."""
    import os
    prefix = os.environ.get("CONDA_PREFIX")
    return str(Path(prefix) / "lib") if prefix and (Path(prefix) / "lib").is_dir() else None
