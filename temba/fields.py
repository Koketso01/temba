"""
temba.fields -- discover the per-field inputs and write field configs.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

A campaign needs, for each field: the science cube, the SoFiA mask from the
science run, the science parameter file, and (for cross-checks) the science
catalogue and RMS cube. Those live in a few directories with names that are
nearly but not quite regular:

    Cluster-Cubes/Abell-194.fits          but  Abell_194_mask-raw.fits
    Cluster-Cubes/Abell-4038.fits         but  Abell_4038_mask-raw.fits
    Cluster-Cubes/J0328.6-5542.fitsss                  (typo'd extension)
    Cluster-Cubes/J0314.3-4525-RXC-J0314.3-4525_mask-raw.fits
    noise_cubes/J0351.1-8212_sofia_basic.par           (not _sofia.par)

So matching is done on a normalised key rather than on an assumed pattern, and
anything missing is reported instead of silently skipped. A field with no
science par cannot be used, because injections must go through the same
configuration as the real catalogue.

Usage
-----
    python -m temba.fields scan \
        --root /net/.../HI_MGCLS_tempfolder --out fields

    python -m temba.fields check --config fields/Abell-194/field.yaml
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError:                                            # pragma: no cover
    yaml = None


MASK_SUFFIXES = ("_mask.fits", "_mask-raw.fits", "_mask-2d.fits")


def normalise(name: str) -> str:
    """Field key that survives the naming inconsistencies."""
    s = Path(name).name
    for suf in (".fitsss", ".fits", ".xml", ".par"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"(_mask(-raw|-2d)?|_cat|_noise|_sofia(_basic)?|_rms)$", "", s)
    s = s.replace("_", "-").strip("-").lower()
    # J0314.3-4525-RXC-J0314.3-4525 -> J0314.3-4525
    m = re.match(r"^(j\d{4}\.\d[+-]\d{4})-rxc-", s)
    if m:
        s = m.group(1)
    return s


def index(directory, patterns=("*.fits", "*.fitsss", "*.xml", "*.par")):
    d = Path(directory)
    out = {}
    if not d.is_dir():
        return out
    for pat in patterns:
        for p in sorted(d.glob(pat)):
            out.setdefault(normalise(p.name), []).append(p)
    return out


def classify(paths):
    """Split a field's files into cube / masks / catalogue / par / rms."""
    got = dict(cube=None, mask=None, mask_raw=None, catalogue=None,
               par=None, rms=None, other=[])
    for p in paths:
        n = p.name
        if n.endswith(".par"):
            got["par"] = got["par"] or p
        elif n.endswith("_cat.xml"):
            got["catalogue"] = got["catalogue"] or p
        elif n.endswith("_noise.fits"):
            got["rms"] = got["rms"] or p
        elif n.endswith("_mask.fits"):
            got["mask"] = got["mask"] or p
        elif n.endswith("_mask-raw.fits"):
            got["mask_raw"] = got["mask_raw"] or p
        elif n.endswith("_mask-2d.fits"):
            pass
        elif n.endswith(".fits") or n.endswith(".fitsss"):
            got["cube"] = got["cube"] or p
        else:
            got["other"].append(p)
    return got


DEFAULT_CFG = dict(
    cosmology=dict(H0=70.0, Om0=0.3),
    substrate_build=dict(dilate_xy=2, dilate_z=2, seed=42),
    injection=dict(
        # Density-matched: 150 sources in Abell-194's 186 Mvox. Scaling by
        # volume keeps the validated source density on every cube instead of
        # sampling the large ones 3.7x more sparsely for 3.7x the SoFiA cost.
        sources_per_mvox=0.806,
        min_sources=50, max_sources=1200,
        sources_per_realisation=150,
        model_oversample=5,
        log_snr_min=-0.5, log_snr_max=2.0,
        w50_min=40.0, w50_max=600.0,
        dratio_min=0.10, dratio_max=4.0,
        inc_min=15.0, inc_max=85.0,
        sigma_hi_min=0.5, sigma_hi_max=30.0,
        w50_tol=0.10, flux_tol=0.05,
    ),
    matching=dict(beam_factor=2.0, dv_channels=2.0, min_rel=0.0),
)


def scan(root, out, cubes="Cluster-Cubes", masks="Cluster-Masks",
         noise="noise_cubes", tools=None):
    root = Path(root)
    idx = {}
    for sub in (cubes, masks, noise):
        for key, paths in index(root / sub).items():
            idx.setdefault(key, []).extend(paths)

    rows, written = [], 0
    for key in sorted(idx):
        g = classify(idx[key])
        # prefer the reliability-filtered mask; the raw mask also contains
        # noise peaks, and blanking those would make the substrate too clean
        mask = g["mask"] or g["mask_raw"]
        ok = bool(g["cube"] and mask and g["par"])
        rows.append((key, g, mask, ok))
        if not ok:
            continue

        # keep the original capitalisation for output paths and figure labels
        disp = Path(g["cube"]).name
        for suf in (".fitsss", ".fits"):
            if disp.endswith(suf):
                disp = disp[: -len(suf)]

        cfg = dict(field=disp,
                   cube=str(g["cube"]), mask=str(mask),
                   mask_is_raw=bool(g["mask"] is None),
                   science_par=str(g["par"]),
                   science_catalogue=str(g["catalogue"]) if g["catalogue"] else None,
                   rms_cube=str(g["rms"]) if g["rms"] else None,
                   substrate=f"substrates/{disp}_substrate.fits",
                   noise_map=f"substrates/{disp}_sigma_chan.fits",
                   tools=tools or dict(sofia=None, bbarolo=None, ld_library_path=None),
                   **DEFAULT_CFG)
        d = Path(out) / disp
        d.mkdir(parents=True, exist_ok=True)
        text = (yaml.safe_dump(cfg, sort_keys=False) if yaml
                else "# install pyyaml to write this properly\n" + repr(cfg))
        (d / "field.yaml").write_text(text)
        written += 1

    w = 96
    print("=" * w)
    print(f"TEMBA FIELD SCAN  {root}")
    print("=" * w)
    print(f"  {'field':<18}{'cube':>6}{'mask':>7}{'raw':>6}{'par':>6}{'cat':>6}{'rms':>6}   status")
    print("  " + "-" * (w - 4))
    for key, g, mask, ok in rows:
        y = lambda v: "yes" if v else " - "
        note = "ready" if ok else "MISSING: " + ", ".join(
            n for n, v in (("cube", g["cube"]), ("mask", mask), ("par", g["par"])) if not v)
        if ok and g["mask"] is None:
            note = "ready (raw mask only)"
        print(f"  {key:<18}{y(g['cube']):>6}{y(g['mask']):>7}{y(g['mask_raw']):>6}"
              f"{y(g['par']):>6}{y(g['catalogue']):>6}{y(g['rms']):>6}   {note}")
    print()
    print(f"  {written} field config(s) written under {out}/")
    print("  Set tools.sofia, tools.bbarolo and tools.ld_library_path before running.")
    return written


def check(config):
    if yaml is None:
        raise SystemExit("pyyaml is required for `check`.")
    cfg = yaml.safe_load(Path(config).read_text())
    ok = True
    print(f"  field: {cfg['field']}")
    for key in ("cube", "mask", "science_par", "science_catalogue", "rms_cube"):
        p = cfg.get(key)
        if p is None:
            print(f"    {key:<20} (not set)")
            continue
        e = Path(p).exists()
        ok &= e or key in ("science_catalogue", "rms_cube")
        print(f"    {key:<20} {'OK  ' if e else 'MISS'} {p}")

    t = cfg.get("tools", {}) or {}
    env = dict(LD_LIBRARY_PATH=t.get("ld_library_path") or "")
    for key in ("sofia", "bbarolo"):
        exe = t.get(key)
        if not exe:
            print(f"    {key:<20} NOT SET"); ok = False; continue
        if not Path(exe).exists():
            print(f"    {key:<20} MISS {exe}"); ok = False; continue
        # actually execute it: a missing shared library only shows up here,
        # and discovering that on realisation 1 of 34 wastes a day
        import os
        e2 = os.environ.copy()
        if env["LD_LIBRARY_PATH"]:
            e2["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"] + ":" + e2.get("LD_LIBRARY_PATH", "")
        try:
            r = subprocess.run([exe], capture_output=True, text=True,
                               timeout=60, env=e2,
                               stdin=subprocess.DEVNULL)
            bad = "error while loading shared libraries" in (r.stderr + r.stdout)
            print(f"    {key:<20} {'RUNS' if not bad else 'LINK ERROR'} {exe}")
            ok &= not bad
        except Exception as exc:
            print(f"    {key:<20} FAIL {exc}"); ok = False

    print(f"\n  {'READY' if ok else 'NOT READY'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)

    s = sub.add_parser("scan", help="Discover fields and write field.yaml files.")
    s.add_argument("--root", required=True)
    s.add_argument("--out", default="fields")
    s.add_argument("--cubes-dir", default="Cluster-Cubes")
    s.add_argument("--masks-dir", default="Cluster-Masks")
    s.add_argument("--noise-dir", default="noise_cubes")
    s.add_argument("--sofia"); s.add_argument("--bbarolo")
    s.add_argument("--ld-library-path")

    c = sub.add_parser("check", help="Verify one field's inputs and tools.")
    c.add_argument("--config", required=True)

    a = ap.parse_args()
    if a.action == "scan":
        tools = dict(sofia=a.sofia, bbarolo=a.bbarolo,
                     ld_library_path=a.ld_library_path)
        scan(a.root, a.out, a.cubes_dir, a.masks_dir, a.noise_dir, tools)
    else:
        raise SystemExit(check(a.config))


if __name__ == "__main__":
    main()
