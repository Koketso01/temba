"""
temba.substrate -- build the injection substrate for TEMBA.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

The substrate is the cube that mock galaxies are injected into. It must be a
signal-free *realisation* of the real data -- not a description of it.

This is the module that exists because v0.5 injected onto SoFiA's local-RMS
cube (output.writeNoise), which is strictly positive and contains no noise at
all. Every source was therefore detected, and every mask summed a positive
pedestal into f_sum. The `validate` action below is the check that would have
caught it: a genuine noise realisation has zero mean and ~50% negative voxels.

Method
------
Take the science cube, blank the real SoFiA mask (dilated, because masks are
tight and emission wings survive outside them), then fill the holes by copying
from a randomly chosen, signal-free part of the *same* spectrum. This keeps:

  * the real per-channel noise level and its spatial variation (primary beam)
  * beam-scale spatial correlation, because donor voxels come from real data
  * residual continuum, imaging artefacts and RFI, which is the whole reason
    the thesis argues for a real substrate rather than white noise

and removes the real sources, so nothing competes with the injections.

Usage
-----
    python -m temba.substrate build \
        --cube    Cluster-Cubes/Abell-194.fits \
        --mask    Cluster-Masks/Abell_194_mask.fits \
        --out     substrates/Abell-194_substrate.fits \
        --rms-cube noise_cubes/Abell-194_noise.fits

    python -m temba.substrate validate --cube <any cube> [--rms-cube <rms>]

`validate` runs automatically at the end of `build` and exits non-zero if the
result does not look like noise, so a bad substrate cannot silently propagate
into a campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

try:
    from scipy.ndimage import binary_dilation
    HAVE_SCIPY = True
except ImportError:                                            # pragma: no cover
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def detect_blank(data, rng=None, nsample=4_000_000):
    """Find the sentinel value a cube uses for blanked voxels, or None.

    MGCLS cubes blank outside the primary beam with a literal constant (1e-10)
    rather than NaN. Those voxels must be excluded from the statistics, and
    must never be used as donors when filling masked holes.
    """
    flat = np.asarray(data).ravel()
    rng = rng or np.random.default_rng(0)
    s = flat[rng.integers(0, flat.size, size=nsample)] if flat.size > nsample else flat
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    v, c = np.unique(s, return_counts=True)
    i = int(np.argmax(c))
    return float(v[i]) if c[i] / s.size > 0.02 else None


def cube_statistics(data, sample=None, rng=None, blank=None):
    """Robust statistics on the valid voxels of `data`.

    `sample` caps how many voxels are used, so this stays cheap on big cubes.
    `blank` excludes a sentinel blank value in addition to NaN.
    """
    finite = np.isfinite(data)
    if blank is not None:
        finite &= (data != blank)
    vals = data[finite]
    if vals.size == 0:
        raise SystemExit("Cube contains no finite voxels.")
    if sample and vals.size > sample:
        rng = rng or np.random.default_rng(0)
        vals = rng.choice(vals, size=sample, replace=False)
    vals = vals.astype(np.float64)

    med = float(np.median(vals))
    mad = float(1.4826 * np.median(np.abs(vals - med)))
    return dict(
        n_finite=int(finite.sum()),
        n_blank=int(finite.size - finite.sum()),
        mean=float(vals.mean()),
        median=med,
        std=float(vals.std()),
        mad_sigma=mad,
        frac_negative=float((vals < 0).mean()),
        vmin=float(vals.min()),
        vmax=float(vals.max()),
    )


def verdict(stats, rms_median=None):
    """Decide whether these statistics are consistent with a noise realisation."""
    problems = []
    fn = stats["frac_negative"]
    if fn < 0.40:
        problems.append(
            f"only {fn:.1%} of voxels are negative (expected ~50%). "
            "This looks like an RMS/noise *map*, an absolute-value cube, or a "
            "cube that still contains signal -- not a noise realisation."
        )
    elif fn > 0.60:
        problems.append(f"{fn:.1%} of voxels are negative (expected ~50%).")

    if stats["mad_sigma"] > 0:
        # Median, not mean: MAD-sigma is a robust scale, so pairing it with a
        # non-robust location is inconsistent. A few very bright voxels -- a
        # continuum residual, say -- pull the mean far from zero while the bulk
        # of the cube is perfectly good noise. On J0314.3-4525 the mean sits
        # 1.76 sigma from zero and the median 0.03 sigma, and the substrate is
        # sound; judging it on the mean rejected a usable field.
        pull = abs(stats["median"]) / stats["mad_sigma"]
        if pull > 0.1:
            problems.append(
                f"median is {pull:.2f} x sigma away from zero; a noise "
                "realisation should have a median consistent with zero."
            )
        # Bright outliers are not a reason to reject, but they should be seen.
        spread = stats["std"] / stats["mad_sigma"]
        if spread > 3.0:
            print(f"  note: std is {spread:.1f}x MAD-sigma, so the cube carries "
                  "bright outliers\n        (continuum residual or RFI). The "
                  "noise itself is judged on the MAD.")

    if rms_median is not None and stats["mad_sigma"] > 0:
        ratio = stats["mad_sigma"] / rms_median
        if not 0.7 < ratio < 1.4:
            problems.append(
                f"MAD-sigma is {ratio:.2f}x the median of the supplied RMS cube; "
                "expected ~1. The substrate noise level does not match the data."
            )
    return problems


def report(stats, rms_median=None, title="CUBE STATISTICS", blank=None):
    w = 70
    print("=" * w)
    print(title)
    print("=" * w)
    if blank is not None:
        print(f"  blank sentinel   {blank:.6g}   (excluded from statistics)")
    print(f"  valid voxels     {stats['n_finite']:,}   excluded {stats['n_blank']:,}")
    print(f"  mean             {stats['mean']:.6g}")
    print(f"  median           {stats['median']:.6g}")
    print(f"  std              {stats['std']:.6g}")
    print(f"  MAD-sigma        {stats['mad_sigma']:.6g}")
    print(f"  frac negative    {stats['frac_negative']:.3f}")
    print(f"  min / max        {stats['vmin']:.6g}  /  {stats['vmax']:.6g}")
    if rms_median is not None:
        print(f"  RMS-cube median  {rms_median:.6g}   "
              f"(ratio {stats['mad_sigma']/rms_median:.3f})")
    problems = verdict(stats, rms_median)
    print()
    if problems:
        print("  NOT A NOISE REALISATION:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  OK: consistent with a signal-free noise realisation.")
    print()
    return problems


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def dilate_slab(mask_slab, nxy, nz):
    """Dilate a boolean mask slab by nxy pixels spatially and nz channels."""
    if not (nxy or nz):
        return mask_slab
    if not HAVE_SCIPY:
        raise SystemExit("scipy is required for mask dilation (pip install scipy).")
    out = mask_slab
    if nz:
        se = np.zeros((2 * nz + 1, 1, 1), dtype=bool)
        se[:] = True
        out = binary_dilation(out, structure=se)
    if nxy:
        se = np.zeros((1, 2 * nxy + 1, 2 * nxy + 1), dtype=bool)
        se[:] = True
        out = binary_dilation(out, structure=se)
    return out


def fill_slab(data, bad, rng, attempts=4):
    """Replace `bad` voxels with signal-free data from the same spectrum.

    data : (nz, ny, nx) float32, modified in place
    bad  : (nz, ny, nx) bool, voxels to replace

    For each spatial pixel the spectrum is rolled by a random offset and the
    rolled value is copied in. Real HI masks occupy a small fraction of the
    band, so one roll almost always lands on clean data; remaining voxels get
    another roll, and anything still unfilled falls back to a Gaussian draw at
    the pixel's own noise level.
    """
    nz, ny, nx = data.shape
    flat = data.reshape(nz, -1)
    flatbad = bad.reshape(nz, -1)
    npix = flat.shape[1]
    todo = flatbad.copy()
    zidx = np.arange(nz)[:, None]

    for _ in range(attempts):
        if not todo.any():
            break
        # a roll well away from zero, so donors are not adjacent channels
        rolls = rng.integers(nz // 8 + 1, nz - nz // 8, size=npix)
        idx = (zidx + rolls[None, :]) % nz
        donor = np.take_along_axis(flat, idx, axis=0)
        donor_bad = np.take_along_axis(flatbad, idx, axis=0) | ~np.isfinite(donor)
        take = todo & ~donor_bad
        flat[take] = donor[take]
        todo &= ~take

    if todo.any():
        # per-pixel noise level from the voxels we did not have to touch
        clean = np.where(flatbad | ~np.isfinite(flat), np.nan, flat)
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # fully blanked columns (cube edges) are expected and give all-NaN slices
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(clean, axis=0)
            sig = 1.4826 * np.nanmedian(np.abs(clean - med[None, :]), axis=0)
        # A pixel whose spectrum is almost entirely blank gives a MAD estimated
        # from a handful of samples, and the Gaussian drawn from it can be wild:
        # on J0314.3-4525, where a quarter of the band is flagged everywhere,
        # this rewrote 37% of some planes and produced 1400-sigma values. Leave
        # such pixels as they were rather than inventing noise for them.
        nclean = np.sum(~flatbad & np.isfinite(flat), axis=0)
        usable = np.isfinite(sig) & (sig > 0) & (nclean >= 20)
        sig = np.where(usable, sig, np.nan)
        med = np.where(np.isfinite(med), med, 0.0)
        cols = np.where(todo.any(axis=0) & usable)[0]
        for c in cols:
            rows = np.where(todo[:, c])[0]
            flat[rows, c] = rng.normal(med[c], sig[c], size=rows.size)
        skipped = int(np.sum(todo.any(axis=0) & ~usable))
        if skipped:
            print(f"    left {skipped} pixels unfilled (fewer than 20 clean "
                  f"channels)", flush=True)

    return int(flatbad.sum())


def build(cube_path, mask_path, out_path, dilate_xy=2, dilate_z=2,
          seed=42, slab=96, rms_path=None):
    rng = np.random.default_rng(seed)

    with fits.open(cube_path, memmap=True) as hd:
        hdr = hd[0].header.copy()
        shape = hd[0].data.shape
    if len(shape) == 4:
        raise SystemExit(
            "Cube has 4 axes. Drop the degenerate Stokes axis first, e.g. with "
            "`fitsheader` + a simple numpy squeeze, and rerun."
        )
    nz, ny, nx = shape

    with fits.open(mask_path, memmap=True) as hd:
        mshape = hd[0].data.shape
    if tuple(mshape[-3:]) != (nz, ny, nx):
        raise SystemExit(f"Mask shape {mshape} does not match cube shape {shape}.")

    # allocate the output file without holding the whole cube in memory
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ohdr = hdr.copy()
    ohdr["BITPIX"] = -32
    ohdr.add_history("TEMBA: signal-free substrate built by temba.substrate")
    ohdr.add_history(f"TEMBA: source cube {Path(cube_path).name}")
    ohdr.add_history(f"TEMBA: mask {Path(mask_path).name} "
                     f"dilated xy={dilate_xy} z={dilate_z}")
    ohdr.tofile(out_path, overwrite=True)
    nbytes = int(np.prod(shape)) * 4
    with open(out_path, "rb+") as fh:
        fh.seek(len(ohdr.tostring()) + (-(-nbytes // 2880) * 2880) - 1)
        fh.write(b"\0")

    with fits.open(cube_path, memmap=True) as ch:
        blank = detect_blank(np.asarray(ch[0].data[::max(1, nz // 32)]))
    if blank is not None:
        print(f"  cube blanks with the sentinel {blank:.6g}; those voxels are "
              "excluded and written through unchanged")

    n_masked = 0
    halo = max(dilate_xy, 1)
    with fits.open(cube_path, memmap=True) as ch, \
         fits.open(mask_path, memmap=True) as mh, \
         fits.open(out_path, mode="update", memmap=True) as oh:

        cdata, mdata, odata = ch[0].data, mh[0].data, oh[0].data
        if mdata.ndim == 4:
            mdata = mdata[0]

        for y0 in range(0, ny, slab):
            y1 = min(y0 + slab, ny)
            ylo, yhi = max(0, y0 - halo), min(ny, y1 + halo)

            block = np.array(cdata[:, ylo:yhi, :], dtype=np.float32)
            blanked = ~np.isfinite(block)
            if blank is not None:
                blanked |= (block == blank)

            bad = np.asarray(mdata[:, ylo:yhi, :]) > 0
            bad = dilate_slab(bad, dilate_xy, dilate_z)
            n_masked += int((bad & ~blanked).sum())

            # blanked voxels are never donors, and are put back afterwards
            bad |= blanked
            fill_slab(block, bad, rng)

            original = np.asarray(cdata[:, ylo:yhi, :])
            block[blanked] = original[blanked]

            a, b = y0 - ylo, y0 - ylo + (y1 - y0)
            odata[:, y0:y1, :] = block[:, a:b, :]
            print(f"  rows {y0:5d}-{y1:5d}  done", flush=True)

        oh.flush()

    rms_median = None
    if rms_path:
        with fits.open(rms_path, memmap=True) as rh:
            r = np.asarray(rh[0].data)
            r = r[np.isfinite(r) & (r > 0)]
            rms_median = float(np.median(r)) if r.size else None

    with fits.open(out_path, memmap=True) as oh:
        stats = cube_statistics(np.asarray(oh[0].data), sample=20_000_000,
                                rng=np.random.default_rng(1), blank=blank)

    print()
    problems = report(stats, rms_median, title=f"SUBSTRATE: {out_path.name}",
                      blank=blank)
    meta = dict(cube=str(cube_path), mask=str(mask_path), out=str(out_path),
                dilate_xy=dilate_xy, dilate_z=dilate_z, seed=seed, blank=blank,
                voxels_replaced=n_masked,
                fraction_replaced=n_masked / float(np.prod(shape)),
                statistics=stats, rms_cube_median=rms_median,
                problems=problems)
    Path(str(out_path) + ".temba.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  replaced {n_masked:,} voxels "
          f"({100*n_masked/np.prod(shape):.3f}% of the cube)")
    print(f"  wrote {out_path}")
    print(f"  wrote {out_path}.temba.json")
    return problems


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)

    v = sub.add_parser("validate", help="Check whether a cube is a noise realisation.")
    v.add_argument("--cube", required=True)
    v.add_argument("--rms-cube", help="SoFiA RMS cube, to check the noise level matches.")

    b = sub.add_parser("build", help="Build a substrate from a science cube and its mask.")
    b.add_argument("--cube", required=True, help="Science cube from the real run.")
    b.add_argument("--mask", required=True, help="SoFiA mask from the real run.")
    b.add_argument("--out", required=True)
    b.add_argument("--rms-cube", help="SoFiA RMS cube, for the noise-level check.")
    b.add_argument("--dilate-xy", type=int, default=2,
                   help="Grow the mask by this many pixels spatially (default 2).")
    b.add_argument("--dilate-z", type=int, default=2,
                   help="Grow the mask by this many channels (default 2).")
    b.add_argument("--slab", type=int, default=96, help="Rows processed at a time.")
    b.add_argument("--seed", type=int, default=42)

    a = ap.parse_args()

    rms_median = None
    if getattr(a, "rms_cube", None):
        with fits.open(a.rms_cube, memmap=True) as rh:
            r = np.asarray(rh[0].data)
            r = r[np.isfinite(r) & (r > 0)]
            rms_median = float(np.median(r)) if r.size else None

    if a.action == "validate":
        with fits.open(a.cube, memmap=True) as hd:
            data = np.asarray(hd[0].data)
            blank = detect_blank(data[::max(1, data.shape[0] // 32)])
            stats = cube_statistics(data, sample=20_000_000, blank=blank)
        problems = report(stats, rms_median, title=f"VALIDATE: {Path(a.cube).name}",
                          blank=blank)
    else:
        problems = build(a.cube, a.mask, a.out, a.dilate_xy, a.dilate_z,
                         a.seed, a.slab, a.rms_cube)

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
