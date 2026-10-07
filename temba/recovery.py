"""
temba.recovery -- how much flux comes back, and what it depends on.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

temba.selection answers "was it detected". This module answers the other half,
which is the point of the experiment: for the sources that *were* detected, how
much of their flux density does the pipeline recover, and how does that depend
on where they are and what they look like?

Three quantities, kept separate throughout:

    capture   = F_model_in_mask / F_inj     bounded by 1; how much of the real
                                            signal SoFiA's mask encloses
    fidelity  = f_sum / F_inj               unbounded; what the catalogue says
    noise_in_mask = f_sum - F_model_in_mask what the mask dragged in

They fail for different reasons. Capture falls when the source is larger or
broader than the mask grows to cover, so it is governed by angular size,
linewidth and inclination. Noise in the mask grows with mask volume, roughly as
sqrt(n_pix) x sigma_local, so it matters most for extended sources in noisy
parts of the field. Fidelity is the sum of the two effects and is therefore the
least interpretable on its own, even though it is what a catalogue user sees.

Covariates
----------
    log SNR_int         integrated signal-to-noise (the controlled axis)
    log W50             linewidth
    log dHI/beam        angular size relative to the beam
    cos i               inclination
    log sigma_chan      local noise: the primary beam varies this by ~7x
                        across an MGCLS field, so position enters here
    log vrot            rotation velocity (collinear with W50 and i by
                        construction, so reported with its VIF and excluded
                        from the joint fit by default)
    radius              distance from the field centre, as an independent
                        check on whether sigma_chan captures the whole
                        positional dependence
    n_pix               mask volume, for the noise term

Channel width is constant within a field, so it only becomes a covariate when
realisations from several fields are combined; --by-field reports each field
separately so the comparison is visible.

Models are fitted to log10 of the response with standardised predictors, so a
coefficient is the change in dex per 1-sigma change in that covariate, and
errors come from bootstrap rather than from an assumed error model.

Usage
-----
    python -m temba.recovery --recovery 'runs/Abell-194/run_*/products/recovery.ecsv' \
        --outdir runs/Abell-194/analysis --label "Abell 194"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from temba.selection import (COL, use_style, finish, save, equal_count_bins,
                             running_quantiles)


# ---------------------------------------------------------------------------

COVARIATES = [
    ("snr_design",         "log",  r"$\log_{10}\,{\rm SNR_{int}}$",            "log SNR"),
    ("W50_realised_kms",   "log",  r"$\log_{10}\,(W_{50}/{\rm km\,s^{-1}})$",  "log W50"),
    ("dHI_over_beam",      "log",  r"$\log_{10}\,(d_{\rm HI}/\theta_{\rm beam})$", "log dHI/beam"),
    ("inc_deg",            "cos",  r"$\cos i$",                                "cos inc"),
    ("sigma_chan_Jybeam",  "log",  r"$\log_{10}\,\sigma_{\rm chan}$",          "log sigma_chan"),
    ("vrot_solved_kms",    "log",  r"$\log_{10}\,(v_{\rm rot}/{\rm km\,s^{-1}})$", "log vrot"),
]

ALIASES = {
    "W50_realised_kms": ["W50_realised_kms", "W50_kms"],
    "vrot_solved_kms": ["vrot_solved_kms", "vrot_kms"],
}


def resolve(t, name):
    for c in ALIASES.get(name, [name]):
        if c in t.colnames:
            return c
    return None


def transform(t, name, kind):
    c = resolve(t, name)
    if c is None:
        return None
    v = np.asarray(t[c], float)
    if kind == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, np.log10(np.where(v > 0, v, 1)), np.nan)
    if kind == "cos":
        return np.cos(np.radians(v))
    return v


def field_radius(t):
    """Distance from the field centre in pixels, as a position proxy."""
    if "xpix" not in t.colnames or "ypix" not in t.colnames:
        return None
    x = np.asarray(t["xpix"], float); y = np.asarray(t["ypix"], float)
    return np.hypot(x - np.median(x), y - np.median(y))


# ---------------------------------------------------------------------------

def standardise(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def vif(X):
    out = []
    for j in range(X.shape[1]):
        o = np.delete(X, j, axis=1)
        b, *_ = np.linalg.lstsq(o, X[:, j], rcond=None)
        r2 = 1 - np.sum((X[:, j] - o @ b) ** 2) / max(np.sum(X[:, j] ** 2), 1e-30)
        out.append(1.0 / max(1 - r2, 1e-12))
    return np.array(out)


def regress(y, X, names, n_boot=500, seed=0):
    """OLS on log10(y) with standardised predictors and bootstrap errors."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)

    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        b, *_ = np.linalg.lstsq(A[k], y[k], rcond=None)
        bs.append(b)
    bs = np.array(bs)
    se = bs.std(axis=0)
    return dict(names=["intercept"] + list(names),
                beta=beta.tolist(), se=se.tolist(),
                z=(beta / np.where(se > 0, se, np.inf)).tolist(),
                r2=float(r2), n=int(len(y)),
                vif=[float("nan")] + vif(X).tolist())


# ---------------------------------------------------------------------------

def fig_covariate_grid(t, det, outdir, label, cols, ycol, ylab, colour, name):
    y = np.asarray(t[ycol], float)
    use = det & np.isfinite(y)
    panels = [(k, tr, xl) for k, tr, xl, _ in cols
              if transform(t, k, tr) is not None]
    r = field_radius(t)
    if r is not None:
        panels.append(("__radius__", None, r"radius from field centre [pix]"))

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.15 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (k, tr, xl) in zip(axes, panels):
        x = r if k == "__radius__" else transform(t, k, tr)
        m = use & np.isfinite(x)
        if m.sum() < 8:
            ax.set_axis_off(); continue
        ax.scatter(x[m], y[m], s=4, color=colour, alpha=0.30, lw=0, zorder=2)
        e = equal_count_bins(x[m], 7, 5)
        xs, md, lo, hi, _ = running_quantiles(x[m], y[m], e)
        if len(xs):
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.20, lw=0, zorder=1)
            ax.plot(xs, md, "-", color=colour, lw=1.4, zorder=3)
        ax.axhline(1.0, color=COL["grey"], lw=0.6, ls="--", zorder=0)
        if k == "dHI_over_beam":
            ax.axvline(0.0, color=COL["grey"], lw=0.5, ls=":", zorder=0)
        ax.set_xlabel(xl)
        finish(ax)
    for ax in axes[len(panels):]:
        ax.set_axis_off()
    for i in range(0, len(axes), ncol):
        axes[i].set_ylabel(ylab)
    fig.suptitle(label, x=0.005, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, outdir, name)


def fig_spatial(t, det, outdir, label, nb=6):
    if "xpix" not in t.colnames:
        return []
    x = np.asarray(t["xpix"], float); y = np.asarray(t["ypix"], float)
    fid = np.asarray(t["flux_fidelity"], float)
    cap = np.asarray(t["flux_capture"], float)
    m = det & np.isfinite(fid)
    if m.sum() < 40:
        print(f"  (no spatial map: only {m.sum()} detections)")
        return []

    xe = np.linspace(x.min(), x.max(), nb + 1)
    ye = np.linspace(y.min(), y.max(), nb + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for ax, (v, lab, cmap) in zip(axes, [(cap, "median capture", "viridis"),
                                         (fid, "median fidelity", "magma")]):
        Z = np.full((nb, nb), np.nan); N = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                s = (m & (x >= xe[j]) & (x < xe[j + 1])
                     & (y >= ye[i]) & (y < ye[i + 1]) & np.isfinite(v))
                if s.sum() >= 3:
                    Z[i, j] = np.median(v[s]); N[i, j] = s.sum()
        pc = ax.pcolormesh(xe, ye, Z, cmap=cmap, shading="flat",
                           edgecolors="white", linewidth=0.3)
        cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.set_label(lab)
        cb.outline.set_linewidth(0.7)
        ax.set_xlabel("x [pix]"); ax.set_aspect("equal")
        finish(ax)
    axes[0].set_ylabel("y [pix]")
    fig.suptitle(label + "   (detections only)", x=0.005, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, outdir, "recovery_spatial")


def fig_noise_in_mask(t, det, outdir, label):
    if "noise_in_mask_Jykms" not in t.colnames or "n_pix" not in t.colnames:
        return []
    n = np.asarray(t["n_pix"], float)
    nm = np.asarray(t["noise_in_mask_Jykms"], float)
    sig = transform(t, "sigma_chan_Jybeam", "log")
    m = det & np.isfinite(n) & np.isfinite(nm) & (n > 0)
    if m.sum() < 10:
        return []

    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    c = 10 ** sig[m] if sig is not None else None
    sc = ax.scatter(np.log10(n[m]), nm[m], s=8, c=c, cmap="viridis", lw=0,
                    alpha=0.75)
    if c is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$\sigma_{\rm chan}$ [Jy beam$^{-1}$]")
        cb.outline.set_linewidth(0.7)
    e = equal_count_bins(np.log10(n[m]), 7, 5)
    xs, md, lo, hi, _ = running_quantiles(np.log10(n[m]), nm[m], e)
    if len(xs):
        ax.fill_between(xs, lo, hi, color=COL["grey"], alpha=0.18, lw=0, zorder=1)
        ax.plot(xs, md, "-", color=COL["data"], lw=1.4, zorder=3)
    ax.axhline(0.0, color=COL["grey"], lw=0.6, ls="--", zorder=0)
    ax.set_xlabel(r"$\log_{10}\,n_{\rm pix}$  (mask volume)")
    ax.set_ylabel(r"noise in mask [Jy km s$^{-1}$]")
    ax.set_title(label, pad=4, loc="left", fontsize=9)
    finish(ax)
    return save(fig, outdir, "noise_in_mask")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recovery", required=True, nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--with-vrot", action="store_true",
                    help="Include log vrot in the joint fit despite its "
                         "collinearity with W50 and inclination.")
    a = ap.parse_args()

    paths = []
    for p in a.recovery:
        paths.extend(sorted(glob.glob(p)) or [p])
    tabs = [Table.read(p, format="ascii.ecsv") for p in paths]
    t = vstack(tabs, metadata_conflicts="silent") if len(tabs) > 1 else tabs[0]
    det = np.asarray(t["detected"], bool)

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    use_style()

    cols = [c for c in COVARIATES if not (c[3] == "log vrot" and not a.with_vrot)]
    names, X = [], []
    for key, tr, _, nm in cols:
        v = transform(t, key, tr)
        if v is not None:
            names.append(nm); X.append(v)
    r = field_radius(t)
    if r is not None:
        names.append("radius"); X.append(r)
    X = np.column_stack(X) if X else np.zeros((len(t), 0))

    res = dict(n_tables=len(paths), n_injected=int(len(t)),
               n_detected=int(det.sum()), covariates=names)
    figs = []

    w = 80
    print("=" * w)
    print(f"TEMBA FLUX RECOVERY   {a.label}")
    print("=" * w)
    print(f"  {len(paths)} table(s), {len(t)} injected, {int(det.sum())} detected")

    # Flux is one recovered parameter among several. The size ratios are the
    # interesting ones: ell3s_maj / d_3sigma should be FLAT against size and
    # local noise if the 0.697 offset is a pure convention constant, which is
    # what turns the observable-to-intrinsic size relation into a formula.
    for ycol, lab, colour, short in (
            ("flux_capture", r"capture  $F_{\rm mask}/F_{\rm inj}$", COL["alt"], "capture"),
            ("flux_fidelity", r"fidelity  $f_{\rm sum}/F_{\rm inj}$", COL["data"], "fidelity"),
            ("W50_ratio", r"$W_{50}^{\rm SoFiA}/W_{50}^{\rm inj}$", COL["band"], "W50"),
            ("ell3s_over_dHI", r"$\theta_{3\sigma}^{\rm SoFiA}/d_{\rm HI}$",
             COL["fit"], "size_vs_dHI"),
            ("ell3s_over_d3sigma", r"$\theta_{3\sigma}^{\rm SoFiA}/d_{3\sigma}$",
             COL["alt"], "size_vs_d3sigma")):
        if ycol not in t.colnames:
            continue
        y = np.asarray(t[ycol], float)
        m = det & np.isfinite(y) & (y > 0) & np.isfinite(X).all(axis=1)
        if m.sum() < 20:
            print(f"\n  {short}: only {m.sum()} usable sources, no fit")
            continue

        Xs, _, _ = standardise(X[m])
        fit = regress(np.log10(y[m]), Xs, names, a.bootstrap)
        res[short] = dict(
            median=float(np.median(y[m])), p16=float(np.percentile(y[m], 16)),
            p84=float(np.percentile(y[m], 84)), fit=fit)

        print()
        print(f"  {short.upper()}   median {np.median(y[m]):.3f} "
              f"[{np.percentile(y[m],16):.3f}, {np.percentile(y[m],84):.3f}]   "
              f"n = {m.sum()}")
        print("  " + "-" * (w - 4))
        print(f"    {'term':<18}{'d(dex)/sigma':>14}{'se':>9}{'z':>8}{'VIF':>8}")
        for nm, b, s_, z, vf in zip(fit["names"], fit["beta"], fit["se"],
                                    fit["z"], fit["vif"]):
            flag = "  <- collinear" if np.isfinite(vf) and vf > 5 else ""
            v = f"{vf:8.2f}" if np.isfinite(vf) else " " * 8
            print(f"    {nm:<18}{b:14.4f}{s_:9.4f}{z:8.2f}{v}{flag}")
        print(f"    R^2 = {fit['r2']:.3f}")

        figs += fig_covariate_grid(t, det, outdir, a.label, cols, ycol, lab,
                                   colour, f"recovery_{short}_vs_covariates")

    figs += fig_spatial(t, det, outdir, a.label)
    figs += fig_noise_in_mask(t, det, outdir, a.label)

    print()
    for f in figs:
        if f.endswith(".pdf"):
            print(f"  {f}")
    (outdir / "recovery.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(f"  {outdir/'recovery.json'}")


if __name__ == "__main__":
    main()
