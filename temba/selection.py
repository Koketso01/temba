"""
temba.selection -- selection function, flux-density recovery, and figures.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

Consumes one or more recovery tables from temba.match (several realisations
can be combined) and produces two co-equal products:

  DETECTION      P(detected | flux, SNR, linewidth, size, local noise)
  FLUX RECOVERY  capture  = F_model_in_mask / F_inj   (bounded by 1)
                 fidelity = f_sum / F_inj             (unbounded)

The second is not a diagnostic afterthought. Detection tells you whether a
source enters the catalogue; flux recovery tells you what its measured flux
means once it does, and the two together describe what the survey is losing.

Layer 1 is the thesis prescription, unchanged: bin in x = log10 S_int, Wilson
score intervals, and a logistic

    C(x) = [1 + exp(-(x - x0)/k)]^-1,
    log S50 = x0,  log S90 = x0 + k ln9,  log S10 = x0 - k ln9

fitted by binomial likelihood rather than least squares on the binned points.
Running this on the thesis-era Abell-194 table must return log S50 ~ -0.43 and
log S90 ~ 0.14 (thesis Table 2.2); that is a regression test, not a formality.

Layer 2 needs the TEMBA design columns and is skipped when they are absent, as
they are for pre-TEMBA runs. It fits

    logit P(det) = b0 + b1 logSNR + b2 logW50 + b3 log(dHI/beam) + b4 log sigma

where the covariates were sampled independently, so the coefficients are
separately identifiable. In a v0.5-style population they are not, and the
module says so rather than reporting numbers.

Usage
-----
    python -m temba.selection --recovery 'runs/Abell-194/run_*/products/recovery.ecsv' \
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
from matplotlib.ticker import AutoMinorLocator


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

COL = dict(data="#1b3a5c", fit="#c1440e", band="#7fa8c9",
           alt="#3d6b35", grey="#8a8a8a", light="#dce6ef")

def use_style(scale=1.0):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5 * scale,
        "axes.labelsize": 9 * scale,
        "axes.titlesize": 9 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "legend.fontsize": 7.5 * scale,
        "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": False,
    })


def finish(ax, minor=True):
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def save(fig, outdir, name):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(str(p))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.0):
    """Wilson score interval; z=1 gives the 68% interval used in the thesis."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def yerr(p, lo, hi):
    """Error bars from a Wilson interval, guarded against float noise.

    p lies inside the interval analytically, but at p = 1 the upper edge
    evaluates to 1 - 1e-16 and matplotlib rejects negative yerr.
    """
    return [np.clip(p - lo, 0.0, None), np.clip(hi - p, 0.0, None)]


def equal_count_bins(x, nbins, min_per_bin=8):
    x = np.asarray(x, float)
    nbins = max(1, min(nbins, len(x) // min_per_bin))
    return np.unique(np.percentile(x, np.linspace(0, 100, nbins + 1)))


def binned_completeness(x, det, edges):
    xs, N, K = [], [], []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        m = (x >= edges[i]) & (x < hi if i < len(edges) - 2 else x <= hi)
        if m.sum() == 0:
            continue
        xs.append(float(np.median(x[m]))); N.append(int(m.sum()))
        K.append(int(np.sum(det[m])))
    xs, N, K = np.array(xs), np.array(N), np.array(K)
    p, lo, hi = wilson(K, N)
    return xs, N, K, p, lo, hi


def logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-np.clip((np.asarray(x, float) - x0) / k, -60, 60)))


def fit_logistic(x, det, n_boot=400, seed=0):
    """Binomial-likelihood logistic fit, with bootstrap errors on S10/50/90."""
    from scipy.optimize import minimize
    x = np.asarray(x, float); det = np.asarray(det, bool)

    def nll(p, xx, dd):
        q = np.clip(logistic(xx, p[0], max(p[1], 1e-3)), 1e-9, 1 - 1e-9)
        return -np.sum(dd * np.log(q) + (~dd) * np.log(1 - q))

    p0 = [float(np.median(x)), 0.4]
    r = minimize(nll, p0, args=(x, det), method="Nelder-Mead",
                 options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
    x0, k = float(r.x[0]), float(abs(r.x[1]))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        j = rng.integers(0, len(x), len(x))
        try:
            rb = minimize(nll, [x0, k], args=(x[j], det[j]), method="Nelder-Mead",
                          options=dict(maxiter=2000, xatol=1e-3, fatol=1e-3))
            boots.append([rb.x[0], abs(rb.x[1])])
        except Exception:
            pass
    boots = np.array(boots) if boots else np.array([[x0, k]])

    def thresholds(x0_, k_):
        return x0_ - k_ * np.log(9), x0_, x0_ + k_ * np.log(9)

    s10, s50, s90 = thresholds(x0, k)
    bs = np.array([thresholds(b[0], b[1]) for b in boots])
    err = np.std(bs, axis=0)
    # is the fitted curve actually increasing and well determined?
    monotone = k > 0 and np.isfinite(k)
    return dict(x0=x0, k=k, logS10=s10, logS50=s50, logS90=s90,
                err_logS10=float(err[0]), err_logS50=float(err[1]),
                err_logS90=float(err[2]), monotone=bool(monotone),
                n=int(len(x)), n_det=int(det.sum()))


def fit_conditional(X, det, names):
    """Logistic regression by IRLS, returning coefficients, errors and AIC."""
    X = np.column_stack([np.ones(len(det)), X])
    y = np.asarray(det, float)
    b = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ b, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-8, None)
        z = eta + (y - mu) / W
        try:
            A = X.T @ (X * W[:, None])
            bn = np.linalg.solve(A + 1e-8 * np.eye(X.shape[1]), X.T @ (W * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(bn - b)) < 1e-9:
            b = bn; break
        b = bn
    eta = np.clip(X @ b, -30, 30); mu = 1 / (1 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-8, None)
    cov = np.linalg.pinv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ll = np.sum(y * np.log(np.clip(mu, 1e-12, 1)) +
                (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1)))
    kpar = X.shape[1]
    return dict(names=["intercept"] + list(names), beta=b.tolist(), se=se.tolist(),
                z=(b / np.where(se > 0, se, np.inf)).tolist(),
                loglike=float(ll), aic=float(2 * kpar - 2 * ll),
                bic=float(kpar * np.log(len(y)) - 2 * ll))


def running_quantiles(x, y, edges):
    xs, med, lo, hi, n = [], [], [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x <= edges[i + 1]) & np.isfinite(y)
        if m.sum() < 3:
            continue
        xs.append(float(np.median(x[m]))); n.append(int(m.sum()))
        med.append(float(np.median(y[m])))
        lo.append(float(np.percentile(y[m], 16)))
        hi.append(float(np.percentile(y[m], 84)))
    return (np.array(xs), np.array(med), np.array(lo), np.array(hi), np.array(n))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_completeness(x, det, fit, edges, label, outdir, xlabel, name):
    """Thesis Figure 2.28 structure, rebuilt: distributions above, curve below."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(3.35, 4.0), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2], hspace=0.06))

    bins = np.linspace(np.min(x), np.max(x), 22)
    a1.hist(x, bins=bins, color=COL["light"], edgecolor=COL["data"],
            linewidth=0.7, label="injected")
    a1.hist(x[det], bins=bins, color=COL["data"], edgecolor="none",
            alpha=0.85, label="recovered")
    a1.set_ylabel("$N$")
    a1.set_ylim(0, a1.get_ylim()[1] * 1.42)          # headroom for the legend
    a1.legend(loc="upper left", ncol=2, handlelength=1.1,
              columnspacing=1.0, borderpad=0.2)
    finish(a1)

    xs, N, K, p, lo, hi = binned_completeness(x, det, edges)
    a2.errorbar(xs, p, yerr=yerr(p, lo, hi), fmt="o", ms=3.4,
                color=COL["data"], ecolor=COL["data"], elinewidth=0.8,
                capsize=1.8, capthick=0.8, zorder=3, label="binned, Wilson 68%")

    if fit and fit["monotone"]:
        xx = np.linspace(np.min(x), np.max(x), 400)
        a2.plot(xx, logistic(xx, fit["x0"], fit["k"]), "-", lw=1.3,
                color=COL["fit"], zorder=2, label="logistic fit")
        for lv, key, ls in ((0.5, "logS50", "--"), (0.9, "logS90", ":")):
            v = fit[key]
            if np.min(x) <= v <= np.max(x):
                a2.axvline(v, color=COL["grey"], lw=0.6, ls=ls, zorder=1)
                # near the top, where the curve and the legend are not
                a2.annotate(f"$S_{{{int(lv*100)}}}$", xy=(v, 1.02),
                            xytext=(2.5, 0), textcoords="offset points",
                            fontsize=7, color=COL["grey"], ha="left", va="top")
        a2.axhline(0.5, color=COL["grey"], lw=0.4, ls="--", alpha=0.6, zorder=0)
        a2.axhline(0.9, color=COL["grey"], lw=0.4, ls=":", alpha=0.6, zorder=0)

    a2.set_ylim(-0.03, 1.08)
    a2.set_xlabel(xlabel)
    a2.set_ylabel("completeness  $C$")
    # the curve runs lower-left to upper-right, so those corners are free
    a2.legend(loc="upper left", handlelength=1.4, borderpad=0.2,
              bbox_to_anchor=(0.0, 0.88))
    finish(a2)

    if fit and fit["monotone"]:
        txt = (rf"$\log S_{{50}} = {fit['logS50']:+.2f} \pm {fit['err_logS50']:.2f}$"
               "\n"
               rf"$\log S_{{90}} = {fit['logS90']:+.2f} \pm {fit['err_logS90']:.2f}$")
        a2.text(0.965, 0.045, txt, transform=a2.transAxes, va="bottom",
                ha="right", fontsize=7.2, color=COL["fit"])
    a1.set_title(label, pad=4, loc="left", fontsize=9)
    return save(fig, outdir, name)


def fig_flux_recovery(t, outdir, label):
    """Capture and fidelity against injected flux: what the catalogue reports."""
    det = np.asarray(t["detected"], bool)
    x = np.log10(np.asarray(t["Fint_injected_Jykms"], float))
    cap = np.asarray(t["flux_capture"], float)
    fid = np.asarray(t["flux_fidelity"], float)
    has_cap = np.isfinite(cap).sum() >= 5

    npan = 2 if has_cap else 1
    fig, axes = plt.subplots(npan, 1, figsize=(3.35, 2.1 * npan + 0.5),
                             sharex=True, gridspec_kw=dict(hspace=0.08))
    axes = np.atleast_1d(axes)
    edges = equal_count_bins(x[det], 8, 6)

    panels = ([("capture  $F_{\\rm mask}/F_{\\rm inj}$", cap, COL["alt"])] if has_cap else []) \
        + [("fidelity  $f_{\\rm sum}/F_{\\rm inj}$", fid, COL["data"])]

    for ax, (ylab, y, c) in zip(axes, panels):
        m = det & np.isfinite(y)
        ax.scatter(x[m], y[m], s=5, color=c, alpha=0.35, lw=0, zorder=2)
        xs, md, lo, hi, _ = running_quantiles(x[m], y[m], edges)
        if len(xs):
            ax.fill_between(xs, lo, hi, color=c, alpha=0.20, lw=0, zorder=1)
            ax.plot(xs, md, "-", color=c, lw=1.4, zorder=3)
        ax.axhline(1.0, color=COL["grey"], lw=0.6, ls="--", zorder=0)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(1.35, np.nanpercentile(y[m], 97.5) * 1.05) if m.any() else 1.35)
        finish(ax)

    axes[-1].set_xlabel(r"$\log_{10}\,(S_{\rm int}\,/\,{\rm Jy\,km\,s^{-1}})$")
    axes[0].set_title(label, pad=4, loc="left", fontsize=9)
    return save(fig, outdir, "flux_recovery")


def fig_selection_2d(t, outdir, label, xcol, ycol, xlabel, ylabel, name, nb=6):
    det = np.asarray(t["detected"], bool)
    x = np.log10(np.asarray(t[xcol], float))
    y = np.log10(np.asarray(t[ycol], float))
    g = np.isfinite(x) & np.isfinite(y)
    x, y, det = x[g], y[g], det[g]

    xe = equal_count_bins(x, nb, 6); ye = equal_count_bins(y, nb, 6)
    Z = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    Nn = np.zeros_like(Z)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            m = (x >= xe[j]) & (x <= xe[j + 1]) & (y >= ye[i]) & (y <= ye[i + 1])
            if m.sum() >= 4:
                Z[i, j] = det[m].mean(); Nn[i, j] = m.sum()

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    pc = ax.pcolormesh(xe, ye, Z, cmap="viridis", vmin=0, vmax=1,
                       shading="flat", edgecolors="white", linewidth=0.4)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            if np.isfinite(Z[i, j]):
                ax.text((xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2,
                        f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if Z[i, j] < 0.55 else "black")
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label("detection probability")
    cb.outline.set_linewidth(0.7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(label, pad=4, loc="left", fontsize=9)
    finish(ax)
    return save(fig, outdir, name)


def fig_conditional(t, cond, outdir, label):
    """Detection probability vs SNR at fixed linewidth: the point of the design."""
    det = np.asarray(t["detected"], bool)
    s = np.log10(np.asarray(t["snr_design"], float))
    w = np.asarray(t["W50_kms"], float)
    qs = np.percentile(w, [0, 33, 67, 100])

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    cols = [COL["data"], COL["alt"], COL["fit"]]
    for i in range(3):
        m = (w >= qs[i]) & (w <= qs[i + 1])
        if m.sum() < 12:
            continue
        e = equal_count_bins(s[m], 5, 5)
        xs, N, K, p, lo, hi = binned_completeness(s[m], det[m], e)
        ax.errorbar(xs, p, yerr=yerr(p, lo, hi), fmt="o", ms=3.2, lw=0,
                    color=cols[i], ecolor=cols[i], elinewidth=0.8, capsize=1.6,
                    label=rf"$W_{{50}} = {qs[i]:.0f}$–${qs[i+1]:.0f}$ km s$^{{-1}}$")
        f = fit_logistic(s[m], det[m], n_boot=0)
        if f["monotone"]:
            xx = np.linspace(s[m].min(), s[m].max(), 200)
            ax.plot(xx, logistic(xx, f["x0"], f["k"]), "-", lw=1.2, color=cols[i])
    ax.axhline(0.5, color=COL["grey"], lw=0.4, ls="--", zorder=0)
    ax.set_xlabel(r"$\log_{10}\,{\rm SNR_{int}}$")
    ax.set_ylabel("detection probability")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right", handlelength=1.4)
    ax.set_title(label, pad=4, loc="left", fontsize=9)
    finish(ax)
    return save(fig, outdir, "conditional_snr_w50")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recovery", required=True, nargs="+",
                    help="Recovery table(s) from temba.match; globs allowed.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-per-bin", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=400)
    a = ap.parse_args()

    paths = []
    for p in a.recovery:
        paths.extend(sorted(glob.glob(p)) or [p])
    tabs = [Table.read(p, format="ascii.ecsv") for p in paths]
    t = vstack(tabs, metadata_conflicts="silent") if len(tabs) > 1 else tabs[0]

    det = np.asarray(t["detected"], bool)
    x = np.log10(np.asarray(t["Fint_injected_Jykms"], float))
    good = np.isfinite(x)
    t, det, x = t[good], det[good], x[good]

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    use_style()
    res = dict(n_realisations=len(paths), n_injected=int(len(t)),
               n_detected=int(det.sum()))
    figs = []

    # ---- Layer 1: thesis flux completeness --------------------------------
    edges = equal_count_bins(x, a.bins, a.min_per_bin)
    fit = fit_logistic(x, det, n_boot=a.bootstrap)
    res["flux_logistic"] = fit
    figs += fig_completeness(
        x, det, fit, edges, a.label, outdir,
        r"$\log_{10}\,(S_{\rm int}\,/\,{\rm Jy\,km\,s^{-1}})$", "completeness_flux")

    # ---- flux density recovery -------------------------------------------
    for c in ("flux_capture", "flux_fidelity"):
        if c not in t.colnames:
            t[c] = np.full(len(t), np.nan)
    figs += fig_flux_recovery(t, outdir, a.label)
    for c, key in (("flux_capture", "capture"), ("flux_fidelity", "fidelity")):
        v = np.asarray(t[c], float)[det]
        v = v[np.isfinite(v)]
        res[key] = (dict(median=float(np.median(v)), p16=float(np.percentile(v, 16)),
                         p84=float(np.percentile(v, 84)), n=int(v.size))
                    if v.size else None)

    # ---- Layer 2: conditional model ---------------------------------------
    need = ["snr_design", "W50_kms", "dHI_over_beam", "sigma_chan_Jybeam"]
    if all(c in t.colnames for c in need):
        labels = [r"log SNR", r"log W50", r"log dHI/beam", r"log sigma_chan"]
        X = np.column_stack([np.log10(np.asarray(t[c], float)) for c in need])
        m = np.isfinite(X).all(axis=1)
        res["conditional"] = fit_conditional(X[m], det[m], labels)
        res["models"] = {}
        for sub, nm in (([0], "SNR"), ([0, 1], "SNR+W50"),
                        ([0, 1, 2], "SNR+W50+size"), ([0, 1, 2, 3], "all")):
            f = fit_conditional(X[m][:, sub], det[m], [labels[i] for i in sub])
            res["models"][nm] = dict(aic=f["aic"], bic=f["bic"])
        figs += fig_conditional(t[m], res["conditional"], outdir, a.label)
        figs += fig_selection_2d(t, outdir, a.label, "snr_design", "W50_kms",
                                 r"$\log_{10}\,{\rm SNR_{int}}$",
                                 r"$\log_{10}\,(W_{50}\,/\,{\rm km\,s^{-1}})$",
                                 "selection_snr_w50")
        figs += fig_selection_2d(t, outdir, a.label, "snr_design", "dHI_over_beam",
                                 r"$\log_{10}\,{\rm SNR_{int}}$",
                                 r"$\log_{10}\,(d_{\rm HI}/\theta_{\rm beam})$",
                                 "selection_snr_size")
    else:
        res["conditional"] = None
        missing = [c for c in need if c not in t.colnames]
        print(f"  (no conditional model: missing {missing} -- pre-TEMBA table)")

    # ---- report -----------------------------------------------------------
    w = 76
    print("=" * w); print(f"TEMBA SELECTION FUNCTION   {a.label}"); print("=" * w)
    print(f"  {len(paths)} realisation(s), {len(t)} injected, {int(det.sum())} detected")
    print()
    print("  FLUX COMPLETENESS (thesis prescription)")
    print("  " + "-" * (w - 4))
    for k in ("logS10", "logS50", "logS90"):
        print(f"    {k:<8s} {fit[k]:+.3f} +/- {fit['err_' + k]:.3f}")
    print(f"    S90      {10**fit['logS90']:.3f} Jy km/s      k = {fit['k']:.3f}")
    print()
    xs, N, K, p, lo, hi = binned_completeness(x, det, edges)
    print(f"    {'logS':>8} {'N':>4} {'rec':>4} {'C':>7} {'lo':>7} {'hi':>7}")
    for i in range(len(xs)):
        print(f"    {xs[i]:8.2f} {N[i]:4d} {K[i]:4d} {p[i]:7.3f} {lo[i]:7.3f} {hi[i]:7.3f}")
    print()
    print("  FLUX DENSITY RECOVERY")
    print("  " + "-" * (w - 4))
    for key, lab in (("capture", "capture  F_mask/F_inj"),
                     ("fidelity", "fidelity f_sum/F_inj")):
        r = res[key]
        print(f"    {lab:<26s} " + (f"{r['median']:.3f}  [{r['p16']:.3f}, {r['p84']:.3f}]"
                                    if r else "not measured"))
    if res.get("conditional"):
        c = res["conditional"]
        print()
        print("  CONDITIONAL DETECTION MODEL")
        print("  " + "-" * (w - 4))
        print(f"    {'term':<18}{'beta':>9}{'se':>8}{'z':>8}")
        for nm, b, s_, z in zip(c["names"], c["beta"], c["se"], c["z"]):
            print(f"    {nm:<18}{b:9.3f}{s_:8.3f}{z:8.2f}")
        print()
        print(f"    {'model':<18}{'AIC':>10}{'BIC':>10}")
        for nm, mm in res["models"].items():
            print(f"    {nm:<18}{mm['aic']:10.1f}{mm['bic']:10.1f}")
    print()
    for f in figs:
        if f.endswith(".pdf"):
            print(f"  {f}")
    (outdir / "selection.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(f"  {outdir/'selection.json'}")


if __name__ == "__main__":
    main()
