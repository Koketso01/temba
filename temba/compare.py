"""
temba.compare -- put every field on the same axes.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

Reads the combined recovery tables for several fields and produces the
cross-field products:

  * every completeness curve C(S_int) on one set of axes, with its fitted
    logistic, so the field-to-field spread is visible at a glance
  * S10 / S50 / S90 per field with bootstrap errors, ordered by S50
  * flux capture and fidelity per field
  * detection and recovery against channel width, which is the one covariate
    that cannot vary within a single field and only becomes measurable here

The thesis made exactly this point with its own Table 2.2: the fitted S90
varied substantially between fields rather than behaving like one survey-wide
limit. This module is the machinery for saying that quantitatively, and for
extending it from a flux threshold to the recovered flux density itself.

Usage
-----
    python -m temba.compare --runs runs --outdir analysis/all_fields

    python -m temba.compare --runs runs --only Abell-194 Abell-168 \
        --outdir analysis/subset
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
                             binned_completeness, fit_logistic, logistic, yerr)


PALETTE = ["#1b3a5c", "#c1440e", "#3d6b35", "#7b5aa6", "#b8860b", "#2a7f7f",
           "#a03050", "#4a6fa5", "#8a6d3b", "#5b8c5a", "#96442e", "#4f4f7a"]


def collect(runs, only=None, pattern="*/run_*/products/recovery.ecsv"):
    """One combined table per field, keyed by field directory name."""
    out = {}
    for p in sorted(glob.glob(str(Path(runs) / pattern))):
        field = Path(p).parts[len(Path(runs).parts)]
        if only and field not in only:
            continue
        out.setdefault(field, []).append(p)
    fields = {}
    for f, paths in out.items():
        tabs = [Table.read(p, format="ascii.ecsv") for p in paths]
        t = vstack(tabs, metadata_conflicts="silent") if len(tabs) > 1 else tabs[0]
        fields[f] = (t, len(paths))
    return fields


def channel_width(t):
    """Channel width for a field, from the population metadata if present."""
    m = (t.meta or {}).get("temba", {})
    if isinstance(m, dict) and "dv_chan_kms" in m:
        return float(m["dv_chan_kms"])
    return float("nan")


# ---------------------------------------------------------------------------

def fig_all_curves(fits, outdir, bootstrap):
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    xx = np.linspace(-2.6, 1.4, 400)
    for i, (f, d) in enumerate(sorted(fits.items(), key=lambda kv: kv[1]["fit"]["logS50"])):
        c = PALETTE[i % len(PALETTE)]
        fit = d["fit"]
        ax.plot(xx, logistic(xx, fit["x0"], fit["k"]), "-", lw=1.3, color=c,
                label=f"{f}  ($n$={fit['n']})")
        ax.errorbar(d["xs"], d["p"], yerr=yerr(d["p"], d["lo"], d["hi"]),
                    fmt="o", ms=2.6, lw=0, color=c, ecolor=c, elinewidth=0.6,
                    capsize=1.2, alpha=0.75)
        ax.plot([fit["logS50"]], [0.5], "|", ms=7, color=c, mew=1.4)
    ax.axhline(0.5, color=COL["grey"], lw=0.4, ls="--", zorder=0)
    ax.axhline(0.9, color=COL["grey"], lw=0.4, ls=":", zorder=0)
    ax.set_xlabel(r"$\log_{10}\,(S_{\rm int}\,/\,{\rm Jy\,km\,s^{-1}})$")
    ax.set_ylabel("completeness  $C$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), handlelength=1.4)
    finish(ax)
    return save(fig, outdir, "completeness_all_fields")


def fig_thresholds(fits, outdir):
    order = sorted(fits.items(), key=lambda kv: kv[1]["fit"]["logS50"])
    names = [f for f, _ in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(3.45, 0.26 * len(order) + 1.1))
    for key, mk, lab, c in (("logS10", "v", r"$S_{10}$", COL["band"]),
                            ("logS50", "o", r"$S_{50}$", COL["data"]),
                            ("logS90", "^", r"$S_{90}$", COL["fit"])):
        v = np.array([d["fit"][key] for _, d in order])
        e = np.array([d["fit"]["err_" + key] for _, d in order])
        ax.errorbar(v, y, xerr=e, fmt=mk, ms=3.4, lw=0, color=c, ecolor=c,
                    elinewidth=0.8, capsize=1.6, label=lab)
    for i, (_, d) in enumerate(order):
        ax.plot([d["fit"]["logS10"], d["fit"]["logS90"]], [i, i], "-",
                color=COL["grey"], lw=0.6, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(r"$\log_{10}\,(S\,/\,{\rm Jy\,km\,s^{-1}})$")
    ax.legend(loc="lower right", handlelength=1.2, ncol=3, columnspacing=0.9)
    finish(ax)
    return save(fig, outdir, "thresholds_all_fields")


def fig_recovery_by_field(fits, outdir):
    order = sorted(fits.items(), key=lambda kv: kv[1]["fit"]["logS50"])
    names = [f for f, _ in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(3.45, 0.26 * len(order) + 1.1))
    for key, mk, lab, c, off in (("capture", "o", "capture", COL["alt"], -0.14),
                                 ("fidelity", "s", "fidelity", COL["data"], 0.14)):
        md = np.array([d[key]["median"] if d[key] else np.nan for _, d in order])
        lo = np.array([d[key]["p16"] if d[key] else np.nan for _, d in order])
        hi = np.array([d[key]["p84"] if d[key] else np.nan for _, d in order])
        ax.errorbar(md, y + off, xerr=[md - lo, hi - md], fmt=mk, ms=3.4, lw=0,
                    color=c, ecolor=c, elinewidth=0.8, capsize=1.6, label=lab)
    ax.axvline(1.0, color=COL["grey"], lw=0.6, ls="--", zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("flux recovered / flux injected")
    ax.legend(loc="lower right", handlelength=1.2, ncol=2)
    finish(ax)
    return save(fig, outdir, "flux_recovery_all_fields")


def fig_vs_channel(fits, outdir):
    """Channel width is constant within a field, so it is only visible here."""
    dv = np.array([d["dv_kms"] for d in fits.values()])
    if not np.isfinite(dv).any() or np.nanstd(dv) < 0.05:     # header rounding only
        print("  (no channel-width figure: all fields share the same dv)")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
    s50 = np.array([d["fit"]["logS50"] for d in fits.values()])
    e50 = np.array([d["fit"]["err_logS50"] for d in fits.values()])
    cap = np.array([d["capture"]["median"] if d["capture"] else np.nan
                    for d in fits.values()])
    for ax, v, e, lab in ((axes[0], s50, e50, r"$\log_{10}S_{50}$"),
                          (axes[1], cap, None, "median capture")):
        ax.errorbar(dv, v, yerr=e, fmt="o", ms=4, lw=0, color=COL["data"],
                    ecolor=COL["data"], elinewidth=0.8, capsize=1.8)
        for f, x, yv in zip(fits, dv, v):
            ax.annotate(f, (x, yv), xytext=(3, 3), textcoords="offset points",
                        fontsize=6, color=COL["grey"])
        ax.set_xlabel(r"channel width [km s$^{-1}$]")
        ax.set_ylabel(lab)
        finish(ax)
    fig.tight_layout()
    return save(fig, outdir, "vs_channel_width")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--min-sources", type=int, default=60,
                    help="Skip a field with fewer injections than this.")
    a = ap.parse_args()

    fields = collect(a.runs, a.only)
    if not fields:
        raise SystemExit(f"No recovery tables under {a.runs}/*/run_*/products/")

    use_style()
    fits = {}
    for f, (t, nr) in sorted(fields.items()):
        det = np.asarray(t["detected"], bool)
        x = np.log10(np.asarray(t["Fint_injected_Jykms"], float))
        g = np.isfinite(x)
        t, det, x = t[g], det[g], x[g]
        if len(t) < a.min_sources:
            print(f"  skipping {f}: only {len(t)} injections")
            continue
        edges = equal_count_bins(x, a.bins, 8)
        xs, N, K, p, lo, hi = binned_completeness(x, det, edges)
        fit = fit_logistic(x, det, n_boot=a.bootstrap)

        def q(c):
            if c not in t.colnames:
                return None
            v = np.asarray(t[c], float)[det]
            v = v[np.isfinite(v)]
            return dict(median=float(np.median(v)), p16=float(np.percentile(v, 16)),
                        p84=float(np.percentile(v, 84)), n=int(v.size)) if v.size else None

        fits[f] = dict(fit=fit, xs=xs, p=p, lo=lo, hi=hi,
                       n_realisations=nr, n_injected=int(len(t)),
                       n_detected=int(det.sum()), dv_kms=channel_width(t),
                       capture=q("flux_capture"), fidelity=q("flux_fidelity"))

    figs = []
    figs += fig_all_curves(fits, a.outdir, a.bootstrap)
    figs += fig_thresholds(fits, a.outdir)
    figs += fig_recovery_by_field(fits, a.outdir)
    figs += fig_vs_channel(fits, a.outdir)

    w = 96
    print("=" * w)
    print(f"TEMBA CROSS-FIELD COMPARISON   {len(fits)} field(s)")
    print("=" * w)
    print(f"  {'field':<16}{'runs':>5}{'N':>7}{'det':>6}"
          f"{'logS10':>9}{'logS50':>9}{'logS90':>9}{'S90':>9}"
          f"{'capture':>9}{'fidelity':>10}")
    print("  " + "-" * (w - 4))
    for f, d in sorted(fits.items(), key=lambda kv: kv[1]["fit"]["logS50"]):
        fit = d["fit"]
        cap = f"{d['capture']['median']:.3f}" if d["capture"] else "     -"
        fid = f"{d['fidelity']['median']:.3f}" if d["fidelity"] else "     -"
        print(f"  {f:<16}{d['n_realisations']:>5}{d['n_injected']:>7}"
              f"{d['n_detected']:>6}{fit['logS10']:>9.2f}{fit['logS50']:>9.2f}"
              f"{fit['logS90']:>9.2f}{10**fit['logS90']:>9.3f}{cap:>9}{fid:>10}")

    s50 = np.array([d["fit"]["logS50"] for d in fits.values()])
    if len(s50) > 1:
        print()
        print(f"  logS50 spread across fields: {s50.min():.2f} to {s50.max():.2f} "
              f"({10**s50.max()/10**s50.min():.1f}x in flux)")
        print("  A single survey-wide flux limit would show no spread here.")

    print()
    for p in figs:
        if p.endswith(".pdf"):
            print(f"  {p}")
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    (Path(a.outdir) / "compare.json").write_text(
        json.dumps({f: {k: v for k, v in d.items()
                        if k not in ("xs", "p", "lo", "hi")}
                    for f, d in fits.items()}, indent=2, default=str) + "\n")
    print(f"  {Path(a.outdir)/'compare.json'}")


if __name__ == "__main__":
    main()
