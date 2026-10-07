"""temba.survey -- survey-wide figures, every field on the same axes.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

temba.compare puts the flux completeness curves of all fields side by side.
This module adds the figures that only say something with every field at once:

  snr_completeness   completeness against flux and against SNR_int, every
                     field. If the spread in S50 were only the noise level,
                     the SNR curves would coincide; what is left over is the
                     SoFiA configuration and the fields themselves.
  snr50_vs_par       SNR50 against each field's reliability.minSNR, marked by
                     reliability.threshold and scfind.threshold.
  thresholds_noise   S50 and S90 against the field's median sigma_chan.
  size_calibration   SoFiA's 3-sigma major axis over the beam against SNR_int,
                     with the point-source curves sqrt(ln(SNR/n)/ln 2).
  spatial_residual   per field, detected minus P(detected | SNR) on a grid
                     across the field. SNR already includes the local noise,
                     so any structure here is selection that SNR does not
                     explain.
  showcase           exemplary sources chosen across all fields from the
                     stamps saved during the campaigns.

    python -m temba.survey --runs runs --fields fields --outdir analysis/survey
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

from .compare import collect, PALETTE
from .selection import (use_style, finish, save, fit_logistic, logistic,
                        binned_completeness, equal_count_bins, COL)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

PAR_KEYS = ("reliability.minSNR", "reliability.threshold", "scfind.threshold")


def par_settings(fields_dir, field):
    """The three SoFiA settings that differ most between fields, or {}."""
    try:
        import yaml
        cfg = yaml.safe_load(open(Path(fields_dir) / field / "field.yaml"))
        txt = open(cfg["science_par"]).read()
    except Exception:
        return {}
    out = {}
    for k in PAR_KEYS:
        m = re.search(r"^\s*" + re.escape(k) + r"\s*=\s*([0-9.eE+-]+)", txt, re.M)
        if m:
            out[k] = float(m.group(1))
    return out


def col(t, name):
    return (np.asarray(t[name], float) if name in t.colnames
            else np.full(len(t), np.nan))


def fit_fields(fields, n_boot, min_sources=60):
    """Flux and SNR logistic fits per field."""
    res = {}
    for f, (t, nr) in sorted(fields.items()):
        det = np.asarray(t["detected"], bool)
        snr, flux = col(t, "snr_design"), col(t, "Fint_injected_Jykms")
        g = np.isfinite(snr) & (snr > 0) & np.isfinite(flux) & (flux > 0)
        if g.sum() < min_sources:
            print(f"  skipping {f}: only {int(g.sum())} usable injections")
            continue
        xs, xf, d = np.log10(snr[g]), np.log10(flux[g]), det[g]
        res[f] = dict(t=t[g], det=d, logsnr=xs, logflux=xf, n_real=nr,
                      fit_snr=fit_logistic(xs, d, n_boot=n_boot),
                      fit_flux=fit_logistic(xf, d, n_boot=n_boot),
                      sigma_med=float(np.nanmedian(col(t, "sigma_chan_Jybeam")[g])))
    return res


def ordered(res):
    return sorted(res, key=lambda f: res[f]["fit_flux"]["logS50"])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_snr_completeness(res, outdir, bins=8):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    for i, f in enumerate(ordered(res)):
        d, c = res[f], PALETTE[i % len(PALETTE)]
        for ax, x, fit in ((axes[0], d["logflux"], d["fit_flux"]),
                           (axes[1], d["logsnr"], d["fit_snr"])):
            xs, N, K, p, lo, hi = binned_completeness(x, d["det"],
                                                      equal_count_bins(x, bins, 8))
            ax.plot(xs, p, "o", ms=2.2, color=c, alpha=0.8)
            xx = np.linspace(x.min(), x.max(), 300)
            ax.plot(xx, logistic(xx, fit["x0"], fit["k"]), color=c, lw=0.9,
                    label=f if ax is axes[0] else None)
    axes[0].set_xlabel(r"$\log_{10} S_{\rm int}$ [Jy km s$^{-1}$]")
    axes[1].set_xlabel(r"$\log_{10}\,{\rm SNR}_{\rm int}$")
    axes[0].set_ylabel("completeness")
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        finish(ax)
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    return save(fig, outdir, "snr_completeness")


def fig_snr50_vs_par(res, pars, outdir):
    have = [f for f in res if pars.get(f, {}).get("reliability.minSNR") is not None]
    if not have:
        print("  (no snr50_vs_par figure: par files not found)")
        return []
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    for f in have:
        p, fit = pars[f], res[f]["fit_snr"]
        rel = p.get("reliability.threshold", np.nan)
        scf = p.get("scfind.threshold", np.nan)
        marker = "s" if rel >= 0.85 else "o"
        face = "none" if scf >= 3.5 else COL["data"]
        ax.errorbar(p["reliability.minSNR"], fit["logS50"], yerr=fit["err_logS50"],
                    fmt=marker, ms=4.5, mfc=face, mec=COL["data"], ecolor=COL["data"],
                    elinewidth=0.8, capsize=1.8, lw=0)
        ax.annotate(f, (p["reliability.minSNR"], fit["logS50"]), xytext=(3, 3),
                    textcoords="offset points", fontsize=5.5, color=COL["grey"])
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=COL["data"], label="rel. threshold 0.75"),
        Line2D([], [], marker="s", ls="", color=COL["data"], label="rel. threshold 0.90"),
        Line2D([], [], marker="o", ls="", mfc="none", color=COL["data"],
               label="scfind threshold 4.0")], fontsize=6, frameon=False, loc="best")
    ax.set_xlabel(r"reliability.minSNR")
    ax.set_ylabel(r"$\log_{10}\,{\rm SNR}_{50}$")
    finish(ax)
    fig.tight_layout()
    return save(fig, outdir, "snr50_vs_par")


def fig_thresholds_noise(res, outdir):
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    sig = np.array([res[f]["sigma_med"] for f in res]) * 1e3      # mJy/beam
    for key, err, m, lab in (("logS50", "err_logS50", "o", "$S_{50}$"),
                             ("logS90", "err_logS90", "s", "$S_{90}$")):
        y = np.array([res[f]["fit_flux"][key] for f in res])
        e = np.array([res[f]["fit_flux"].get(err, np.nan) for f in res])
        ax.errorbar(np.log10(sig), y, yerr=e, fmt=m, ms=4, lw=0, label=lab,
                    color=COL["data"] if m == "o" else COL["fit"],
                    elinewidth=0.8, capsize=1.8)
        off = np.nanmedian(y - np.log10(sig))
        xx = np.linspace(np.log10(sig).min() - 0.05, np.log10(sig).max() + 0.05, 2)
        ax.plot(xx, xx + off, ls="--", lw=0.7,
                color=COL["data"] if m == "o" else COL["fit"])
    for f, x, yv in zip(res, np.log10(sig), [res[f]["fit_flux"]["logS50"] for f in res]):
        ax.annotate(f, (x, yv), xytext=(3, -8), textcoords="offset points",
                    fontsize=5.0, color=COL["grey"])
    ax.set_xlabel(r"$\log_{10}$ median $\sigma_{\rm chan}$ [mJy beam$^{-1}$]")
    ax.set_ylabel(r"$\log_{10} S$ [Jy km s$^{-1}$]")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              title="dashed: proportional to noise", title_fontsize=5.5)
    finish(ax)
    fig.tight_layout()
    return save(fig, outdir, "thresholds_vs_noise")


def point_source_size(snr, n):
    """d_nsigma / theta for an unresolved source: sqrt(ln(SNR/n) / ln 2)."""
    snr = np.asarray(snr, float)
    return np.sqrt(np.log(np.clip(snr / n, 1.0, None)) / np.log(2.0))


def fig_size_calibration(res, outdir):
    xs, ys, cs = [], [], []
    for f in res:
        t, det = res[f]["t"], res[f]["det"]
        e3 = col(t, "ell3s_maj_sofia_arcsec")
        beam = col(t, "dHI_arcsec") / col(t, "dHI_over_beam")
        g = det & np.isfinite(e3) & np.isfinite(beam) & (beam > 0)
        xs.append(col(t, "snr_design")[g]); ys.append((e3 / beam)[g])
        cs.append(np.log10(col(t, "dHI_over_beam")[g]))
    if not xs or not sum(len(x) for x in xs):
        print("  (no size_calibration figure: no SoFiA sizes)")
        return []
    x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    sc = ax.scatter(x, y, c=c, s=1.5, cmap="viridis", alpha=0.5, lw=0,
                    rasterized=True, vmin=np.log10(0.2), vmax=np.log10(4.0))
    xx = np.geomspace(max(1.01, np.nanmin(x)), np.nanmax(x), 300)
    for n, ls in ((1, ":"), (2, "-."), (3, "-")):
        m = xx > n
        ax.plot(xx[m], point_source_size(xx[m], n), color="k", lw=0.9, ls=ls,
                label=fr"point source, {n}$\sigma$")
    ax.set_xscale("log")
    ax.set_ylim(0, min(6.0, np.nanpercentile(y, 99.5) * 1.1))
    ax.set_xlabel(r"${\rm SNR}_{\rm int}$")
    ax.set_ylabel(r"SoFiA $3\sigma$ major axis / $\theta_{\rm beam}$")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(r"$\log_{10}\, d_{\rm HI}/\theta_{\rm beam}$", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.tight_layout()
    return save(fig, outdir, "size_calibration")


def spatial_maps(res, nb=8, min_n=12):
    out = {}
    for f in res:
        d, fit = res[f], res[f]["fit_snr"]
        t = d["t"]
        x, y = col(t, "xpix"), col(t, "ypix")
        g = np.isfinite(x) & np.isfinite(y)
        if g.sum() < nb * nb:
            continue
        p = logistic(d["logsnr"], fit["x0"], fit["k"])
        r = d["det"].astype(float) - p
        ex = np.linspace(np.nanmin(x[g]), np.nanmax(x[g]), nb + 1)
        ey = np.linspace(np.nanmin(y[g]), np.nanmax(y[g]), nb + 1)
        ix = np.clip(np.digitize(x[g], ex) - 1, 0, nb - 1)
        iy = np.clip(np.digitize(y[g], ey) - 1, 0, nb - 1)
        mean = np.full((nb, nb), np.nan); expect = np.full((nb, nb), np.nan)
        for j in range(nb):
            for i in range(nb):
                m = (ix == i) & (iy == j)
                if m.sum() >= min_n:
                    mean[j, i] = r[g][m].mean()
                    expect[j, i] = np.sqrt(np.mean(p[g][m] * (1 - p[g][m])) / m.sum())
        ok = np.isfinite(mean)
        chi2 = float(np.sum((mean[ok] / expect[ok]) ** 2)) if ok.any() else np.nan
        out[f] = dict(mean=mean, extent=[ex[0], ex[-1], ey[0], ey[-1]],
                      chi2=chi2, dof=int(ok.sum()))
    return out


def fig_spatial_residual(maps, outdir):
    if not maps:
        return []
    fs = list(maps)
    ncol = 4
    nrow = int(np.ceil(len(fs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.9, 1.85 * nrow), squeeze=False)
    im = None
    for ax, f in zip(axes.flat, fs):
        m = maps[f]
        im = ax.imshow(m["mean"], origin="lower", extent=m["extent"], cmap="RdBu_r",
                       vmin=-0.3, vmax=0.3, interpolation="nearest", aspect="auto")
        ax.set_title(f"{f}\n" + r"$\chi^2$" + f" {m['chi2']:.0f} / {m['dof']}",
                     fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(fs):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label(r"detected $-$ $P$(detected | SNR)", fontsize=7)
    return save(fig, outdir, "spatial_residual")


def fig_showcase(runs, outdir, n=6):
    from .showcase import choose, panel
    stamps = []
    for p in sorted(glob.glob(str(Path(runs) / "*" / "run_*" / "products" / "stamps.npz"))):
        field = Path(p).parts[len(Path(runs).parts)]
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            d = json.loads(str(z[k]))
            d["field"] = field
            stamps.append(d)
    stamps = [d for d in stamps if not d.get("neighbour")]
    if not stamps:
        print("  (no showcase: no stamps.npz files)")
        return []
    t = Table(dict(detected=np.ones(len(stamps), bool),
                   flux_capture=[d["capture"] for d in stamps],
                   flux_fidelity=[d["fidelity"] for d in stamps],
                   snr_design=[d["snr"] for d in stamps],
                   dHI_over_beam=[d["dratio"] for d in stamps]))
    ids, why = choose(t, n)
    chosen = []
    for i, w in zip(ids, why):
        d = dict(stamps[i]); d["label"] = f"{d['field']}\n{w}"
        chosen.append(d)
    chosen.sort(key=lambda d: -d["capture"])
    fig = plt.figure(figsize=(11.5, 2.5 * len(chosen)))
    gs = fig.add_gridspec(len(chosen), 4, width_ratios=[1, 1, 1, 1.25],
                          hspace=0.12, wspace=0.06)
    for r, d in enumerate(chosen):
        panel(fig, gs, r, d)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("pdf", 200), ("png", 140)):
        q = outdir / f"showcase_all_fields.{ext}"
        fig.savefig(q, bbox_inches="tight", dpi=dpi)
        paths.append(str(q))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fields", default="fields",
                    help="directory of field.yaml files, for the par settings")
    ap.add_argument("--outdir", default="analysis/survey")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--showcase", type=int, default=6)
    a = ap.parse_args()

    fields = collect(a.runs)
    if not fields:
        raise SystemExit(f"No recovery tables under {a.runs}/*/run_*/products/")
    use_style()
    res = fit_fields(fields, a.bootstrap)
    pars = {f: par_settings(a.fields, f) for f in res}
    maps = spatial_maps(res)

    figs = []
    figs += fig_snr_completeness(res, a.outdir)
    figs += fig_snr50_vs_par(res, pars, a.outdir)
    figs += fig_thresholds_noise(res, a.outdir)
    figs += fig_size_calibration(res, a.outdir)
    figs += fig_spatial_residual(maps, a.outdir)
    if a.showcase:
        figs += fig_showcase(a.runs, a.outdir, a.showcase)

    w = 104
    print("=" * w)
    print(f"TEMBA SURVEY   {len(res)} field(s)")
    print("=" * w)
    print(f"  {'field':<14}{'runs':>5}{'N':>6}{'logS50':>8}{'logSNR50':>10}{'+-':>6}"
          f"{'k_snr':>7}{'sigma mJy':>10}{'minSNR':>8}{'rel':>6}{'scfind':>7}"
          f"{'spatial chi2/dof':>18}")
    print("  " + "-" * (w - 4))
    for f in ordered(res):
        d, p = res[f], pars.get(f, {})
        fs = d["fit_snr"]
        m = maps.get(f)
        sp = f"{m['chi2']:.0f}/{m['dof']}" if m else "-"
        g = lambda k: f"{p[k]:.2f}" if k in p else "-"
        print(f"  {f:<14}{d['n_real']:>5}{len(d['t']):>6}{d['fit_flux']['logS50']:>8.2f}"
              f"{fs['logS50']:>10.2f}{fs['err_logS50']:>6.2f}{fs['k']:>7.2f}"
              f"{1e3 * d['sigma_med']:>10.3f}{g('reliability.minSNR'):>8}"
              f"{g('reliability.threshold'):>6}{g('scfind.threshold'):>7}{sp:>18}")
    lf = np.array([res[f]["fit_flux"]["logS50"] for f in res])
    ls = np.array([res[f]["fit_snr"]["logS50"] for f in res])
    if len(res) > 1:
        print()
        print(f"  S50 spread:    {10 ** (lf.max() - lf.min()):.2f}x   (std {lf.std():.3f} dex)")
        print(f"  SNR50 spread:  {10 ** (ls.max() - ls.min()):.2f}x   (std {ls.std():.3f} dex)")
    print()
    for p in figs:
        if p.endswith(".pdf"):
            print(f"  {p}")
    out = {f: dict(fit_flux=res[f]["fit_flux"], fit_snr=res[f]["fit_snr"],
                   sigma_med_Jybeam=res[f]["sigma_med"], n_real=res[f]["n_real"],
                   par=pars.get(f, {}),
                   spatial_chi2=maps.get(f, {}).get("chi2"),
                   spatial_dof=maps.get(f, {}).get("dof"))
           for f in res}
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    (Path(a.outdir) / "survey.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"  {Path(a.outdir) / 'survey.json'}")


if __name__ == "__main__":
    main()
