"""Per-source diagnostic panels for TEMBA.

The population-level figures say how often a source is recovered and how much
of its flux comes back. They do not show what recovery looks like, and a
reader who wants to trust the flux decomposition needs to see a case where the
mask truncates a disc, not just a median.

This module makes that figure. For a handful of detected sources it shows,
side by side:

    injected moment-0, with the d_HI, d_1sigma and d_3sigma isophotes
    recovered moment-0, restricted to the SoFiA mask, with ell3s_maj
    the residual, injected minus recovered, on a symmetric scale
    the two global profiles on shared axes

The noiseless model needs no saving: the injected cube is the substrate plus
the model, so the model is exactly their difference over the cutout. Both
files exist when match runs, which is why the stamps step belongs there and
not after --cleanup has removed them.

Two steps, because the cubes are large and transient while the stamps are
small and permanent:

    python -m temba.showcase stamps --run runs/F/run_0001 --injected ... \\
        --substrate ... --mask ... --recovery ... --n 6
    python -m temba.showcase plot --stamps runs/F/run_0001/products/stamps.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import physics as P

COL = dict(inj="#2a78d6", rec="#1baf7a", res="#eb6834", grey="#898781",
           iso1="#eda100", iso2="#4ea6c8", iso3="#e34948", ishi="#6250d6")


def _has_neighbour(mom0, frac=0.25):
    """True if a second peak of at least `frac` of the central one is present.

    Looks outside the central third of the cutout, where the injected source
    sits by construction, so anything bright out there came from somewhere
    else.
    """
    m = np.asarray(mom0, float)
    ny, nx = m.shape
    cy, cx = ny // 2, nx // 2
    h = max(ny, nx) // 6
    outer = m.copy()
    outer[cy - h:cy + h + 1, cx - h:cx + h + 1] = 0.0
    pk = float(np.nanmax(m[cy - h:cy + h + 1, cx - h:cx + h + 1]))
    return pk > 0 and float(np.nanmax(outer)) > frac * pk


# ---------------------------------------------------------------- selection
def choose(rec, n=6):
    """Pick sources that show different things, not the n brightest.

    A figure of six well-recovered discs says nothing a median could not. The
    interesting cases are the ones where the three quantities disagree: where
    the mask truncates the source, where it grows into the noise, and where
    detection is marginal.
    """
    det = np.asarray(rec["detected"], bool) if "detected" in rec.colnames \
        else np.ones(len(rec), bool)
    rows = np.where(det)[0]                 # index into the ORIGINAL table
    t = rec[det]
    keep, why = [], []

    def take(mask, label):
        for i in np.where(mask)[0]:
            if int(rows[i]) not in keep:
                keep.append(int(rows[i])); why.append(label); return

    cap = np.asarray(t["flux_capture"], float)
    fid = np.asarray(t["flux_fidelity"], float)
    snr = np.asarray(t["snr_design"], float)
    dr = np.asarray(t["dHI_over_beam"], float)
    fin = np.isfinite(cap) & np.isfinite(fid)

    take(fin & (cap > 0.98) & (dr > 1.5), "well resolved, fully captured")
    take(fin & (cap == np.nanmin(np.where(fin, cap, np.nan))), "worst mask truncation")
    take(fin & (fid == np.nanmax(np.where(fin, fid, np.nan))), "most noise in mask")
    take(fin & (snr < np.nanpercentile(snr, 15)), "near the detection threshold")
    take(fin & (dr < 0.5), "unresolved")
    take(fin & (np.abs(cap - np.nanmedian(cap)) < 0.005), "typical")
    while len(keep) < n and len(keep) < len(t):
        for i in np.argsort(-snr):
            if int(rows[i]) not in keep:
                keep.append(int(rows[i])); why.append("bright"); break
        else:
            break
    return keep[:n], why[:n]


# -------------------------------------------------------------------- stamps
def cut(cube, x, y, half, zlo, zhi):
    x0, x1 = max(x - half, 0), min(x + half + 1, cube.shape[2])
    y0, y1 = max(y - half, 0), min(y + half + 1, cube.shape[1])
    return np.asarray(cube[zlo:zhi, y0:y1, x0:x1], float), (x0, y0)


def stamps(a):
    rec = Table.read(a.recovery, format="ascii.ecsv")
    ids, why = choose(rec, a.n)
    print(f"  selected {len(ids)} of {len(rec)} rows")

    inj = fits.open(a.injected, memmap=True)[0]
    sub = fits.getdata(a.substrate, memmap=True)
    msk = fits.getdata(a.mask, memmap=True)
    hdr = inj.header
    w = WCS(hdr)
    pix = abs(hdr["CDELT1"]) * 3600.0
    bmaj = hdr.get("BMAJ", 0.0) * 3600.0
    dv = abs(hdr["CDELT3"]) / (1e3 if abs(hdr["CDELT3"]) > 100 else 1.0)
    barea = 1.133 * hdr["BMAJ"] * hdr["BMIN"] / (abs(hdr["CDELT1"]) *
                                                 abs(hdr["CDELT2"]))

    out = {}
    for sid, lab in zip(ids, why):
        r = rec[sid]                        # sid is a row index
        x, y = int(round(float(r["xpix"]))), int(round(float(r["ypix"])))
        nz = inj.data.shape[0]
        if "zpix" in rec.colnames:
            zc = int(np.clip(round(float(r["zpix"])), 0, nz - 1))
        elif "vsys_kms" in rec.colnames:
            vax = (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"] / 1e3 \
                + hdr["CRVAL3"] / 1e3
            zc = int(np.argmin(np.abs(vax - float(r["vsys_kms"]))))
        else:
            zc = nz // 2
        w20 = float(r["W20_kms"]) if "W20_kms" in rec.colnames \
            else 1.4 * float(r["W50_kms"])
        halfz = int(np.ceil(0.75 * w20 / dv)) + 3
        zlo, zhi = max(zc - halfz, 0), min(zc + halfz + 1, nz)

        d_hi = float(r["dHI_arcsec"]) if "dHI_arcsec" in rec.colnames \
            else float(r["dHI_over_beam"]) * bmaj
        half = int(np.ceil((1.2 * d_hi + 2.5 * bmaj) / pix / 2.0))
        half = int(np.clip(half, 12, 90))

        ci, (x0, y0) = cut(inj.data, x, y, half, zlo, zhi)
        cs, _ = cut(sub, x, y, half, zlo, zhi)
        cm, _ = cut(msk, x, y, half, zlo, zhi)
        model = ci - cs                       # exactly the noiseless model
        inmask = cm > 0

        out[f"s{sid}"] = dict(
            row=int(sid), label=lab,
            model_mom0=model.sum(axis=0) * dv,
            recov_mom0=np.where(inmask, ci, 0.0).sum(axis=0) * dv,
            resid_mom0=(model - np.where(inmask, ci, 0.0)).sum(axis=0) * dv,
            spec_inj=model.sum(axis=(1, 2)) / barea,
            spec_rec=np.where(inmask, ci, 0.0).sum(axis=(1, 2)) / barea,
            vel=(np.arange(zlo, zhi) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"] / 1e3
                + hdr["CRVAL3"] / 1e3,
            sigma_chan=float(r["sigma_chan_Jybeam"]),
            nchan=int(zhi - zlo), pix=pix, bmaj=bmaj, dv=dv,
            snr=float(r["snr_design"]), w50=float(r["W50_kms"]),
            dHI=d_hi, dratio=float(r["dHI_over_beam"]),
            capture=float(r["flux_capture"]), fidelity=float(r["flux_fidelity"]),
            # A second source in the cutout muddies the residual; say so on the
            # panel rather than silently letting the reader puzzle over it.
            neighbour=bool(_has_neighbour(model.sum(axis=0))),
            ell3s=float(r["ell3s_maj_sofia_arcsec"])
            if "ell3s_maj_sofia_arcsec" in rec.colnames else float("nan"),
        )

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: json.dumps({kk: (vv.tolist()
                                                  if isinstance(vv, np.ndarray)
                                                  else vv)
                                             for kk, vv in v.items()})
                              for k, v in out.items()})
    print(f"  wrote {p}")
    return p


# --------------------------------------------------------------------- plot
def iso_contour(ax, mom0, level, pix, colour, label, ls="-"):
    if not np.isfinite(level) or level <= 0:
        return
    ny, nx = mom0.shape
    ext = [-nx / 2 * pix, nx / 2 * pix, -ny / 2 * pix, ny / 2 * pix]
    xx = (np.arange(nx) - nx / 2 + 0.5) * pix
    yy = (np.arange(ny) - ny / 2 + 0.5) * pix
    ax.contour(xx, yy, mom0, levels=[level], colors=[colour],
               linewidths=1.0, linestyles=[ls])


def panel(fig, gs, row, d):
    pix, bmaj = d["pix"], d["bmaj"]
    m0, r0, res = (np.array(d["model_mom0"]), np.array(d["recov_mom0"]),
                   np.array(d["resid_mom0"]))
    ny, nx = m0.shape
    ext = [-nx / 2 * pix, nx / 2 * pix, -ny / 2 * pix, ny / 2 * pix]
    vmax = np.nanmax(np.abs(m0)) or 1.0
    sig0 = d["sigma_chan"] * d["dv"] * np.sqrt(max(d["nchan"], 1))

    for j, (img, ttl, cmap, lim) in enumerate((
            (m0, "injected", "viridis", (0, vmax)),
            (r0, "recovered (in mask)\nsame scale, includes mask noise", "viridis", (0, vmax)),
            (res, "residual", "coolwarm", (-0.5 * vmax, 0.5 * vmax)))):
        ax = fig.add_subplot(gs[row, j])
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap,
                  vmin=lim[0], vmax=lim[1], interpolation="nearest")
        if j == 0:
            for ns, c, ls in ((1.0, COL["iso1"], ":"), (2.0, COL["iso2"], "-."),
                              (3.0, COL["iso3"], "-")):
                iso_contour(ax, img, ns * sig0, pix, c, "", ls)
            th = np.linspace(0, 2 * np.pi, 180)
            ax.plot(0.5 * d["dHI"] * np.cos(th), 0.5 * d["dHI"] * np.sin(th),
                    color=COL["ishi"], lw=1.2, ls="--")
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            if row == 0:
                from matplotlib.lines import Line2D
                ax.legend(handles=[
                    Line2D([], [], color=COL["ishi"], ls="--", lw=1.2,
                           label="$d_{\\rm HI}$"),
                    Line2D([], [], color=COL["iso1"], ls=":", lw=1.0,
                           label="$1\\sigma$"),
                    Line2D([], [], color=COL["iso2"], ls="-.", lw=1.0,
                           label="$2\\sigma$"),
                    Line2D([], [], color=COL["iso3"], ls="-", lw=1.0,
                           label="$3\\sigma$")],
                    fontsize=6, frameon=False, labelcolor="w",
                    loc="lower right", handlelength=1.4, borderpad=0.2)
        if j == 1 and np.isfinite(d["ell3s"]):
            th = np.linspace(0, 2 * np.pi, 180)
            ax.plot(0.5 * d["ell3s"] * np.cos(th), 0.5 * d["ell3s"] * np.sin(th),
                    color="w", lw=1.0)
        ax.add_patch(plt.Circle((ext[0] + 1.3 * bmaj, ext[2] + 1.3 * bmaj),
                                0.5 * bmaj, fc="none", ec="w", lw=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(ttl, fontsize=9, pad=3)

    ax = fig.add_subplot(gs[row, 3])
    v = np.array(d["vel"])
    if d.get("neighbour"):
        ax.text(0.98, 0.02, "neighbour in cutout", transform=ax.transAxes,
                fontsize=5.5, ha="right", va="bottom", color=COL["res"])
    ax.plot(v, np.array(d["spec_inj"]) * 1e3, color=COL["inj"], lw=1.4,
            label="injected")
    ax.plot(v, np.array(d["spec_rec"]) * 1e3, color=COL["rec"], lw=1.4,
            ls="--", label="recovered")
    ax.axhline(0, color=COL["grey"], lw=0.5)
    ax.set_xlabel("$v$ [km s$^{-1}$]", fontsize=8)
    ax.set_ylabel("mJy", fontsize=8)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if row == 0:
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.text(0.02, 0.96,
            f"{d['label']}\nSNR {d['snr']:.1f}   $W_{{50}}$ {d['w50']:.0f}\n"
            f"$d/\\theta$ {d['dratio']:.2f}\n"
            f"cap {d['capture']:.3f}  fid {d['fidelity']:.3f}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color=COL["grey"])


def plot(a):
    z = np.load(a.stamps, allow_pickle=False)
    ds = [json.loads(str(z[k])) for k in z.files]
    # Best to worst capture: the figure then reads as a progression from a
    # source recovered whole to one the mask cuts in half, which is the point
    # being made.
    ds.sort(key=lambda d: -d["capture"])
    n = len(ds)
    fig = plt.figure(figsize=(11.5, 2.5 * n))
    gs = fig.add_gridspec(n, 4, width_ratios=[1, 1, 1, 1.25],
                          hspace=0.12, wspace=0.06)
    for i, d in enumerate(ds):
        panel(fig, gs, i, d)
    if a.label:
        fig.suptitle(a.label, x=0.01, ha="left", fontsize=10)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=140)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stamps", help="Extract cutouts before --cleanup.")
    s.add_argument("--recovery", required=True)
    s.add_argument("--injected", required=True)
    s.add_argument("--substrate", required=True)
    s.add_argument("--mask", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=6)

    p_ = sub.add_parser("plot", help="Draw the panels from a stamps file.")
    p_.add_argument("--stamps", required=True)
    p_.add_argument("--out", required=True)
    p_.add_argument("--label", default="")

    a = ap.parse_args()
    (stamps if a.cmd == "stamps" else plot)(a)


if __name__ == "__main__":
    main()
