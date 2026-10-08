"""temba.examiner -- numerical checks behind the answers to the thesis examiners.

    python -m temba.examiner --runs runs --fields fields

Prints three tables:

  plateau   Does completeness reach 1 at high SNR? A three-parameter logistic,
            C = A / (1 + exp(-(x - x0)/k)), is fitted in log SNR alongside the
            usual two-parameter one (A = 1). A < 1 with a significant likelihood
            gain means the curve levels off below one, so a plain logistic would
            misplace S90. For the sources missed at SNR > 30 it reports where
            they sit in observed frequency.
  chance    How often could an undetected injected source be "recovered" by an
            unrelated catalogue source falling inside its matching gate? The
            expected rate is N_cat x gate volume / searched volume, with the
            gate 2 beams on the sky and max(2 channels, W20/2) in velocity.
  design    How the injected sources are drawn: parameter ranges, inclinations,
            position angles, and the fraction that is unresolved.
  edge      Detection against the SNR expectation in narrow frequency bins at
            the bottom of each field's band, to place a band-edge cut.
"""
import argparse
import numpy as np

from .config import field_key
from scipy.optimize import minimize

from .compare import collect
from .survey import fit_fields, col

F0_MHZ, C_KMS = 1420.405751768, 299792.458


def nll(p, x, d):
    a, x0, lk = p
    if not 0 < a <= 1:
        return 1e30
    c = np.clip(a / (1 + np.exp(-np.clip((x - x0) / np.exp(lk), -60, 60))), 1e-9, 1 - 1e-9)
    return -np.sum(d * np.log(c) + (~d) * np.log(1 - c))


def plateau(res):
    print("PLATEAU  three-parameter logistic in log SNR (A = asymptote)")
    print("  %-14s %6s %8s %9s   %-26s %s" % ("field", "A", "+-", "dlnL", "detected at SNR>30",
                                            "missed at SNR>30: median MHz, share < 1310 MHz"))
    for f in sorted(res, key=field_key):
        x, d = res[f]["logsnr"], res[f]["det"]
        fs = res[f]["fit_snr"]
        p2 = np.array([1.0, fs["x0"], np.log(fs["k"])])
        r = minimize(nll, np.array([0.97, fs["x0"], np.log(fs["k"])]), args=(x, d),
                     method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
        a = r.x[0]
        # curvature error on A from a 1-D profile
        h = 1e-3
        f0 = nll(r.x, x, d)
        fp = nll(r.x + [h, 0, 0], x, d) if a + h <= 1 else f0
        fm = nll(r.x - [h, 0, 0], x, d)
        curv = (fp - 2 * f0 + fm) / h ** 2
        ea = 1 / np.sqrt(curv) if curv > 0 else np.nan
        dlnl = nll(p2, x, d) - r.fun
        hi = x > np.log10(30)
        miss = hi & ~d
        fq = F0_MHZ * (1 - col(res[f]["t"], "vsys_kms") / C_KMS)
        tail = "-"
        if miss.any():
            tail = "%.0f, %.0f%%" % (np.median(fq[miss]), 100 * np.mean(fq[miss] < 1310))
        print("  %-14s %6.3f %8.3f %9.1f   %4d of %4d (%5.1f%%)        %s" % (
            f, a, ea, dlnl, d[hi].sum(), hi.sum(), 100 * d[hi].mean() if hi.any() else np.nan, tail))
    print("  dlnL > 2 favours the plateau at about 2 sigma; > 4.5 at about 3 sigma.\n")


def chance(res):
    print("CHANCE  probability that an unrelated catalogue source falls inside a matching gate")
    print("  %-14s %8s %9s %11s %12s %12s" % ("field", "N_cat", "gate [']", "gate [km/s]",
                                             "P per source", "spurious/real"))
    for f in sorted(res, key=field_key):
        t, d = res[f]["t"], res[f]["det"]
        nreal = max(res[f]["n_real"], 1)
        ncat = d.sum() / nreal                      # detections per injected cube
        beam = np.nanmedian(col(t, "dHI_arcsec") / col(t, "dHI_over_beam")) / 60.0
        r_area = np.pi * (2 * beam) ** 2
        xp, yp = col(t, "xpix"), col(t, "ypix")
        rad = np.nanpercentile(np.hypot(xp - np.nanmedian(xp), yp - np.nanmedian(yp)), 99) * 6 / 60.0
        area = np.pi * rad ** 2
        dv = 44.11
        gate_v = 2 * np.nanmedian(np.maximum(2 * dv, 0.5 * col(t, "W20_realised_kms")))
        v = col(t, "vsys_kms")
        vrange = np.nanmax(v) - np.nanmin(v) + gate_v
        p = ncat * (r_area / area) * (gate_v / vrange)
        miss = 1 - d.mean()
        print("  %-14s %8.0f %9.2f %11.0f %12.2e %12.2e" % (f, ncat, 2 * beam, gate_v / 2, p,
                                                           p * miss))
    print("  'spurious/real': expected spurious matches per injected source, i.e. P x (1 - C).\n")


def design(res):
    print("DESIGN  what was injected, pooled over fields")
    t = np.concatenate
    snr = t([col(res[f]["t"], "snr_design") for f in res])
    w = t([col(res[f]["t"], "W50_kms") for f in res])
    dr = t([col(res[f]["t"], "dHI_over_beam") for f in res])
    inc = t([col(res[f]["t"], "inc_deg") for f in res])
    pa = t([col(res[f]["t"], "pa_deg") for f in res])
    lm = t([col(res[f]["t"], "logMHI") for f in res])
    zz = t([col(res[f]["t"], "z") for f in res])
    for name, v, fmt in (("SNR_int", snr, "%.2f"), ("W50 [km/s]", w, "%.0f"),
                         ("d_HI / beam", dr, "%.2f"), ("inclination [deg]", inc, "%.0f"),
                         ("PA [deg]", pa, "%.0f"), ("log M_HI", lm, "%.2f"), ("z", zz, "%.3f")):
        g = np.isfinite(v)
        if g.any():
            print("  %-18s " % name + " to ".join(fmt % q for q in np.percentile(v[g], [0.5, 99.5]))
                  + "   (median " + fmt % np.median(v[g]) + ")")
    g = np.isfinite(dr)
    print("  unresolved (d_HI < beam): %.0f%%;  N injected: %d over %d fields\n"
          % (100 * np.mean(dr[g] < 1), len(snr), len(res)))


def edge(res, width=4.0, span=36.0):
    """Detection against the SNR expectation in narrow bins at the low-frequency edge."""
    from .selection import logistic
    print("EDGE  detected - P(SNR) in %.0f MHz bins from each field's lowest injected frequency" % width)
    for f in sorted(res, key=field_key):
        d, fit = res[f], res[f]["fit_snr"]
        fq = F0_MHZ * (1 - col(d["t"], "vsys_kms") / C_KMS)
        r = d["det"].astype(float) - logistic(d["logsnr"], fit["x0"], fit["k"])
        lo = np.nanmin(fq)
        cells = []
        for a in np.arange(lo, lo + span, width):
            m = (fq >= a) & (fq < a + width)
            if m.sum() >= 15:
                cells.append("%6.1f:%+.2f" % (a + width / 2, r[m].mean()))
        print("  %-14s %s" % (f, "  ".join(cells)))
    print("  Each entry is bin-centre MHz : mean(detected - P). Look for where it drops below")
    print("  about -0.1 and stays there; --min-freq in temba.paperfigs cuts just above it.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fields", default="fields")
    ap.add_argument("--bootstrap", type=int, default=50)
    a = ap.parse_args()
    fields = collect(a.runs)
    res = fit_fields(fields, a.bootstrap)
    if not res:
        raise SystemExit("no field has enough injections to fit (at least 60 usable); "
                         "inject more: injection.total_per_field or injection.realisations")
    design(res)
    plateau(res)
    edge(res)
    chance(res)


if __name__ == "__main__":
    main()
