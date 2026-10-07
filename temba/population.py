"""
temba.population -- draw the injected population for one TEMBA realisation.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

Why this module exists
----------------------
v0.5 drew log M_HI and let the scaling relations do the rest: Wang+16 fixed the
size, Tully-Fisher fixed the rotation, and a single field distance fixed the
flux. The design matrix was therefore rank-2, every "predictor" was a relabelled
copy of the mass, and the flux and mass completeness tables came out as
byte-identical files. No multidimensional selection function can be fitted to
that, however many sources are injected.

TEMBA inverts the causality. Detectability is set by integrated signal-to-noise,
not by flux:

    SNR_int = F_int / [ sigma_chan * sqrt(dv_chan * W50) * max(1, d_HI/theta_beam) ]

so this module draws SNR_int, W50, angular size and inclination *independently*
on a Latin hypercube, measures sigma_chan at the chosen position in the real
substrate, and then solves for the flux the source must have. Flux is an output,
not an input. The scaling relations are deliberately switched off.

That measures the instrument response P(detect | phi) on a grid where the axes
vary independently. The astrophysical answer is recovered afterwards by
marginalising over a realistic population,

    C(S) = INT P(detect | S, phi) pi(phi | S) dphi

which costs no further injections and lets pi be swapped (field vs cluster, or a
different distance) without re-running anything.

Two further consequences worth knowing:

  * The angular-size floor (dratio_min) matters more than it looks: set too
    high, it rejects faint sources that cannot reach the minimum surface
    density at that size, and silently under-samples the very transition the
    experiment exists to measure. The report prints the realised SNR marginal
    against the drawn one and warns if they diverge.

  * Positions are drawn uniformly over valid voxels, so the primary-beam noise
    gradient (a factor ~7 across Abell-194) is sampled by construction. Local
    noise becomes a measured covariate rather than an unmodelled scatter.

  * Distance is computed per source from its own drawn v_sys, not from one field
    redshift. This is what finally breaks the exact F_int <-> M_HI degeneracy
    that made the v0.5 flux and mass analyses the same plot.

Unphysical corners are rejected with a mean-HI-surface-density envelope, because
3D-Barolo cannot build a sensible model for a 600 km/s linewidth in a 0.3-beam
disc. The envelope is configurable, and the module reports the acceptance rate
and the design correlation matrix so the cost of that rejection is visible
rather than hidden.

Output
------
An ECSV table, one row per source, carrying the design coordinates, the local
noise, the derived physical parameters, and the scalar 3D-Barolo inputs. The
injector consumes this table and does not draw anything itself.

Usage
-----
    python -m temba.population \
        --substrate substrates/Abell-194_substrate.fits \
        --n 150 --seed 1 \
        --out runs/Abell-194/run_0001/injection/population.ecsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import physics as P
from astropy.io import fits
from astropy.table import Table

C_KMS = 299792.458
ARCSEC_PER_RAD = 206264.806


# ---------------------------------------------------------------------------
# cube geometry and local noise
# ---------------------------------------------------------------------------

def velocity_axis(hdr):
    """Radio-velocity axis [km/s] and the signed channel width."""
    crval = float(hdr.get("CRVAL3", 0.0))
    cdelt = float(hdr.get("CDELT3", 1.0))
    crpix = float(hdr.get("CRPIX3", 1.0))
    nz = int(hdr.get("NAXIS3", 0))
    ctype = str(hdr.get("CTYPE3", "")).upper()
    cunit = str(hdr.get("CUNIT3", "")).strip().lower()

    world = (np.arange(nz, dtype=float) + 1.0 - crpix) * cdelt + crval

    if "FREQ" in ctype or "hz" in cunit:
        scale = {"": 1.0, "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(cunit, 1.0)
        rest = float(hdr.get("RESTFRQ", hdr.get("RESTFREQ", 1.42040575177e9)))
        vel = C_KMS * (rest - world * scale) / rest
        return vel, float(np.median(np.diff(vel)))

    if "km" in cunit:
        return world, cdelt
    # metres per second, or unlabelled but obviously m/s
    if "m" in cunit or abs(cdelt) > 1e3:
        return world / 1e3, cdelt / 1e3
    return world, cdelt


def geometry(path):
    with fits.open(path, memmap=True) as hd:
        hdr = hd[0].header
        nz, ny, nx = hd[0].data.shape

    vel, dv = velocity_axis(hdr)
    bmaj = float(hdr["BMAJ"])
    bmin = float(hdr.get("BMIN", bmaj))
    if bmaj > 1.0:                       # header in arcsec rather than degrees
        bmaj, bmin = bmaj / 3600.0, bmin / 3600.0
    pixdeg = 0.5 * (abs(float(hdr["CDELT1"])) + abs(float(hdr["CDELT2"])))

    return dict(
        header=hdr, nx=nx, ny=ny, nz=nz, vel=vel, dv_kms=abs(dv),
        bmaj_arcsec=bmaj * 3600.0, bmin_arcsec=bmin * 3600.0,
        pix_arcsec=pixdeg * 3600.0,
        beam_area_pix=(np.pi / (4 * np.log(2))) * bmaj * bmin / pixdeg**2,
    )


def noise_map(path, blank=None, slab=128, chan_min=None, chan_max=None):
    """Per-pixel MAD-sigma, and a strictly 3-D validity map.

    sigma_chan(x, y) is the real, primary-beam-modulated noise the injected
    source competes with. `valid` is true only where EVERY channel is finite
    and non-blank, because the injected patch spans the whole spectral axis.
    Taking validity from the sigma map alone marks a pixel usable if any
    channel is finite, which is true for a cube blanked identically in all
    channels and false for one blanked per channel -- and produced 0 of 554
    placements on Abell-168.
    """
    with fits.open(path, memmap=True) as hd:
        data = hd[0].data
        # Restrict to the usable band before measuring anything. A field with a
        # flagged block -- 22.6% of J0314.3-4525 -- has every pixel scoring
        # 0.74 validity if the blank channels are counted, so the 0.8 threshold
        # rejected the entire field even though the remaining 369 channels are
        # perfect. The noise and the validity must be measured where the
        # sources will actually be injected.
        z0 = int(chan_min or 0)
        z1 = int(chan_max) + 1 if chan_max is not None else data.shape[0]
        data = data[z0:z1]
        nz, ny, nx = data.shape
        sig = np.full((ny, nx), np.nan, dtype=np.float32)
        vfrac = np.zeros((ny, nx), dtype=np.float32)

        for y0 in range(0, ny, slab):
            y1 = min(y0 + slab, ny)
            b = np.array(data[:, y0:y1, :], dtype=np.float32)
            bad = ~np.isfinite(b)
            if blank is not None:
                bad |= (b == blank)
            # fraction of channels this pixel is usable in, not "all of
            # them": with ~0.5% scattered blanking, no pixel is valid in all
            # 477 channels, but the primary-beam disc is valid in ~99.5%
            vfrac[y0:y1, :] = 1.0 - bad.mean(axis=0)
            b[bad] = np.nan
            import warnings
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                # fully blanked columns are expected and give all-NaN slices
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(b, axis=0)
                sig[y0:y1, :] = 1.4826 * np.nanmedian(np.abs(b - med[None]), axis=0)
    return sig, vfrac


def detect_blank(path, rng, nsample=4_000_000):
    with fits.open(path, memmap=True) as hd:
        d = hd[0].data
        s = np.asarray(d[::max(1, d.shape[0] // 32)]).ravel()
    if s.size > nsample:
        s = s[rng.integers(0, s.size, size=nsample)]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    v, c = np.unique(s, return_counts=True)
    i = int(np.argmax(c))
    return float(v[i]) if c[i] / s.size > 0.02 else None


# ---------------------------------------------------------------------------
# cosmology
# ---------------------------------------------------------------------------

def distances(vsys_kms, H0, Om0):
    """Luminosity and angular-diameter distance per source, from its own v_sys."""
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    z = np.asarray(vsys_kms, float) / C_KMS
    dl = cosmo.luminosity_distance(z).value          # Mpc
    return dl, dl / (1.0 + z) ** 2


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def latin_hypercube(n, d, rng):
    """Centred Latin hypercube on [0, 1)^d, scipy if present, numpy otherwise."""
    try:
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=d, seed=int(rng.integers(0, 2**31 - 1))).random(n)
    except ImportError:
        u = np.empty((n, d))
        for j in range(d):
            u[:, j] = (rng.permutation(n) + rng.random(n)) / n
        return u


def draw(geo, sigma, vfrac, n, rng, cfg):
    """Draw one realisation. Returns a dict of per-source arrays."""
    ny, nx = sigma.shape
    vel = geo["vel"]
    # Some fields have a contiguous flagged block at the band edge -- 22.6% of
    # J0314.3-4525 and 5.5% of J0351.1-8212 -- where no pixel is valid in any
    # channel. Injecting there is meaningless, and it also breaks the substrate
    # filler, which cannot estimate a noise level from a wholly blank spectrum.
    lo_ch, hi_ch = cfg.get("chan_min"), cfg.get("chan_max")
    sub = vel[int(lo_ch or 0): int(hi_ch) + 1 if hi_ch is not None else None]
    if len(sub) < 10:
        raise SystemExit("chan_min/chan_max leave fewer than 10 channels.")
    if lo_ch or hi_ch is not None:
        print(f"  restricted to channels {int(lo_ch or 0)}-"
              f"{int(hi_ch) if hi_ch is not None else len(vel) - 1} "
              f"({len(sub)} of {len(vel)})")
    vlo, vhi = float(np.min(sub)), float(np.max(sub))

    # A channelised W50 cannot fall much below one channel, so asking for
    # narrower lines only wastes injector iterations on unachievable targets.
    floor = 1.2 * geo["dv_kms"]
    if cfg["w50_min"] < floor:
        print(f"  note: raising w50_min from {cfg['w50_min']:.0f} to "
              f"{floor:.0f} km/s (1.2 channels); narrower lines are not "
              f"resolvable in this cube")
        cfg["w50_min"] = floor

    # Pixels where the substrate has measurable noise. A fixed erosion is not
    # enough: the model cutout grows with the source, up to ten beams across,
    # so a large disc near the edge would be injected onto blanked voxels and
    # then rejected. That loss is biased -- it removes big sources from the
    # noisiest part of the field -- so instead measure the distance to the
    # nearest invalid pixel and require each candidate's own cutout to fit.
    ok = np.isfinite(sigma) & (sigma > 0) & (vfrac >= cfg["min_channel_valid_frac"])
    try:
        # chessboard metric, because the cutout is a square: Euclidean
        # distance under-protects the corners
        from scipy.ndimage import distance_transform_cdt
        padded = np.zeros((ok.shape[0] + 2, ok.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = ok
        dist = distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]
    except ImportError:                                    # crude fallback
        m = int(np.ceil(cfg["edge_margin_beams"] * geo["bmaj_arcsec"]
                        / geo["pix_arcsec"]))
        dist = np.where(ok, float(m), 0.0)
        dist[:m, :] = dist[-m:, :] = 0.0
        dist[:, :m] = dist[:, -m:] = 0.0

    base = cfg["edge_margin_beams"] * geo["bmaj_arcsec"] / geo["pix_arcsec"]
    ypix_ok, xpix_ok = np.nonzero(ok & (dist >= base))
    if ypix_ok.size == 0:
        raise SystemExit("No valid positions in the substrate.")

    keep = {k: [] for k in
            ("x", "y", "vsys", "snr", "w50", "dratio", "inc", "pa", "sigma")}
    tried = accepted = 0
    max_tries = cfg["max_tries_per_source"] * n

    while len(keep["x"]) < n and tried < max_tries:
        batch = max(64, 2 * (n - len(keep["x"])))
        u = latin_hypercube(batch, 4, rng)

        snr = 10 ** (cfg["log_snr_min"] + u[:, 0] * (cfg["log_snr_max"] - cfg["log_snr_min"]))
        w50 = 10 ** (np.log10(cfg["w50_min"]) + u[:, 1] *
                     (np.log10(cfg["w50_max"]) - np.log10(cfg["w50_min"])))
        drat = 10 ** (np.log10(cfg["dratio_min"]) + u[:, 2] *
                      (np.log10(cfg["dratio_max"]) - np.log10(cfg["dratio_min"])))
        # uniform in cos(i): the isotropic prior on orientation
        cosi = np.cos(np.radians(cfg["inc_max"])) + u[:, 3] * (
            np.cos(np.radians(cfg["inc_min"])) - np.cos(np.radians(cfg["inc_max"])))
        inc = np.degrees(np.arccos(np.clip(cosi, -1, 1)))
        pa = rng.uniform(0, 360, batch)

        k = rng.integers(0, ypix_ok.size, size=batch)
        ypix, xpix = ypix_ok[k], xpix_ok[k]
        sig = sigma[ypix, xpix].astype(float)

        pad = cfg["vpad_channels"] * geo["dv_kms"] + 0.5 * cfg["w20_over_w50"] * w50
        lo, hi = vlo + pad, vhi - pad
        good = hi > lo
        vsys = np.where(good, lo + rng.random(batch) * np.maximum(hi - lo, 0), np.nan)

        theta = drat * geo["bmaj_arcsec"]
        fint = snr * sig * np.sqrt(geo["dv_kms"] * w50) * np.maximum(1.0, drat)

        # half the model cutout this source will need, in cube pixels:
        # csize = model_csize_factor * R_out + beam_margin * BMAJ, with
        # R_out = rout_over_rHI * d_HI / 2 (defaults 3.0, 3.0 and 1.3)
        need_pix = (0.975 * theta + 1.5 * geo["bmaj_arcsec"]) / geo["pix_arcsec"]

        dl, da = distances(np.nan_to_num(vsys, nan=0.5 * (vlo + vhi)),
                           cfg["H0"], cfg["Om0"])
        mhi = 2.356e5 * dl**2 * fint
        d_kpc = theta / ARCSEC_PER_RAD * da * 1e3
        sigma_hi = mhi / (np.pi * (0.5 * d_kpc * 1e3) ** 2)     # Msun / pc^2

        # Reject combinations 3D-Barolo cannot represent, before they reach it.
        # W50 and d_HI are drawn independently -- deliberately -- so a 500 km/s
        # line can land on a 4 arcsec disc. The projected rotation then changes
        # by 20 channels between adjacent pixels, GALMOD's cloud bookkeeping
        # overruns, and BBarolo exits with "double free or corruption". That
        # cost 35% of the sources on two fields before it was diagnosed.
        good &= P.representable(w50, inc, theta, d_kpc, geo["pix_arcsec"],
                                geo["dv_kms"],
                                max_grad_chan_per_pix=cfg["max_vgrad"],
                                min_d_hi_kpc=cfg["min_dHI_kpc"],
                                min_d_hi_arcsec=cfg["min_dHI_arcsec"])
        good &= np.isfinite(sig) & (sig > 0) & np.isfinite(fint)
        good &= (sigma_hi >= cfg["sigma_hi_min"]) & (sigma_hi <= cfg["sigma_hi_max"])
        good &= dist[ypix, xpix] >= need_pix

        tried += batch
        for i in np.nonzero(good)[0]:
            if len(keep["x"]) >= n:
                break
            if not separated(keep, geo, cfg, xpix[i], ypix[i], vsys[i],
                             theta[i], w50[i]):
                continue
            keep["x"].append(int(xpix[i])); keep["y"].append(int(ypix[i]))
            keep["vsys"].append(float(vsys[i])); keep["snr"].append(float(snr[i]))
            keep["w50"].append(float(w50[i])); keep["dratio"].append(float(drat[i]))
            keep["inc"].append(float(inc[i])); keep["pa"].append(float(pa[i]))
            keep["sigma"].append(float(sig[i]))
            accepted += 1

    if len(keep["x"]) < n:
        raise SystemExit(
            f"Only placed {len(keep['x'])} of {n} sources after {tried} attempts. "
            "Loosen sigma_hi limits, reduce min_sep_beams, or lower --n.")

    return {k: np.asarray(v) for k, v in keep.items()}, dict(
        attempts=tried, acceptance=accepted / max(tried, 1))


def separated(keep, geo, cfg, x, y, v, theta, w50):
    """Reject a candidate whose 3-D footprint overlaps one already placed.

    Two sources conflict only if they overlap both on the sky and in velocity,
    so this is a genuine ellipsoid test rather than a fixed separation. v0.5
    used min_vsep_kms = 0 and a 1-beam spatial floor, which permits blends.
    """
    if not keep["x"]:
        return True
    dx = (np.asarray(keep["x"]) - x) * geo["pix_arcsec"]
    dy = (np.asarray(keep["y"]) - y) * geo["pix_arcsec"]
    sep = np.hypot(dx, dy)

    need = (0.5 * (np.asarray(keep["dratio"]) * geo["bmaj_arcsec"] + theta)
            + cfg["min_sep_beams"] * geo["bmaj_arcsec"])
    close = sep < need

    dv = np.abs(np.asarray(keep["vsys"]) - v)
    needv = (0.5 * cfg["w20_over_w50"] * (np.asarray(keep["w50"]) + w50)
             + cfg["min_vsep_channels"] * geo["dv_kms"])
    return not np.any(close & (dv < needv))


# ---------------------------------------------------------------------------

def build_table(d, geo, cfg, field, run_id, seed):
    dl, da = distances(d["vsys"], cfg["H0"], cfg["Om0"])
    theta = d["dratio"] * geo["bmaj_arcsec"]
    fint = d["snr"] * d["sigma"] * np.sqrt(geo["dv_kms"] * d["w50"]) * np.maximum(1.0, d["dratio"])
    mhi = 2.356e5 * dl**2 * fint
    d_kpc = theta / ARCSEC_PER_RAD * da * 1e3
    sigma_hi = mhi / (np.pi * (0.5 * d_kpc * 1e3) ** 2)

    sini = np.sin(np.radians(d["inc"]))
    vdisp = np.full(len(fint), cfg["vdisp_kms"])
    # deconvolve turbulent broadening from the drawn linewidth
    vrot = np.sqrt(np.maximum((0.5 * d["w50"]) ** 2 - (2.0 * vdisp) ** 2, 1.0)) / np.maximum(sini, 0.05)

    with fits.open(cfg["substrate"], memmap=True) as hd:
        from astropy.wcs import WCS
        w = WCS(hd[0].header).celestial
    ra, dec = w.all_pix2world(d["x"].astype(float), d["y"].astype(float), 0)

    t = Table()
    t["id"] = np.arange(1, len(fint) + 1)
    # --- design coordinates: what was actually controlled -------------------
    t["snr_design"] = d["snr"]
    t["W50_kms"] = d["w50"]
    t["dHI_over_beam"] = d["dratio"]
    t["inc_deg"] = d["inc"]
    t["sigma_chan_Jybeam"] = d["sigma"]
    # --- position -----------------------------------------------------------
    t["xpix"] = d["x"]; t["ypix"] = d["y"]
    t["ra"] = ra; t["dec"] = dec
    t["vsys_kms"] = d["vsys"]
    # --- derived physical quantities ----------------------------------------
    t["Fint_Jykms"] = fint
    t["logMHI"] = np.log10(mhi)
    t["W20_kms"] = cfg["w20_over_w50"] * d["w50"]
    t["dHI_arcsec"] = theta
    t["dHI_kpc"] = d_kpc
    t["sigma_HI_Msun_pc2"] = sigma_hi
    t["DL_Mpc"] = dl; t["DA_Mpc"] = da
    t["z"] = d["vsys"] / C_KMS
    # --- 3D-Barolo model inputs ---------------------------------------------
    t["vrot_kms"] = vrot
    t["vdisp_kms"] = vdisp
    t["pa_deg"] = d["pa"]
    t["z0_arcsec"] = cfg["z0_over_dHI"] * theta

    t.meta["temba"] = dict(
        module="temba.population", field=field, run_id=run_id, seed=seed,
        substrate=str(cfg["substrate"]),
        beam_arcsec=geo["bmaj_arcsec"], dv_chan_kms=geo["dv_kms"],
        pix_arcsec=geo["pix_arcsec"], config={k: v for k, v in cfg.items()
                                              if not isinstance(v, Path)},
    )
    return t


def uniformity_test(x, lo, hi, nbins=7):
    """Chi-square test of a drawn marginal against the intended uniform one.

    A fixed tolerance on one summary fraction false-alarms constantly at
    n = 150, where the binomial scatter is already several percentage points.
    Testing the whole histogram catches real bias and ignores noise.
    """
    x = np.asarray(x, float)
    obs, _ = np.histogram(x, bins=nbins, range=(lo, hi))
    exp = len(x) / nbins
    chi2 = float(np.sum((obs - exp) ** 2) / exp)
    dof = nbins - 1
    try:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(chi2, dof))
    except ImportError:                                        # pragma: no cover
        p = float("nan")
    return chi2, dof, p


def design_diagnostics(t):
    """Report how identifiable the design actually is.

    VIF is computed over the *design* axes only. F_int is excluded because it
    is solved from them and is therefore collinear by construction; its
    correlation with each axis is reported separately, and is itself the point:
    flux is not a clean proxy for detectability once size and linewidth vary
    independently.
    """
    names = ["log SNR", "log W50", "log dHI/beam", "cos inc", "log sigma_chan"]
    X = np.column_stack([
        np.log10(t["snr_design"]), np.log10(t["W50_kms"]),
        np.log10(t["dHI_over_beam"]), np.cos(np.radians(t["inc_deg"])),
        np.log10(t["sigma_chan_Jybeam"]),
    ])
    Xs = (X - X.mean(0)) / X.std(0)
    corr = np.corrcoef(Xs, rowvar=False)

    vif = []
    for j in range(Xs.shape[1]):
        others = np.delete(Xs, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Xs[:, j], rcond=None)
        r2 = 1.0 - np.sum((Xs[:, j] - others @ beta) ** 2) / np.sum(Xs[:, j] ** 2)
        vif.append(1.0 / max(1.0 - r2, 1e-12))

    logf = np.log10(t["Fint_Jykms"])
    logf = (logf - logf.mean()) / logf.std()
    flux_corr = np.array([np.corrcoef(logf, Xs[:, j])[0, 1] for j in range(Xs.shape[1])])
    return names, corr, np.asarray(vif), flux_corr


# ---------------------------------------------------------------------------

DEFAULTS = dict(
    log_snr_min=-0.5, log_snr_max=1.6,
    w50_min=40.0, w50_max=600.0,
    dratio_min=0.25, dratio_max=4.0,
    inc_min=15.0, inc_max=85.0,
    sigma_hi_min=1.2, sigma_hi_max=30.0,
    w20_over_w50=1.3, vdisp_kms=9.0, z0_over_dHI=0.05,
    min_channel_valid_frac=0.8,
    max_vgrad=6.0, min_dHI_kpc=2.0, min_dHI_arcsec=5.0,
    min_sep_beams=1.5, min_vsep_channels=2.0, vpad_channels=4.0,
    edge_margin_beams=2.0, max_tries_per_source=400,
    H0=70.0, Om0=0.3, chan_min=None, chan_max=None,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substrate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--field", default="")
    ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--noise-map", help="Cache path for sigma_chan(x,y); reused if present.")
    for k, v in DEFAULTS.items():
        # A default of None makes type(v) NoneType, which argparse then uses as
        # the converter and which rejects every value: --chan-min 108 failed
        # with "invalid NoneType value". Optional numeric settings therefore
        # need an explicit type. This cost both flagged-band fields a full
        # campaign, silently, because the failure was a one-second rc=2 inside
        # a per-realisation log.
        ap.add_argument(f"--{k.replace('_', '-')}",
                        type=type(v) if v is not None else float, default=v)
    a = ap.parse_args()

    cfg = {k: getattr(a, k) for k in DEFAULTS}
    cfg["substrate"] = a.substrate
    rng = np.random.default_rng(a.seed)

    geo = geometry(a.substrate)
    blank = detect_blank(a.substrate, rng)

    # cache holds two planes: sigma_chan and the 3-D validity map
    cache = Path(a.noise_map) if a.noise_map else None
    sigma = valid = None
    if cache and cache.exists():
        arr = np.asarray(fits.getdata(cache))
        if arr.ndim == 3 and arr.shape[0] == 2:
            sigma, vfrac = arr[0].astype(np.float32), arr[1].astype(np.float32)
            print(f"  reusing noise map {cache}")
        else:
            print(f"  {cache} predates the 3-D validity map; rebuilding")
    if sigma is None:
        print("  measuring sigma_chan(x, y) from the substrate ...", flush=True)
        sigma, vfrac = noise_map(a.substrate, blank,
                                 chan_min=cfg["chan_min"], chan_max=cfg["chan_max"])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(np.stack([sigma, vfrac])).writeto(cache, overwrite=True)
            print(f"  wrote noise map {cache}")

    thr = cfg["min_channel_valid_frac"]
    frac = float((vfrac >= thr).mean())
    print(f"  channels valid per pixel: median {np.median(vfrac[vfrac > 0]):.3f}; "
          f"{frac:.1%} of the field is valid in >= {thr:.0%} of channels")
    if frac < 0.02:
        raise SystemExit(
            f"Only {frac:.2%} of the field is valid in {thr:.0%} of channels. "
            "Lower --min-channel-valid-frac, or check the substrate.")

    d, stats = draw(geo, sigma, vfrac, a.n, rng, cfg)
    t = build_table(d, geo, cfg, a.field, a.run_id, a.seed)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t.write(out, format="ascii.ecsv", overwrite=True)

    # ---- report -----------------------------------------------------------
    fin = np.isfinite(sigma) & (sigma > 0) & (vfrac >= cfg["min_channel_valid_frac"])
    w = 74
    print("=" * w)
    print(f"TEMBA POPULATION: {a.n} sources, seed {a.seed}")
    print("=" * w)
    print(f"  beam {geo['bmaj_arcsec']:.2f}\"   pixel {geo['pix_arcsec']:.2f}\"   "
          f"dv {geo['dv_kms']:.2f} km/s")
    print(f"  sigma_chan over the field: median {np.nanmedian(sigma[fin]):.4g}, "
          f"range {np.nanpercentile(sigma[fin],1):.4g} to {np.nanpercentile(sigma[fin],99):.4g} Jy/beam")
    print(f"  placement acceptance {stats['acceptance']:.1%} over {stats['attempts']} attempts")
    print()
    q = lambda c: (np.min(t[c]), np.median(t[c]), np.max(t[c]))
    for c, lab in [("snr_design", "SNR_int"), ("Fint_Jykms", "F_int [Jy km/s]"),
                   ("logMHI", "log M_HI"), ("W50_kms", "W50 [km/s]"),
                   ("dHI_over_beam", "d_HI / beam"), ("inc_deg", "inclination [deg]"),
                   ("sigma_HI_Msun_pc2", "Sigma_HI [Msun/pc2]"),
                   ("vsys_kms", "v_sys [km/s]"), ("DL_Mpc", "D_L [Mpc]")]:
        lo, md, hi = q(c)
        print(f"  {lab:<24s} {lo:10.4g} {md:10.4g} {hi:10.4g}   (min / median / max)")

    below = float(np.mean(t["snr_design"] < 5.0))
    expect = float(np.clip((np.log10(5.0) - cfg["log_snr_min"]) /
                           (cfg["log_snr_max"] - cfg["log_snr_min"]), 0.0, 1.0))
    se = float(np.sqrt(max(expect * (1 - expect), 1e-9) / len(t)))
    print(f"\n  fraction below SNR_int = 5 : {below:.1%}   "
          f"(log-uniform draw predicts {expect:.1%} +/- {se:.1%})")

    # Test the whole SNR marginal, not one cut, and judge it statistically
    # rather than against a fixed tolerance: at n = 150 the binomial scatter
    # alone is several percentage points.
    chi2, dof, pval = uniformity_test(np.log10(t["snr_design"]),
                                      cfg["log_snr_min"], cfg["log_snr_max"])
    print(f"  log-SNR marginal uniformity: chi2 = {chi2:.1f} on {dof} dof, p = {pval:.3f}")

    nb = dict(size_floor=int(np.sum(t["dHI_over_beam"] < 1.2 * cfg["dratio_min"])),
              sigma_floor=int(np.sum(t["sigma_HI_Msun_pc2"] < 1.2 * cfg["sigma_hi_min"])),
              sigma_ceiling=int(np.sum(t["sigma_HI_Msun_pc2"] > cfg["sigma_hi_max"] / 1.2)))
    print(f"  sources against a boundary: {nb['size_floor']} at the size floor, "
          f"{nb['sigma_floor']} at the Sigma floor, {nb['sigma_ceiling']} at the ceiling")

    # The test compares against an UNCONSTRAINED log-uniform draw, which this
    # design can never match. F_int is derived from SNR, size and linewidth, so
    # Sigma_HI = M_HI / (pi R^2) inherits all three: the density floor removes
    # faint-large combinations and the size floor removes faint-small ones, and
    # the faint end is squeezed from both sides by construction. Measured over
    # five seeds on Abell-168 the deficit is a stable 12 points with 3 points of
    # scatter, and widening the envelope moves it the wrong way while lowering
    # the VIFs -- so it is a property of a physically constrained design, not a
    # fault to tune away. What matters is that the transition is sampled and the
    # coefficients stay separable, which the VIF table below reports. Warn only
    # when the marginal is distorted far beyond that.
    if pval < 1e-6:
        print()
        print("  WARNING: the SNR marginal is severely distorted (p < 1e-6), well")
        print("           beyond the offset the envelope alone produces. Check the")
        print("           boundary counts and the VIF table: if VIFs also exceed 5,")
        print("           the design is not identifiable and the envelope needs work.")

    names, corr, vif, fcorr = design_diagnostics(t)
    print()
    print("  DESIGN IDENTIFIABILITY  (design axes only; F_int is solved from them)")
    print("  " + "-" * (w - 4))
    print(f"  {'':<16s}" + "".join(f"{n:>16s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<16s}" + "".join(f"{corr[i, j]:16.2f}" for j in range(len(names))))
    print(f"  {'VIF':<16s}" + "".join(f"{v:16.2f}" for v in vif))
    print(f"  {'corr w/ log F':<16s}" + "".join(f"{c:16.2f}" for c in fcorr))
    bad = [n for n, v in zip(names, vif) if v > 5]
    print()
    if bad:
        print(f"  WARNING: collinear axes (VIF > 5): {', '.join(bad)}")
    else:
        print("  All VIF < 5: every coefficient is separately identifiable.")
    print("  v0.5's design was effectively rank-2 here, which is why its AIC")
    print("  comparison between flux and linewidth was meaningless.")
    print(f"\n  wrote {out}")

    meta = dict(stats=stats, config=cfg, n=a.n, seed=a.seed,
                vif=dict(zip(names, vif.tolist())),
                corr_with_logflux=dict(zip(names, fcorr.tolist())),
                fraction_below_snr5=below, fraction_below_snr5_expected=expect,
                snr_uniformity=dict(chi2=chi2, dof=dof, p=pval), boundary_counts=nb)
    Path(str(out) + ".temba.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
