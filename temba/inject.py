"""
temba.inject -- build and inject the population into the substrate.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

This module draws nothing. temba.population decides what to inject; this module
executes that table row by row and verifies that what landed in the cube is
what was asked for.

What changed from v0.5's inject_bbarolo
---------------------------------------
v0.5 called

    mygal.define_galaxy(radii=..., xpos=..., ypos=..., vsys=...,
                        vrotdwarf=is_dwarf, radmotion=None)

and let pyBBarolo generate inclination, position angle, rotation curve,
dispersion, scale height, the density profile, and -- because `ishole=None`
means "randomly decide" -- whether the disc even has a central hole. Those
were then read back out of the saved ring file as `inc_model_deg`,
`pa_model_deg`, `vrot_max_model_kms`. So the properties that govern
detectability were random variables discovered after the fact, not controlled
ones. No selection function can be measured that way.

TEMBA sets all of them explicitly, and switches off every remaining random
draw (`ishole=False`, `warpinc=False`, `warppa=False`, `flare=False`,
`radmotion=False`).

The W50 loop
------------
W50 is a *design axis*, so it is not enough to set v_rot and hope. The realised
W50 depends on inclination, the shape of the rotation curve, the turbulent
dispersion and the channel width. This module measures W50 on the noiseless
model exactly as the analysis will later measure it, and iterates v_rot by
secant until the realised value matches the design within tolerance. A source
that will not converge is rolled back rather than injected with the wrong
linewidth.

Size convention
---------------
The population specifies d_HI, the HI diameter at the 1 Msun/pc^2 isophote.
The emission extends beyond that, so rings run out to

    R_out = rout_over_rHI * d_HI / 2      (default 1.3)

and the realised isophotal diameter is recorded in the truth table. Getting
this wrong biases the measured size dependence, so it is explicit rather than
implied.

Usage
-----
    python -m temba.inject \
        --population runs/Abell-194/run_0001/injection/population.ecsv \
        --substrate  substrates/Abell-194_substrate.fits \
        --bb-exe     /path/to/BBarolo \
        --outdir     runs/Abell-194/run_0001/injection

    # inspect the ring tables without running BBarolo at all
    python -m temba.inject ... --dry-model

    # end-to-end on the first few sources before committing to 150
    python -m temba.inject ... --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

C_KMS = 299792.458


# ---------------------------------------------------------------------------
# cube helpers (kept compatible with v0.5's conventions)
# ---------------------------------------------------------------------------

def velocity_axis(hdr):
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
    if "m" in cunit or abs(cdelt) > 1e3:
        return world / 1e3, cdelt / 1e3
    return world, cdelt


def measure_width(spec, vel, frac):
    """Full width at `frac` of the global-profile peak, from the outermost crossings.

    Walk inwards from each end of the profile to the first channel at or above
    frac x peak, and interpolate outwards from there. The earlier version walked
    outwards from the peak and stopped at the first channel below the level.
    GALMOD models built on too few sub-rings have comb-shaped spectra with dips
    below half the peak inside the line (see min_rout_model_pix), so it measured
    only part of the line: realised W50 = 381 km/s with W20 = 1304 km/s. The W50
    solver then raised v_rot until that part matched the design, and injected a
    line far wider than the truth table says. On a profile without interior dips
    the two definitions agree.

    Interpolating the crossings (rather than counting channels above the level)
    keeps lines narrower than two channels measurable and the result continuous.
    Only ever applied to noiseless model spectra, where the outermost crossing is
    well defined; on a noisy spectrum it would pick up noise spikes.
    """
    s = np.asarray(spec, float).copy()
    v = np.asarray(vel, float)
    s[~np.isfinite(s)] = 0.0
    if s.size < 2:
        return np.nan
    peak = float(np.max(s))
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    lvl = frac * peak
    above = np.nonzero(s >= lvl)[0]
    lo, hi = int(above[0]), int(above[-1])

    def edge(i, j):
        """Crossing between channel i (at or above lvl) and its outer neighbour j."""
        if not (0 <= j < s.size):
            return float(v[i])                    # runs off the band edge
        d = s[i] - s[j]
        if abs(d) < 1e-30:
            return float(v[j])
        return float(v[i] + (s[i] - lvl) / d * (v[j] - v[i]))

    return float(abs(edge(hi, hi + 1) - edge(lo, lo - 1)))


def profile_dip(spec):
    """Depth of the deepest dip inside a profile, as a fraction of its peak.

    For each channel, the dip is how far it sits below the lower of the highest
    channels on either side of it. On the J0540 test sources, models built on
    ~7 GALMOD sub-rings scored 0.13 to 0.31, and the same sources on ~20
    sub-rings scored 0.00 to 0.07.
    """
    s = np.asarray(spec, float).copy()
    s[~np.isfinite(s)] = 0.0
    peak = float(np.max(s)) if s.size else 0.0
    if s.size < 3 or peak <= 0:
        return float("nan")
    left = np.maximum.accumulate(s)
    right = np.maximum.accumulate(s[::-1])[::-1]
    return float(np.max(np.minimum(left, right) - s) / peak)


def moment_centroid_2d(img):
    w = np.asarray(img, float).copy()
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    tot = w.sum()
    if tot <= 0:
        return np.nan, np.nan
    ny, nx = w.shape
    y, x = np.mgrid[0:ny, 0:nx]
    return float((w * x).sum() / tot), float((w * y).sum() / tot)


def isophotal_size(mom0, sigma_chan, dv_kms, n_chan, pix_arcsec, nsigma=3.0):
    """Diameter of the n-sigma contour of a moment-0 map, two ways.

    Returns (equivalent-circular diameter, second-moment major axis), both in
    arcsec. This is NOT the same quantity as d_HI: it depends on the beam and
    on the local noise, so the same galaxy measures smaller where the cube is
    noisier. SoFiA's ell3s_maj uses its own convention, so the ratio between
    these and that is measured empirically rather than assumed.
    """
    m = np.asarray(mom0, float)
    lvl = nsigma * float(sigma_chan) * abs(dv_kms) * np.sqrt(max(n_chan, 1))
    sel = np.isfinite(m) & (m > lvl)
    n = int(sel.sum())
    if n < 3:
        return float("nan"), float("nan")

    d_area = 2.0 * np.sqrt(n / np.pi) * pix_arcsec

    yy, xx = np.nonzero(sel)
    w = m[sel]
    tot = w.sum()
    if tot <= 0:
        return d_area, float("nan")
    xc = (w * xx).sum() / tot
    yc = (w * yy).sum() / tot
    cxx = (w * (xx - xc) ** 2).sum() / tot
    cyy = (w * (yy - yc) ** 2).sum() / tot
    cxy = (w * (xx - xc) * (yy - yc)).sum() / tot
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    lam = 0.5 * tr + np.sqrt(max(0.25 * tr * tr - det, 0.0))
    return d_area, float(4.0 * np.sqrt(max(lam, 0.0)) * pix_arcsec)


def border_flux_fraction(mom0, border=2):
    a = np.asarray(mom0, float).copy()
    a[~np.isfinite(a)] = 0.0
    tot = a.sum()
    if tot <= 0:
        return np.nan
    edge = (a[:border, :].sum() + a[-border:, :].sum()
            + a[:, :border].sum() + a[:, -border:].sum())
    return float(edge / tot)


# ---------------------------------------------------------------------------
# model definition
# ---------------------------------------------------------------------------

def solve_concentration(sigma_mean, lo=1e-3, hi=30.0, tol=1e-6):
    """Return x = R_HI / R_s for an exponential disc with this mean density.

    d_HI is defined as the diameter where Sigma = 1 Msun/pc^2, so the profile
    is not free: for an exponential,

        Sigma_mean / Sigma(R_HI) = g(x) = 2 (e^x - 1 - x) / x^2

    and requiring Sigma(R_HI) = 1 fixes x from Sigma_mean. g is monotonic with
    g -> 1 as x -> 0, so a bisection is safe. Using a fixed scale length
    instead, as the first version did, pins the ratio at 2.78 regardless of
    what the population asked for, which makes d_HI a nominal size rather than
    the Wang isophote.
    """
    s = float(sigma_mean)
    if not np.isfinite(s) or s <= 1.0:
        return float(lo)

    def g(x):
        return 2.0 * (np.expm1(x) - x) / (x * x)

    if g(hi) < s:
        return float(hi)
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if g(m) < s:
            a = m
        else:
            b = m
        if b - a < tol:
            break
    return float(0.5 * (a + b))


def scale_length(d_hi_arcsec, sigma_mean=None):
    """Exponential scale length, from the mean density when it is known."""
    if sigma_mean is not None:
        x = solve_concentration(sigma_mean)
        return max(0.5 * d_hi_arcsec / max(x, 1e-6), 1e-6)
    return max(d_hi_arcsec / 5.0, 1e-6)


def density_profile(radii_arcsec, d_hi_arcsec, kind="exponential",
                    sigma_mean=None):
    """Radial HI surface density, in units of 1e20 cm^-2.

    Only the *shape* matters: the model cube is rescaled globally to the target
    F_int afterwards, so the normalisation cancels. The shape does matter for
    detectability, which is why it is fixed here instead of being left to
    pyBBarolo's random default (v0.5 also let `ishole` randomise whether there
    was a central depression at all).

    When sigma_mean is given, the scale length is solved so that the
    1 Msun/pc^2 isophote falls at exactly d_HI/2, making d_HI the Wang
    isophote rather than a nominal size.
    """
    r = np.asarray(radii_arcsec, float)
    rs = scale_length(d_hi_arcsec, sigma_mean if kind == "exponential" else None)
    if kind == "flat":
        p = np.ones_like(r)
    elif kind == "hole":                     # exponential with a central depression
        p = np.exp(-r / rs) * (1.0 - 0.6 * np.exp(-(r / (0.6 * rs)) ** 2))
    else:
        p = np.exp(-r / rs)
    # Only the shape survives the later global rescale to F_int, but absurd
    # absolute values can upset GALMOD, so anchor the peak at 5e20 cm^-2.
    pmax = float(np.max(p))
    return 5.0 * p / pmax if pmax > 0 else np.ones_like(r)


def rotation_curve(radii_arcsec, vmax, r_turn_arcsec):
    """Arctan rotation curve normalised to reach vmax at the outermost ring.

    The unnormalised form, (2/pi) vmax arctan(r/r_turn), only reaches
    0.905 vmax at R_out when r_turn = 0.15 R_out. Every injected line therefore
    came out about 9% narrower than its design W50 -- a systematic, not
    scatter. Dividing by the value at R_out removes it.
    """
    r = np.asarray(radii_arcsec, float)
    rt = max(r_turn_arcsec, 1e-6)
    rmax = float(np.max(r)) if np.size(r) else rt
    norm = np.arctan(rmax / rt)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(r)
    return vmax * np.arctan(r / rt) / norm


def build_rings(row, geo, cfg):
    """Ring radii and per-ring model parameters for one population row."""
    d_hi = float(row["dHI_arcsec"])
    r_out = cfg["rout_over_rHI"] * 0.5 * d_hi

    # Ring spacing must exceed BBarolo's tolerance in *model* pixels, and the
    # grid is therefore chosen per source: a 0.1-beam dwarf needs a grid eight
    # times finer than the cube, while a 4-beam disc needs none at all. A
    # single global factor starves one end or wastes time at the other.
    # A disc whose scale length is a fraction of a model pixel makes GALMOD's
    # cloud allocation degenerate: on Abell-4038 (35" beam) sources with
    # d_HI = 3.5" -- 0.1 beams, R_s = 0.15" -- crashed BBarolo with "double
    # free or corruption". Such a source is a point source at any beam, so
    # nothing is lost by refusing to model it below a floor.
    if d_hi < cfg["min_dHI_arcsec"]:
        return np.array([]), np.array([]), 0.0, geo["pix_arcsec"], 1, 1e-6

    # GALMOD fills the disc with sub-rings 0.75 model pixels apart and gives
    # every cloud in a sub-ring that ring's single rotation speed, so each
    # sub-ring is a narrow double horn. With only ~7 sub-rings, neighbours on the
    # rising part of the curve differ by about two 44 km/s channels, the 9 km/s
    # dispersion cannot fill the gaps, and the spectrum becomes a comb with dips
    # below half the peak (J0540 test: 0.71 0.53 0.97 0.86 1.00 0.48 0.74).
    # Requiring R_out >= min_rout_model_pix model pixels (15 -> ~20 sub-rings)
    # smooths it to a few per cent, at no measurable cost in run time.
    need = max((cfg["min_rings"] - 1) * cfg["min_ring_spacing_pix"],
               cfg["min_rout_model_pix"])
    f = int(np.ceil(need * geo["pix_arcsec"] / max(r_out, 1e-6)))
    f = int(np.clip(max(f, int(cfg["model_oversample"]), 1),
                    1, int(cfg["max_oversample"])))

    pix = geo["pix_arcsec"] / f
    r_out_pix = r_out / pix
    n_rings = int(np.floor(r_out_pix / cfg["min_ring_spacing_pix"])) + 1
    n_rings = int(np.clip(n_rings, cfg["min_rings"], cfg["max_rings"]))

    radii = np.linspace(0.0, r_out, n_rings)
    sm = float(row["sigma_HI_Msun_pc2"]) if "sigma_HI_Msun_pc2" in row.colnames \
        else None
    dens = density_profile(radii, d_hi, cfg["density_profile"], sm)
    rs = scale_length(d_hi, sm if cfg["density_profile"] == "exponential" else None)
    return radii, dens, r_out, pix, f, rs


# ---------------------------------------------------------------------------
# injector
# ---------------------------------------------------------------------------

class Injector:

    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        self.outdir = Path(a.outdir)
        for sub in ("models", "bb_output"):
            (self.outdir / sub).mkdir(parents=True, exist_ok=True)

        with fits.open(a.substrate, memmap=True) as hd:
            self.hdr = hd[0].header.copy()
            self.cube = np.array(hd[0].data, dtype=np.float32)
        self.nz, self.ny, self.nx = self.cube.shape
        self.vel, self.dv = velocity_axis(self.hdr)

        bmaj = float(self.hdr["BMAJ"]); bmin = float(self.hdr.get("BMIN", bmaj))
        if bmaj > 1.0:
            bmaj, bmin = bmaj / 3600.0, bmin / 3600.0
        pixdeg = 0.5 * (abs(float(self.hdr["CDELT1"])) + abs(float(self.hdr["CDELT2"])))
        self.geo = dict(pix_arcsec=pixdeg * 3600.0,
                        bmaj_arcsec=bmaj * 3600.0, bmin_arcsec=bmin * 3600.0)
        self.beam_area_pix = (np.pi / (4 * np.log(2))) * bmaj * bmin / pixdeg**2
        # BBarolo wants the beam in DEGREES, like the FITS header. Passing
        # arcsec gives a beam 3600x too large, a beam area of ~1e10 pixels,
        # and an empty model cube.
        self.beam = [bmaj, bmin, float(self.hdr.get("BPA", 0.0))]

        # blank sentinel: MGCLS cubes use 1e-10 outside the primary beam
        self.blank = cfg["blank_value"]
        if self.blank is None:
            s = self.cube[::max(1, self.nz // 16)].ravel()
            s = s[np.isfinite(s)]
            v, c = np.unique(s[::max(1, s.size // 2_000_000)], return_counts=True)
            i = int(np.argmax(c))
            self.blank = float(v[i]) if c[i] / max(s[::max(1, s.size // 2_000_000)].size, 1) > 0.02 else None

        # cdelts are filled in per source in make_model, because the model
        # grid now depends on the source size. BBarolo measures ring
        # separation in MODEL pixels, so cdelt must shrink with the grid;
        # leaving it at the cube value makes GALMOD abort with
        # "Radius separation too small".
        self.cubecommon = dict(
            cdelts=[self.hdr["CDELT1"], self.hdr["CDELT2"], self.dv],
            crpixs=[self.hdr["CRPIX1"], self.hdr["CRPIX2"], self.hdr["CRPIX3"]],
            crvals=[self.hdr["CRVAL1"], self.hdr["CRVAL2"], float(self.vel[0])
                    - (float(self.hdr["CRPIX3"]) - 1.0) * self.dv],
            ctypes=["RA---SIN", "DEC--SIN", "VELO-LSR"],
            cunits=["deg", "deg", "km/s"],
            beam=self.beam, obj="TEMBA", bunit="JY/BEAM",
        )
        self.reasons = {}

    def fail(self, key):
        self.reasons[key] = self.reasons.get(key, 0) + 1

    # -- model construction -------------------------------------------------

    def make_model(self, row, vrot_max, tag):
        """Run BBarolo once and return the model cube, or None."""
        from pyBBarolo.utils import SimulatedGalaxyCube

        radii, dens, r_out, mpix, f, rs = build_rings(row, self.geo, self.cfg)
        if len(radii) < 2 or not np.isfinite(radii).all() or r_out <= 0:
            self.fail("bad_rings"); return None, None

        # The cutout has to hold the source AFTER convolution with the beam,
        # not just the rings. Sizing it on R_out alone gave 48-arcsec cutouts
        # for a 38.8-arcsec beam and rejected the source as model_cutout_tight.
        csize_arcsec = (self.cfg["model_csize_factor"] * r_out
                        + self.cfg["beam_margin"] * self.geo["bmaj_arcsec"])
        csize = max(int(np.ceil(csize_arcsec / mpix)), 40)
        if csize > self.cfg["max_model_csize"]:
            self.fail("model_too_large"); return None, None

        # The turnover must scale with the DENSITY scale length, not with
        # R_out. Solving the profile from Sigma_mean gives a dense small disc
        # R_s = d_HI/12.9, smaller than a turnover set at 0.15 R_out = d_HI/10:
        # all the gas then sits on the rising part of the curve, the line comes
        # out far too narrow, and v_rot runs into its clip without converging.
        # Real curves turn over at one to two scale lengths.
        # GALMOD fills each ring with discrete clouds, and the count goes as
        # DENS x area. A fixed peak column density therefore gave this 4.5-pixel
        # disc about two clouds in total ("GALMOD ERROR: No clouds used. Choose
        # higher CDENS.") while a large disc got thousands -- so model quality
        # depended on source size, which is a selection effect inside the tool
        # built to measure selection effects. Only the profile SHAPE survives
        # the later rescale to F_int, so normalise the cloud budget instead and
        # every source gets a model of the same quality.
        rpix = np.asarray(radii, float) / mpix
        area = np.pi * np.diff(rpix ** 2)
        wsum = float(np.sum(np.asarray(dens, float)[1:] * area)) if len(radii) > 1 else 0.0
        if wsum > 0:
            dens = np.clip(np.asarray(dens, float) * (self.cfg["cloud_budget"] / wsum),
                           0.0, self.cfg["max_dens"])

        vrot = rotation_curve(radii, vrot_max, self.cfg["r_turn_over_Rs"] * rs)

        cc = dict(self.cubecommon)
        cc["cdelts"] = [self.hdr["CDELT1"] / f, self.hdr["CDELT2"] / f, self.dv]
        gal = SimulatedGalaxyCube(axisDim=[csize, csize, self.nz], **cc)
        gal.define_galaxy(
            radii=radii, dens=dens, ishole=False,
            xpos=csize / 2.0, ypos=csize / 2.0,
            vsys=float(row["vsys_kms"]),
            inc=float(row["inc_deg"]), warpinc=False,
            pa=float(row["pa_deg"]), warppa=False,
            vrot=vrot, vdisp=float(row["vdisp_kms"]),
            z0=float(row["z0_arcsec"]), flare=False,
            vrad=0.0, radmotion=False,
        )
        param_file = self.outdir / "models" / f"{tag}_rings.txt"
        gal.savefile(str(param_file))

        folder = self.outdir / "bb_output" / tag
        folder.mkdir(parents=True, exist_ok=True)
        # pyBBarolo's run() writes galaxy_params.txt and emptycube.fits into the
        # CURRENT WORKING DIRECTORY and points BBarolo at them, so two injectors
        # started from the same directory can read each other's model. Run each
        # model build in its own folder. (This race was real but is NOT what
        # cost three fields a third of their sources: that was GALMOD crashing
        # on four-ring discs built on an oversampled grid; see min_rings.)
        import contextlib, os
        @contextlib.contextmanager
        def _in(d):
            prev = os.getcwd(); os.makedirs(d, exist_ok=True); os.chdir(d)
            try:
                yield
            finally:
                os.chdir(prev)
        log = str(Path(folder).resolve() / "bbarolo.log")
        with _in(str(folder)):
            gal.run(exe=self.a.bb_exe, outfolder=".", stdout=log)

        path = find_nonorm_model(folder)
        if path is None:
            self.fail("no_model")
            log = folder / "bbarolo.log"
            if log.exists():
                for line in log.read_text().strip().splitlines()[-12:]:
                    print("  [BBarolo]", line)
            try:
                print("  [BBarolo] files:", sorted(q.name for q in folder.iterdir())[:12])
            except OSError:
                pass
            return None, param_file
        with fits.open(path) as h:
            m = h[0].data.astype(np.float32)
        if m.ndim != 3:
            self.fail("model_ndim"); return None, param_file
        if m.shape[0] != self.nz:
            if m.shape[-1] == self.nz:
                m = np.moveaxis(m, -1, 0)
            else:
                self.fail("model_shape"); return None, param_file
        m[~np.isfinite(m)] = 0.0

        if f > 1:
            m = bin_down(m, f)
        return m, param_file

    def solve_vrot(self, row, tag):
        """Iterate v_rot until the measured W50 matches the design.

        W50 is a design axis, so a nominal v_rot is not sufficient: the
        realised width depends on inclination, curve shape, dispersion and
        channel width together.
        """
        target = float(row["W50_kms"])
        # A channelised W50 is quantised in units of the channel width, so a
        # fractional tolerance alone is unsatisfiable for narrow lines: with
        # dv = 44 km/s, a target of 73 km/s lies between achievable widths.
        # The channel-sized floor was needed when the width was quantised; the
        # interpolated measurement is continuous, and a 0.75-channel floor is
        # 16% of a 200 km/s target, wide enough that the first undershooting
        # model is accepted and the loop never runs.
        tol = max(self.cfg["w50_tol"] * target, 0.25 * abs(self.dv))
        # The arctan curve reaches about 0.77 of v_max at the half-mass radius,
        # so the population's v_rot = W50 / (2 sin i) is systematically low and
        # the secant spends its first BBarolo call discovering that. Each call
        # costs ~8 s of fixed overhead, which dominates the cloud sampling, so
        # the initial guess is worth more than the cloud budget.
        v = float(row["vrot_kms"]) / self.cfg["vrot_guess_factor"]
        hist = []
        best = None
        for it in range(self.cfg["w50_max_iter"]):
            m, pf = self.make_model(row, v, tag)
            if m is None:
                return None, None, hist
            w50 = measure_width(m.sum(axis=(1, 2)), self.vel, 0.50)
            hist.append(dict(iter=it, vrot=v, w50=w50))
            if not np.isfinite(w50) or w50 <= 0:
                self.fail("w50_unmeasurable"); return None, None, hist
            if best is None or abs(w50 - target) < abs(best[1] - target):
                best = (m, w50, v)
            if abs(w50 - target) <= tol:
                return m, v, hist
            # the same width twice means the target is between achievable
            # values and further iteration cannot help
            if len(hist) >= 2 and abs(hist[-1]["w50"] - hist[-2]["w50"]) < 1e-6:
                break
            v = float(np.clip(v * target / w50, 1.0, 5000.0))

        # Take the closest attempt rather than dropping the source: rejecting
        # here would preferentially discard narrow faint lines and bias the
        # design. The realised width is recorded as W50_realised_kms and is
        # what the analysis uses.
        m, w50, v = best
        if abs(w50 - target) <= 0.5 * target:
            self.fail("w50_quantised")
            return m, v, hist
        self.fail("w50_no_convergence")
        return None, None, hist

    # -- placement ----------------------------------------------------------

    def place(self, row, gal, zlo, zhi):
        """Add the model at the position the population chose.

        `zlo:zhi` is the channel range the model occupies; only those channels
        are tested and modified, so blanking elsewhere in the band is
        irrelevant to this source.
        """
        nz_g, ny_g, nx_g = gal.shape
        x0 = int(round(float(row["xpix"]) - nx_g / 2.0))
        y0 = int(round(float(row["ypix"]) - ny_g / 2.0))
        x1, y1 = x0 + nx_g, y0 + ny_g
        if x0 < 0 or y0 < 0 or x1 > self.nx or y1 > self.ny:
            self.fail("patch_off_cube"); return None

        before = self.cube[zlo:zhi, y0:y1, x0:x1].copy()
        bad = ~np.isfinite(before)
        if self.blank is not None:
            bad |= (before == self.blank)

        # A hard zero-blank test is unachievable where ~0.5% of voxels are
        # blanked independently per channel: a 40x40x20 patch holds ~32,000
        # voxels and will always touch one. Allow a small fraction, and zero
        # the model there rather than adding signal to blanked data.
        f = float(bad.mean())
        if f > self.cfg["max_blank_frac"]:
            self.fail("patch_too_blank"); return None
        return (x0, y0, x1, y1, before, bad)


def bin_down(cube, f):
    """Average an oversampled model back onto the cube's pixel grid."""
    if f <= 1:
        return cube
    nz, ny, nx = cube.shape
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    c = cube[:, :ny2, :nx2]
    # Jy/beam is a surface brightness and the beam is unchanged by the finer
    # grid, so the correct reduction is the mean, not the sum.
    return c.reshape(nz, ny2 // f, f, nx2 // f, f).mean(axis=(2, 4))


def find_nonorm_model(folder):
    cands = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.endswith(".fits") and "nonorm" in fn.lower():
                cands.append(os.path.join(root, fn))
    if not cands:
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if fn.endswith(".fits") and "mod" in fn.lower():
                    cands.append(os.path.join(root, fn))
    return sorted(cands)[0] if cands else None


# ---------------------------------------------------------------------------

DEFAULTS = dict(
    rout_over_rHI=1.3, r_turn_over_rout=0.15, r_turn_over_Rs=1.0,
    vrot_guess_factor=0.77,
    density_profile="exponential",
    model_csize_factor=3.0, max_model_csize=600,
    model_oversample=1, max_oversample=16, beam_margin=3.0,
    min_ring_spacing_pix=1.05, min_dHI_arcsec=5.0,
    # BBarolo 1.7 corrupts its heap ("double free or corruption",
    # "corrupted size vs. prev_size") on every disc built with four rings on
    # an oversampled grid: 29 of 29 such sources in the J0540 test set, and
    # none of the other 31. Six rings rescued 26 of them, and left the models
    # of sources that already worked unchanged within GALMOD noise. The rest
    # were below 0.13 beams, now excluded by dratio_min = 0.2.
    min_rings=6, max_rings=60, min_rout_model_pix=15.0,
    w50_tol=0.10, w50_max_iter=9, max_blank_frac=0.05, max_flux_lost=0.10,
    # GALMOD samples the disc with N discrete clouds, so the model carries
    # Poisson noise of 1/sqrt(N) on the integrated profile. 2000 clouds
    # gives 2.2%, inside both the 5% flux and 5% W50 tolerances, at ~10 s
    # per source. 20000 gave 0.7% and 100 s -- a 40x cost for precision
    # neither check can use.
    cloud_budget=5.0e2, max_dens=1.0e5,
    flux_tol=0.05, center_tol_pix=2.0, vcent_tol_kms=None,
    model_edge_frac_max=0.02, blank_value=None,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", required=True)
    ap.add_argument("--substrate", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bb-exe", help="BBarolo executable (not needed with --dry-model)")
    ap.add_argument("--limit", type=int, help="Inject only the first N rows.")
    ap.add_argument("--dry-model", action="store_true",
                    help="Write ring tables and report geometry; do not run BBarolo.")
    ap.add_argument("--keep-models", action="store_true")
    for k, v in DEFAULTS.items():
        if v is None:
            ap.add_argument(f"--{k.replace('_', '-')}", default=None)
        elif isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", action="store_true")
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    cfg = {k: getattr(a, k) for k in DEFAULTS}
    for k in ("blank_value", "vcent_tol_kms"):
        if cfg[k] is not None:
            cfg[k] = float(cfg[k])

    pop = Table.read(a.population, format="ascii.ecsv")
    if a.limit:
        pop = pop[:a.limit]

    inj = Injector(a, cfg)
    if cfg["vcent_tol_kms"] is None:
        cfg["vcent_tol_kms"] = 1.0 * abs(inj.dv)

    # ---- dry model: geometry only ----------------------------------------
    if a.dry_model:
        rows = []
        for r in pop:
            radii, dens, r_out, mpix, f, rs = build_rings(r, inj.geo, cfg)
            cs = max(int(np.ceil((cfg["model_csize_factor"] * r_out
                                  + cfg["beam_margin"] * inj.geo["bmaj_arcsec"])
                                 / mpix)), 40)
            rows.append(dict(id=int(r["id"]), dHI_arcsec=float(r["dHI_arcsec"]),
                             R_out_arcsec=r_out, n_rings=len(radii),
                             oversample=f, R_out_model_pix=r_out / mpix,
                             csize=cs, csize_arcsec=cs * mpix,
                             csize_beams=cs * mpix / inj.geo["bmaj_arcsec"]))
        t = Table(rows=rows)
        print(f"  model geometry for {len(t)} sources")
        for c, f in [("dHI_arcsec", ".2f"), ("R_out_arcsec", ".2f"),
                     ("R_out_model_pix", ".2f")]:
            t[c].info.format = f
        print(t[:10])
        print(f"\n  n_rings      min {t['n_rings'].min()}  max {t['n_rings'].max()}")
        print(f"  oversample   min {t['oversample'].min()}  max {t['oversample'].max()}")
        print(f"  csize [pix]  min {t['csize'].min()}  max {t['csize'].max()}")
        print(f"  csize [beam] min {t['csize_beams'].min():.1f}  "
              f"max {t['csize_beams'].max():.1f}")
        tiny = int(np.sum(t["R_out_model_pix"] < 2 * cfg["min_ring_spacing_pix"]))
        narrow = int(np.sum(t["csize_beams"] < 2.0))
        if tiny:
            print(f"\n  WARNING: {tiny} sources still have R_out below two ring "
                  "spacings; raise --max-oversample.")
        if narrow:
            print(f"\n  WARNING: {narrow} cutouts are under two beams wide; raise "
                  "--beam-margin or they will be rejected as model_cutout_tight.")
        out = Path(a.outdir) / "model_geometry.ecsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        t.write(out, format="ascii.ecsv", overwrite=True)
        print(f"\n  wrote {out}")
        return

    if not a.bb_exe:
        # pip-installed pyBBarolo ships its own BBarolo executable: use it
        try:
            import inspect
            from pyBBarolo.utils import SimulatedGalaxyCube
            a.bb_exe = inspect.signature(SimulatedGalaxyCube.run).parameters["exe"].default
        except Exception:
            a.bb_exe = None
        if not a.bb_exe or not Path(a.bb_exe).exists():
            sys.exit("no BBarolo executable: install pyBBarolo (pip install pyBBarolo) "
                     "or set tools.bbarolo")

    # ---- inject -----------------------------------------------------------
    truth = []
    for i, row in enumerate(pop):
        tag = f"src_{int(row['id']):04d}"
        print(f"\n[{i+1}/{len(pop)}] {tag}  SNR {row['snr_design']:.2f}  "
              f"F {row['Fint_Jykms']:.4g}  W50 {row['W50_kms']:.0f}  "
              f"d/beam {row['dHI_over_beam']:.2f}", flush=True)

        gal, vrot, hist = inj.solve_vrot(row, tag)
        if gal is None:
            print("  [SKIP] model/W50 failed")
            continue

        mom0 = gal.sum(axis=0)
        ef = border_flux_fraction(mom0, 2)
        if np.isfinite(ef) and ef > cfg["model_edge_frac_max"]:
            inj.fail("model_cutout_tight")
            print(f"  [SKIP] edge flux fraction {ef:.3f}")
            continue

        f_model = float(np.nansum(gal)) / inj.beam_area_pix * abs(inj.dv)
        if not np.isfinite(f_model) or f_model <= 0:
            inj.fail("model_flux"); print("  [SKIP] model flux <= 0"); continue
        gal = gal * (float(row["Fint_Jykms"]) / f_model)

        # The model occupies a dozen channels of a 477-channel cube; carrying
        # the rest means blanking far from the galaxy disqualifies the position.
        prof = np.nansum(gal, axis=(1, 2))
        pk = float(np.nanmax(prof)) if prof.size else 0.0
        occ = np.where(prof > 1e-4 * pk)[0] if pk > 0 else np.array([], int)
        if occ.size == 0:
            inj.fail("model_empty"); print("  [SKIP] model has no signal"); continue
        zlo = max(int(occ[0]) - 2, 0)
        zhi = min(int(occ[-1]) + 3, inj.nz)
        gal = gal[zlo:zhi]
        vel = inj.vel[zlo:zhi]

        p = inj.place(row, gal, zlo, zhi)
        if p is None:
            print("  [SKIP] placement")
            continue
        x0, y0, x1, y1, before, bad = p
        lost = 0.0
        if bad.any():
            tot = float(np.nansum(gal))
            lost = float(np.nansum(gal[bad]) / tot) if tot > 0 else 0.0
            gal = np.where(bad, 0.0, gal)
        inj.cube[zlo:zhi, y0:y1, x0:x1] += gal
        delta = inj.cube[zlo:zhi, y0:y1, x0:x1] - before

        # What the cube SHOULD have gained is the model after blanked voxels
        # were zeroed. Comparing against the design flux instead flags every
        # source that lost flux to blanking as a bookkeeping failure, and those
        # sit near the primary-beam edge where the noise is highest -- exactly
        # the part of the field the selection function is about.
        f_expect = float(np.nansum(gal)) / inj.beam_area_pix * abs(inj.dv)
        f_meas = float(np.nansum(delta)) / inj.beam_area_pix * abs(inj.dv)
        rel = abs(f_meas - f_expect) / f_expect if f_expect > 0 else float("inf")
        if not np.isfinite(rel) or rel > cfg["flux_tol"]:
            inj.cube[zlo:zhi, y0:y1, x0:x1] = before
            inj.fail("flux_rollback"); print(f"  [ROLLBACK] dF/F = {rel:.3f}"); continue

        # A source can still be too truncated to be a fair test of the finder.
        if lost > cfg["max_flux_lost"]:
            inj.cube[zlo:zhi, y0:y1, x0:x1] = before
            inj.fail("model_truncated")
            print(f"  [ROLLBACK] {lost:.1%} of the model fell on blanked voxels")
            continue

        m0 = delta.sum(axis=0)
        cx, cy = moment_centroid_2d(m0)
        dcen = float(np.hypot(x0 + cx - row["xpix"], y0 + cy - row["ypix"]))
        if np.isfinite(dcen) and dcen > cfg["center_tol_pix"]:
            inj.cube[zlo:zhi, y0:y1, x0:x1] = before
            inj.fail("centre_rollback"); print(f"  [ROLLBACK] dcen = {dcen:.2f} pix"); continue

        spec = delta.sum(axis=(1, 2))
        z = np.arange(zhi - zlo, dtype=float)
        vcen = float(np.interp(np.sum(z * spec) / np.sum(spec), z, vel)) \
            if np.sum(spec) > 0 else np.nan
        if np.isfinite(vcen) and abs(vcen - row["vsys_kms"]) > cfg["vcent_tol_kms"]:
            inj.cube[zlo:zhi, y0:y1, x0:x1] = before
            inj.fail("vcent_rollback")
            print(f"  [ROLLBACK] dv = {abs(vcen - row['vsys_kms']):.1f} km/s"); continue

        w50r = measure_width(spec, vel, 0.50)
        w20r = measure_width(spec, vel, 0.20)
        dip = profile_dip(spec)

        # the 3-sigma isophote of this source at THIS position: it depends on
        # the local noise, so it is not a property of the galaxy alone
        nch = int(np.sum(spec > 0.2 * np.nanmax(spec))) if np.nanmax(spec) > 0 else 1
        # m0 is a sum over channels in Jy/beam; the threshold is in
        # Jy/beam km/s, so the moment map needs the channel width too
        d3a, d3m = isophotal_size(m0 * abs(inj.dv),
                                  float(row["sigma_chan_Jybeam"]), inj.dv,
                                  nch, inj.geo["pix_arcsec"])
        conc = solve_concentration(float(row["sigma_HI_Msun_pc2"])) \
            if "sigma_HI_Msun_pc2" in pop.colnames else float("nan")

        rec = {c: row[c] for c in pop.colnames}
        rec.update(d3sigma_area_arcsec=d3a, d3sigma_moment_arcsec=d3m,
                   concentration_RHI_over_Rs=conc, n_chan_above_20pct=nch,
                   injected=True, vrot_solved_kms=vrot, w50_iters=len(hist),
                   W50_realised_kms=w50r, W20_realised_kms=w20r,
                   profile_dip=dip,
                   Fint_realised_Jykms=f_meas, centroid_offset_pix=dcen,
                   vcent_realised_kms=vcen, model_edge_frac=ef,
                   patch_blank_frac=float(bad.mean()), model_flux_lost=lost,
                   patch_zlo=zlo, patch_nz=zhi - zlo,
                   patch_x0=x0, patch_y0=y0, patch_nx=x1 - x0, patch_ny=y1 - y0)
        truth.append(rec)
        print(f"  [OK] W50 {w50r:.0f} (target {row['W50_kms']:.0f}, "
              f"{len(hist)} iter)  dF/F {rel:.3f}  dcen {dcen:.2f} pix")

        if not a.keep_models:
            shutil.rmtree(inj.outdir / "bb_output" / tag, ignore_errors=True)

    # ---- write ------------------------------------------------------------
    outdir = Path(a.outdir)
    cube_path = outdir / "substrate_plus_galaxies.fits"
    fits.PrimaryHDU(inj.cube, inj.hdr).writeto(cube_path, overwrite=True)

    t = Table(rows=truth) if truth else Table()
    truth_path = outdir / "injected_galaxies_truth.ecsv"
    t.write(truth_path, format="ascii.ecsv", overwrite=True)

    n = len(pop)
    print("\n" + "=" * 74)
    print(f"INJECTED {len(truth)} / {n}")
    print("=" * 74)
    for k, v in sorted(inj.reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26s} {v}")
    if truth:
        w = np.array([r["W50_realised_kms"] for r in truth], float)
        d = np.array([r["W50_kms"] for r in truth], float)
        f = np.array([r["Fint_realised_Jykms"] for r in truth], float)
        ft = np.array([r["Fint_Jykms"] for r in truth], float)
        print(f"\n  W50 realised / design : median {np.median(w/d):.4f}  "
              f"[{np.percentile(w/d,16):.4f}, {np.percentile(w/d,84):.4f}]")
        print(f"  F   realised / design : median {np.median(f/ft):.4f}  "
              f"[{np.percentile(f/ft,16):.4f}, {np.percentile(f/ft,84):.4f}]")
    print(f"\n  wrote {cube_path}")
    print(f"  wrote {truth_path}")

    (outdir / "inject.temba.json").write_text(json.dumps(
        dict(n_requested=n, n_injected=len(truth), reasons=inj.reasons,
             config={k: v for k, v in cfg.items()}), indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
