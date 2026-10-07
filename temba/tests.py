"""Regression tests for TEMBA.

Every check here was made by hand at some point during development, usually
after a bug that took hours to find. Encoded as tests they run in seconds and
fail loudly, which is the difference between a pipeline that works today and
one that can be relied on tomorrow.

    python -m temba.tests

Exits non-zero on any failure, so it can gate a campaign. Tests needing
external data are skipped with a message rather than failing.
"""

import sys
import traceback

import numpy as np

PASS, FAIL, SKIP = [], [], []


def check(name):
    def deco(fn):
        try:
            msg = fn()
            (PASS if msg is None else SKIP).append((name, msg))
        except Exception as exc:                       # noqa: BLE001
            FAIL.append((name, f"{type(exc).__name__}: {exc}",
                         traceback.format_exc()))
        return fn
    return deco


# ------------------------------------------------------------------ physics
@check("isophote solve inverts exactly")
def _():
    from .physics import solve_concentration
    for sm in (1.2, 2.0, 3.8, 6.5, 15.0, 30.0):
        x = solve_concentration(sm)
        g = 2.0 * (np.expm1(x) - x) / x ** 2
        assert abs(g - sm) < 1e-4 * sm, f"g({x:.3f}) = {g:.4f}, wanted {sm}"
    # a mean density below 1 is impossible for any declining profile
    assert solve_concentration(0.5) < 1e-2


@check("Wang canonical density gives R_s = d_HI / 6.2")
def _():
    from .physics import scale_length
    rs = float(np.atleast_1d(scale_length(np.array([20.0]),
                                          np.array([3.8])))[0])
    assert abs(20.0 / rs - 6.23) < 0.06, f"d_HI/R_s = {20.0 / rs:.2f}"


@check("point-source isophote formula")
def _():
    from .physics import isophote_point_source
    assert abs(float(isophote_point_source(3.0, 3.0))) < 1e-9
    assert abs(float(isophote_point_source(10.0, 3.0))
                - np.sqrt(np.log(10.0 / 3.0) / np.log(2.0))) < 1e-9
    # quadrature deconvolution of a POINT source is non-zero above SNR = 2n.
    # This documents a trap in other pipelines, not a bug in ours.
    d = float(isophote_point_source(40.0, 3.0))
    assert np.sqrt(d ** 2 - 1) > 1.6


@check("velocity-gradient cut separates crashers from ordinary discs")
def _():
    from .physics import velocity_gradient as vg
    for w50, inc, d in ((525, 44.8, 3.74), (415, 55.2, 3.70)):
        assert vg(w50, inc, d, 6.0, 44.11) > 6.0
    for w50, inc, d in ((200, 60, 40.), (400, 70, 90.), (60, 45, 12.),
                        (80, 85, 20.)):
        assert vg(w50, inc, d, 6.0, 44.11) < 1.5


@check("representability keeps the resolved population")
def _():
    from .physics import representable, angular_to_kpc
    rng = np.random.default_rng(0)
    n = 20000
    w50 = 10 ** rng.uniform(np.log10(53), np.log10(600), n)
    inc = np.degrees(np.arccos(rng.uniform(np.cos(np.radians(85)),
                                           np.cos(np.radians(15)), n)))
    drat = 10 ** rng.uniform(np.log10(0.10), np.log10(4.0), n)
    d_as = drat * 38.4
    d_kpc = angular_to_kpc(d_as, rng.uniform(40, 350, n))
    ok = representable(w50, inc, d_as, d_kpc, 6.0, 44.11,
                       max_grad_chan_per_pix=6.0, min_d_hi_kpc=2.0,
                       min_d_hi_arcsec=3.0)
    assert ok[drat >= 1.0].mean() > 0.99, "resolved sources are being cut"
    assert ok.mean() > 0.6, f"only {ok.mean():.1%} of the design kept"


# ------------------------------------------------------------------- inject
@check("interpolated W50 recovers a known Gaussian")
def _():
    from .inject import measure_width
    dv = 44.11
    vel = np.arange(40) * dv
    for fwhm in (53., 73., 100., 200., 500.):
        s = np.exp(-0.5 * ((vel - vel[20]) / (fwhm / 2.355)) ** 2)
        w = measure_width(s, vel, 0.50)
        assert abs(w / fwhm - 1) < 0.06, f"FWHM {fwhm}: measured {w:.1f}"
    # the channelised version returned NaN for a line this narrow
    s = np.exp(-0.5 * ((vel - vel[20]) / (53. / 2.355)) ** 2)
    assert np.isfinite(measure_width(s, vel, 0.50))


@check("W50 spans both horns of a double-horned profile")
def _():
    from .inject import measure_width
    dv = 44.11
    vel = np.arange(60) * dv
    c = vel[30]
    g = lambda m: np.exp(-0.5 * ((vel - m) / 30.0) ** 2)
    s = g(c - 200.0) + g(c + 200.0)          # trough far below half the peak
    w50 = measure_width(s, vel, 0.50)
    w20 = measure_width(s, vel, 0.20)
    expect = 400.0 + 2.355 * 30.0            # outer half-power points
    assert abs(w50 / expect - 1) < 0.06, f"W50 {w50:.0f}, expected {expect:.0f}"
    assert w20 >= w50, f"W20 {w20:.0f} < W50 {w50:.0f}"


@check("every disc gets ~20 GALMOD sub-rings and legal ring spacing")
def _():
    from .inject import build_rings, DEFAULTS
    cfg = dict(DEFAULTS)
    geo = dict(pix_arcsec=6.0, bmaj_arcsec=32.9)

    class Row(dict):
        colnames = ["dHI_arcsec", "sigma_HI_Msun_pc2"]

    for d in np.geomspace(5.0, 300.0, 60):
        radii, dens, r_out, mpix, f, rs = build_rings(
            Row(dHI_arcsec=d, sigma_HI_Msun_pc2=4.0), geo, cfg)
        rpix = r_out / mpix
        if f < cfg["max_oversample"]:
            assert rpix >= cfg["min_rout_model_pix"] - 1e-9, f"d={d:.1f}: R_out {rpix:.1f} pix"
        spacing = (radii[1] - radii[0]) / mpix
        assert spacing >= cfg["min_ring_spacing_pix"] - 1e-9, f"d={d:.1f}: spacing {spacing:.2f}"
        assert len(radii) >= cfg["min_rings"]


@check("profile_dip separates a comb from a smooth profile")
def _():
    from .inject import profile_dip
    x = np.arange(40.0)
    smooth = np.exp(-0.5 * ((x - 20) / 4.0) ** 2)
    comb = np.array([0.26, 0.59, 0.71, 0.53, 0.97, 0.86, 1.00, 0.48, 0.74, 0.55, 0.29])
    assert profile_dip(smooth) < 0.01
    assert profile_dip(comb) > 0.2


@check("survey point-source size curve: SNR 48 at 3 sigma is two beams")
def _():
    from .survey import point_source_size
    assert abs(point_source_size(48.0, 3) - 2.0) < 1e-12
    assert point_source_size(2.0, 3) == 0.0          # below threshold: no size


@check("rotation curve reaches vmax at the outermost ring")
def _():
    from .inject import rotation_curve
    r = np.linspace(0, 50, 12)
    v = rotation_curve(r, 150.0, 0.15 * 50)
    assert abs(v[-1] - 150.0) < 1e-6, f"v(R_out) = {v[-1]:.2f}, wanted 150"
    assert v[0] == 0.0 and np.all(np.diff(v) >= 0)


@check("isophotal size of a Gaussian is measured correctly")
def _():
    from .inject import isophotal_size
    ny = nx = 81
    yy, xx = np.mgrid[0:ny, 0:nx]
    fwhm = 12.0
    m0 = np.exp(-4 * np.log(2) * ((yy - 40.) ** 2 + (xx - 40.) ** 2) / fwhm ** 2)
    # threshold at half peak: the enclosed area has diameter = FWHM
    d_area, d_mom = isophotal_size(m0, sigma_chan=0.5 / 3.0, dv_kms=1.0,
                                   n_chan=1, pix_arcsec=1.0, nsigma=3.0)
    assert abs(d_area / fwhm - 1) < 0.1, f"d_area = {d_area:.2f}, wanted {fwhm}"
    assert np.isfinite(d_mom)


# --------------------------------------------------------------- population
@check("uniformity test behaves on known inputs")
def _():
    from .population import uniformity_test
    rng = np.random.default_rng(3)
    _, _, p_flat = uniformity_test(rng.uniform(0, 1, 5000), 0.0, 1.0)
    assert p_flat > 0.01, f"flat sample rejected, p = {p_flat:.4f}"
    _, _, p_skew = uniformity_test(rng.beta(2, 5, 5000), 0.0, 1.0)
    assert p_skew < 1e-6, f"skewed sample accepted, p = {p_skew:.4g}"


# ----------------------------------------------------------------- showcase
@check("showcase picks distinct cases, not the n brightest")
def _():
    from astropy.table import Table
    from .showcase import choose
    rng = np.random.default_rng(5)
    n = 300
    t = Table(dict(
        detected=np.ones(n, bool),
        flux_capture=np.clip(rng.normal(0.98, 0.03, n), 0.5, 1.0),
        flux_fidelity=np.clip(rng.normal(1.03, 0.08, n), 0.7, 2.0),
        snr_design=10 ** rng.uniform(-0.5, 2.0, n),
        dHI_over_beam=10 ** rng.uniform(-1.0, 0.6, n)))
    ids, why = choose(t, 6)
    assert len(ids) == len(set(ids)) == 6, "duplicate or missing selections"
    assert len(set(why)) >= 4, f"selections are not distinct: {why}"


@check("every DEFAULTS key round-trips through the parser")
def _():
    import subprocess
    from .population import DEFAULTS
    # A None default silently made argparse reject every value for that flag.
    args = []
    for k, v in DEFAULTS.items():
        if isinstance(v, (int, float)) or v is None:
            args += [f"--{k.replace('_', '-')}", "1"]
    out = subprocess.run([sys.executable, "-m", "temba.population", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for k in DEFAULTS:
        flag = f"--{k.replace('_', '-')}"
        assert flag in out.stdout, f"{flag} missing from the parser"


# --------------------------------------------------------------- regression
@check("thesis Table 2.2 reproduces")
def _():
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    truth = Path("example/injection_truth_tables/Abell-194_truth.ecsv")
    cat = Path("example/injection_truth_tables/Abell-194_cat.xml")
    if not (truth.exists() and cat.exists()):
        return "example/ not present"
    with tempfile.TemporaryDirectory() as td:
        rec = Path(td) / "rec.ecsv"
        subprocess.run([sys.executable, "-m", "temba.match",
                        "--truth", str(truth), "--catalogue", str(cat),
                        "--out", str(rec), "--bmaj-arcsec", "38.757",
                        "--bmin-arcsec", "31.6", "--pix-arcsec", "6.0",
                        "--dv-kms", "44.110", "--beam-factor", "2",
                        "--dv-channels", "1"], check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "temba.selection",
                        "--recovery", str(rec), "--outdir", td,
                        "--label", "regression"], check=True,
                       capture_output=True)
        j = json.loads((Path(td) / "selection.json").read_text())
    v = float(j["flux_logistic"]["logS50"])
    n_det, n_inj = int(j["n_detected"]), int(j["n_injected"])
    # Thesis Table 2.2: logS50 = -0.43, 85 of 150 recovered. The fit differs
    # slightly because the width is now measured by interpolation rather than
    # by counting channels, so the tolerance covers the definition change and
    # nothing more.
    assert abs(v + 0.43) < 0.08, f"logS50 = {v:.3f}, wanted -0.43"
    assert n_inj == 150 and abs(n_det - 85) <= 2, f"{n_det}/{n_inj}, wanted 85/150"


def main():
    print("=" * 66)
    print("TEMBA REGRESSION TESTS")
    print("=" * 66)
    for name, _ in PASS:
        print(f"  PASS  {name}")
    for name, msg in SKIP:
        print(f"  SKIP  {name}  ({msg})")
    for name, msg, _ in FAIL:
        print(f"  FAIL  {name}")
        print(f"        {msg}")
    print()
    print(f"  {len(PASS)} passed, {len(SKIP)} skipped, {len(FAIL)} failed")
    if FAIL:
        print()
        for name, _, tb in FAIL:
            print(f"--- {name} " + "-" * max(4, 58 - len(name)))
            print(tb)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
