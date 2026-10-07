"""Derived physics for TEMBA, and the limits of what can be modelled.

This module holds the quantities the injected population derives rather than
draws, and the two cuts that keep the design inside the region 3D-Barolo can
actually represent. Keeping them here rather than scattered through population
and inject means the assumptions are in one place, which is where a reader
looking for them will expect to find them.

Nothing here touches a cube or a file: it is arithmetic on the design
parameters, and everything in it is covered by temba.tests.
"""

import numpy as np

# 2.356e5 D_L^2 F_int, with D_L in Mpc and F_int in Jy km/s (Roberts 1962)
MHI_PER_JYKMS_MPC2 = 2.356e5
# G in kpc (km/s)^2 / Msun
G_KPC = 4.30091e-6
ARCSEC_PER_RAD = 206264.806


def hi_mass(f_int_jykms, dl_mpc):
    """HI mass from integrated flux and luminosity distance."""
    return MHI_PER_JYKMS_MPC2 * np.asarray(dl_mpc, float) ** 2 * \
        np.asarray(f_int_jykms, float)


def angular_to_kpc(theta_arcsec, da_mpc):
    """Angular size to physical size, using the angular diameter distance."""
    return np.asarray(theta_arcsec, float) / ARCSEC_PER_RAD * \
        np.asarray(da_mpc, float) * 1e3


def solve_concentration(sigma_mean, lo=1e-3, hi=30.0, tol=1e-6):
    """Return x = R_HI / R_s for an exponential disc of this mean density.

    d_HI is defined as the diameter where Sigma_HI = 1 Msun/pc^2, so the
    profile is not free. For an exponential disc,

        Sigma_mean / Sigma(R_HI) = g(x) = 2 (e^x - 1 - x) / x^2

    and requiring Sigma(R_HI) = 1 fixes x from the mean density alone. g is
    monotonic with g -> 1 as x -> 0, which is also why a mean density below
    1 Msun/pc^2 is impossible for any declining profile.
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
    """Exponential scale length, solved from the mean density when known."""
    if sigma_mean is None:
        return np.maximum(np.asarray(d_hi_arcsec, float) / 5.0, 1e-6)
    x = np.array([solve_concentration(s) for s in np.atleast_1d(sigma_mean)])
    rs = 0.5 * np.atleast_1d(d_hi_arcsec) / np.maximum(x, 1e-6)
    return np.maximum(rs, 1e-6).reshape(np.shape(d_hi_arcsec))


def vrot_from_w50(w50_kms, inc_deg, vdisp_kms=9.0):
    """Rotation speed implied by a linewidth, with turbulence deconvolved.

    W50^2 = (2 v_rot sin i)^2 + (2 vdisp)^2 is the usual first-order form; it
    is only a starting guess, because the realised width also depends on the
    density weighting across rings and on the channel response. The injector
    iterates on the measured width rather than trusting this.
    """
    w = np.asarray(w50_kms, float)
    si = np.sin(np.radians(np.asarray(inc_deg, float)))
    core = np.maximum((0.5 * w) ** 2 - (2.0 * vdisp_kms) ** 2, 1.0)
    return np.sqrt(core) / np.maximum(si, 1e-3)


def velocity_gradient(w50_kms, inc_deg, d_hi_arcsec, pix_arcsec, dv_kms,
                      vdisp_kms=9.0):
    """Velocity change across one spatial pixel, in channels.

    This is what decides whether a source can be modelled at all. GALMOD
    populates the disc with discrete clouds and smears each into the cube; when
    the projected rotation changes by many channels between adjacent pixels the
    cloud bookkeeping overruns, and BBarolo exits with "double free or
    corruption" or "corrupted size vs. prev_size" rather than an error message.

    Measured on the sources that crashed: 20 to 25 channels per pixel, against
    0.6 to 0.9 for ordinary discs -- a factor of twenty, so the cut is not
    finely balanced. The combination arises because W50 and d_HI are drawn
    independently, which is deliberate, so a 500 km/s line can land on a 4
    arcsec disc; such an object is not a galaxy, and no survey would detect one
    as anything but a point source in any case.
    """
    v = vrot_from_w50(w50_kms, inc_deg, vdisp_kms)
    si = np.sin(np.radians(np.asarray(inc_deg, float)))
    d_pix = np.maximum(np.asarray(d_hi_arcsec, float) / pix_arcsec, 1e-6)
    return 2.0 * v * si / d_pix / abs(dv_kms)


def representable(w50_kms, inc_deg, d_hi_arcsec, d_hi_kpc, pix_arcsec, dv_kms,
                  max_grad_chan_per_pix=2.0, min_d_hi_kpc=2.0,
                  min_d_hi_arcsec=5.0, vdisp_kms=9.0):
    """Boolean mask: can this combination of parameters be modelled?

    Two independent limits, both numerical rather than astrophysical, and both
    stated in the paper as limits of the experiment:

      * the velocity gradient across a pixel (see velocity_gradient)
      * a size floor, in physical and angular units. These cubes reach
        z ~ 0.09, so a fixed angular floor is a different physical size at
        either end of the band; below about 2 kpc the scale length falls to a
        fraction of a model pixel and the ring geometry degenerates.

    Neither cut removes anything a survey could resolve: every source excluded
    is a point source at this beam, and the completeness at fixed flux is
    therefore unaffected.
    """
    grad = velocity_gradient(w50_kms, inc_deg, d_hi_arcsec, pix_arcsec, dv_kms,
                             vdisp_kms)
    ok = grad <= max_grad_chan_per_pix
    ok &= np.asarray(d_hi_kpc, float) >= min_d_hi_kpc
    ok &= np.asarray(d_hi_arcsec, float) >= min_d_hi_arcsec
    return ok


def isophote_point_source(snr_int, nsigma=3.0):
    """d_nsigma / theta_beam for an unresolved source.

    A point source convolved with a Gaussian beam of FWHM theta has
    I(r) = I_peak exp(-4 ln2 r^2 / theta^2), and its peak moment-0 brightness
    is F_int, so I_peak / sigma_mom0 is exactly SNR_int. Setting
    I = nsigma sigma_mom0 gives

        d_nsigma / theta_beam = sqrt( ln(SNR_int / nsigma) / ln 2 )

    which is why the measured size of an unresolved source depends on its
    signal-to-noise and therefore on where in the field it sits, and why
    subtracting the beam in quadrature from an isophotal size manufactures a
    finite size for a true point source at any SNR > 2 nsigma.
    """
    s = np.asarray(snr_int, float)
    return np.sqrt(np.clip(np.log(s / nsigma) / np.log(2.0), 0.0, None))
