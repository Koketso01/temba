"""
temba.match -- match recovered sources to injected ones, and decompose the
recovered flux density.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

This module answers two questions per injected source, not one.

1. Was it detected?
   Recovery follows the thesis definition: detected by the same SoFiA-2
   configuration, passing the same reliability selection as the science
   catalogue, and uniquely matched in position and velocity. Three things
   differ from v0.5's match_recovery:

     * Centroid columns (ra, dec, v_rad), never the peak-voxel columns.
       v0.5 preferred ra_peak/v_rad_peak, which sit on a horn of a
       double-horned profile and so fail the velocity gate in proportion to
       linewidth. On the thesis-era Abell-194 catalogue that alone takes the
       recovery from 57% to 32% and manufactures a spurious W50 dependence.

     * The velocity gate scales with the source: max(2 channels, 0.5 W20).
       A fixed one-channel tolerance is itself a linewidth-dependent
       selection, imposed by the matcher rather than by the source finder.

     * Global optimal assignment (scipy linear_sum_assignment) instead of a
       greedy loop in velocity order, so which source claims a contested
       counterpart does not depend on the order of iteration.

   Where the SoFiA mask is available, 3-D containment of the injected voxel
   is recorded alongside the positional match, because it is unambiguous.

2. How much of its flux came back, and what is that flux made of?
   Three quantities, kept separate:

       F_inj                  the truth, exact by construction
       F_model_in_mask        injected signal enclosed by SoFiA's mask
       f_sum                  what the catalogue reports

   giving

       capture  = F_model_in_mask / F_inj        bounded by 1
       fidelity = f_sum / F_inj                  unbounded
       noise_in_mask = f_sum - F_model_in_mask   mask growth into noise

   v0.5 reported only a single ratio and got 0.69 in one run and 1.53 in
   another; both were right, and they were measuring different things.
   F_model_in_mask is obtained by differencing the injected cube against the
   substrate inside the mask, so it needs no assumption about the profile.

Units
-----
SoFiA's f_sum is in Jy/beam summed over voxels unless parameter.physical is
set, in which case it is already an integrated flux. The VOTable carries the
unit, so it is read rather than guessed; v0.5's to_jykms() inferred units from
the magnitude of the numbers, which is not reliable.

Usage
-----
    python -m temba.match \
        --truth     runs/Abell-194/run_0001/injection/injected_galaxies_truth.ecsv \
        --catalogue runs/Abell-194/run_0001/sofia/Abell-194_run_0001_cat.xml \
        --mask      runs/Abell-194/run_0001/sofia/Abell-194_run_0001_mask.fits \
        --injected  runs/Abell-194/run_0001/injection/substrate_plus_galaxies.fits \
        --substrate substrates/Abell-194_substrate.fits \
        --out       runs/Abell-194/run_0001/products/recovery.ecsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

C_KMS = 299792.458


# ---------------------------------------------------------------------------

def pick(t, cands, required=False, what=""):
    low = {c.lower(): c for c in t.colnames}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise SystemExit(f"No column for {what or cands[0]}. Have: {t.colnames}")
    return None


def col(t, name, default=np.nan):
    if name is None:
        return np.full(len(t), default, float)
    return np.asarray(t[name], float)


def velocity_axis(hdr):
    crval = float(hdr.get("CRVAL3", 0.0)); cdelt = float(hdr.get("CDELT3", 1.0))
    crpix = float(hdr.get("CRPIX3", 1.0)); nz = int(hdr.get("NAXIS3", 0))
    ctype = str(hdr.get("CTYPE3", "")).upper(); cunit = str(hdr.get("CUNIT3", "")).strip().lower()
    world = (np.arange(nz, dtype=float) + 1.0 - crpix) * cdelt + crval
    if "FREQ" in ctype or "hz" in cunit:
        sc = {"": 1.0, "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(cunit, 1.0)
        rest = float(hdr.get("RESTFRQ", hdr.get("RESTFREQ", 1.42040575177e9)))
        vel = C_KMS * (rest - world * sc) / rest
        return vel, float(np.median(np.diff(vel)))
    if "km" in cunit:
        return world, cdelt
    if "m" in cunit or abs(cdelt) > 1e3:
        return world / 1e3, cdelt / 1e3
    return world, cdelt


def fsum_to_jykms(cat, fsum_col, dv_kms, beam_area_pix):
    """Convert the catalogue's f_sum to Jy km/s, using its declared unit.

    Returns (values, description). SoFiA writes physical units when
    parameter.physical = true, and Jy/beam-summed otherwise. Guessing from the
    magnitude of the numbers, as v0.5 did, silently produces factors of the
    beam area or the channel width.
    """
    v = np.asarray(cat[fsum_col], float)
    unit = ""
    try:
        unit = str(cat[fsum_col].unit or "")
    except Exception:
        pass
    u = unit.replace(" ", "").lower()

    if "beam" in u:
        return v * dv_kms / beam_area_pix, f"f_sum [{unit}] * dv / beam_area"
    if "m/s" in u or "m.s-1" in u:
        return v / 1e3, f"f_sum [{unit}] / 1000"
    if "km/s" in u or "km.s-1" in u:
        return v, f"f_sum [{unit}] as-is"
    # no usable unit: fall back on the raw-sum convention and say so loudly
    return (v * dv_kms / beam_area_pix,
            f"f_sum unit '{unit}' unrecognised; ASSUMED Jy/beam-summed")


def angsep(ra1, dec1, ra2, dec2):
    d2r = np.pi / 180.0
    dra = (np.asarray(ra2, float) - float(ra1)) * d2r * np.cos(float(dec1) * d2r)
    ddec = (np.asarray(dec2, float) - float(dec1)) * d2r
    return np.hypot(dra, ddec) * (180.0 / np.pi) * 3600.0


def assign(t_ra, t_dec, t_v, t_w20, r_ra, r_dec, r_v, rmax, dv_floor):
    """Globally optimal one-to-one assignment within a per-source gate."""
    nt, nr = len(t_ra), len(r_ra)
    big = 1e6
    cost = np.full((nt, nr), big)
    gate_dv = np.maximum(dv_floor, 0.5 * np.nan_to_num(t_w20))
    for i in range(nt):
        s = angsep(t_ra[i], t_dec[i], r_ra, r_dec)
        dv = np.abs(np.asarray(r_v, float) - t_v[i])
        ok = (s <= rmax) & (dv <= gate_dv[i])
        if ok.any():
            cost[i, ok] = (s[ok] / rmax) ** 2 + (dv[ok] / gate_dv[i]) ** 2
    out = np.full(nt, -1, int)
    if nt and nr:
        try:
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < big:
                    out[i] = j
        except ImportError:                                    # greedy fallback
            used = np.zeros(nr, bool)
            for i in np.argsort(cost.min(axis=1)):
                j = int(np.argmin(np.where(used, big, cost[i])))
                if cost[i, j] < big:
                    used[j] = True; out[i] = j
    return out, gate_dv


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--injected", help="Cube with the sources in it.")
    ap.add_argument("--substrate", help="Cube without them.")
    ap.add_argument("--mask", help="SoFiA mask cube; with --injected and "
                                   "--substrate this enables the capture measurement.")
    ap.add_argument("--geometry-from", help="Any cube to take beam/WCS from, if the "
                                            "injected cube is not available.")
    ap.add_argument("--bmaj-arcsec", type=float)
    ap.add_argument("--bmin-arcsec", type=float)
    ap.add_argument("--pix-arcsec", type=float)
    ap.add_argument("--dv-kms", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam-factor", type=float, default=2.0,
                    help="Spatial gate in units of BMAJ (thesis uses 2).")
    ap.add_argument("--dv-channels", type=float, default=2.0,
                    help="Velocity gate floor in channels; the gate is "
                         "max(this, 0.5*W20).")
    ap.add_argument("--min-rel", type=float, default=0.0,
                    help="Extra reliability cut. SoFiA already applies the par's "
                         "reliability.threshold, so leave at 0 unless tightening.")
    a = ap.parse_args()

    truth = Table.read(a.truth, format="ascii.ecsv")
    cat = Table.read(a.catalogue, format="votable")

    # Geometry: from a cube when there is one, otherwise from explicit values.
    # Older runs kept only the truth table and the catalogue, and those still
    # need to be re-analysable -- they are the regression tests.
    cube_for_geom = a.geometry_from or a.injected or a.substrate or a.mask
    if cube_for_geom:
        with fits.open(cube_for_geom, memmap=True) as h:
            hdr = h[0].header
        dv = abs(velocity_axis(hdr)[1])
        bmaj = float(hdr["BMAJ"]); bmin = float(hdr.get("BMIN", bmaj))
        if bmaj > 1.0:
            bmaj, bmin = bmaj / 3600.0, bmin / 3600.0
        pixdeg = 0.5 * (abs(float(hdr["CDELT1"])) + abs(float(hdr["CDELT2"])))
    elif None not in (a.bmaj_arcsec, a.pix_arcsec, a.dv_kms):
        bmaj = a.bmaj_arcsec / 3600.0
        bmin = (a.bmin_arcsec or a.bmaj_arcsec) / 3600.0
        pixdeg = a.pix_arcsec / 3600.0
        dv = abs(a.dv_kms)
    else:
        raise SystemExit("Give a cube (--injected / --geometry-from) or "
                         "--bmaj-arcsec --pix-arcsec --dv-kms.")
    beam_area_pix = (np.pi / (4 * np.log(2))) * bmaj * bmin / pixdeg**2
    bmaj_arcsec = bmaj * 3600.0

    # ---- reliability ------------------------------------------------------
    relc = pick(cat, ["rel", "reliability"])
    n_cat_all = len(cat)
    if relc and a.min_rel > 0:
        cat = cat[np.asarray(cat[relc], float) >= a.min_rel]

    # ---- columns ----------------------------------------------------------
    r_ra = col(cat, pick(cat, ["ra"], True, "catalogue ra"))
    r_dec = col(cat, pick(cat, ["dec"], True, "catalogue dec"))
    vc = pick(cat, ["v_rad", "v_opt", "v_app"], True, "catalogue velocity")
    r_v = col(cat, vc)
    if np.nanmedian(np.abs(r_v)) > 1e5:
        r_v = r_v / 1e3
    fint_cat, fdesc = fsum_to_jykms(cat, pick(cat, ["f_sum"], True, "f_sum"),
                                    dv, beam_area_pix)
    r_id = col(cat, pick(cat, ["id"]), default=np.nan)
    r_rel = col(cat, relc) if relc else np.full(len(cat), np.nan)
    r_npix = col(cat, pick(cat, ["n_pix"]))
    # SoFiA's own measurements of the parameters TEMBA knows the truth for
    r_w50 = col(cat, pick(cat, ["w50"]))
    r_w20 = col(cat, pick(cat, ["w20"]))
    if np.nanmedian(np.abs(r_w50)) > 1e4:          # m/s
        r_w50, r_w20 = r_w50 / 1e3, r_w20 / 1e3
    r_emaj = col(cat, pick(cat, ["ell_maj"]))
    r_e3maj = col(cat, pick(cat, ["ell3s_maj"]))
    box = {k: col(cat, pick(cat, [k])) for k in
           ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}

    t_ra = col(truth, pick(truth, ["ra"], True, "truth ra"))
    t_dec = col(truth, pick(truth, ["dec"], True, "truth dec"))
    t_v = col(truth, pick(truth, ["vsys_kms", "vsys"], True, "truth vsys"))
    t_f = col(truth, pick(truth, ["Fint_realised_Jykms", "Fint_Jykms"], True, "truth flux"))
    t_w20 = col(truth, pick(truth, ["W20_realised_kms", "W20_kms", "w20"]))
    t_w20 = np.where(np.isfinite(t_w20), t_w20, 0.0)
    t_x = col(truth, pick(truth, ["xpix"]))
    t_y = col(truth, pick(truth, ["ypix"]))

    # ---- match ------------------------------------------------------------
    rmax = a.beam_factor * bmaj_arcsec
    idx, gate = assign(t_ra, t_dec, t_v, t_w20, r_ra, r_dec, r_v,
                       rmax, a.dv_channels * dv)
    det = idx >= 0

    # ---- flux decomposition ----------------------------------------------
    n = len(truth)
    f_model_mask = np.full(n, np.nan)
    f_sum_jy = np.full(n, np.nan)
    npix = np.full(n, np.nan)
    o_w50 = np.full(n, np.nan); o_w20 = np.full(n, np.nan)
    o_emaj = np.full(n, np.nan); o_e3maj = np.full(n, np.nan)
    sep_out = np.full(n, np.nan); dvel_out = np.full(n, np.nan)

    have_mask = all(bool(x) and Path(x).exists()
                    for x in (a.mask, a.injected, a.substrate))
    if have_mask:
        mh = fits.open(a.mask, memmap=True); mdata = mh[0].data
        if mdata.ndim == 4:
            mdata = mdata[0]
        ih = fits.open(a.injected, memmap=True); idata = ih[0].data
        sh = fits.open(a.substrate, memmap=True); sdata = sh[0].data

    for i in np.nonzero(det)[0]:
        j = idx[i]
        sep_out[i] = angsep(t_ra[i], t_dec[i], r_ra[j], r_dec[j])
        dvel_out[i] = abs(r_v[j] - t_v[i])
        f_sum_jy[i] = fint_cat[j]
        npix[i] = r_npix[j]
        o_w50[i], o_w20[i] = r_w50[j], r_w20[j]
        # ell_maj / ell3s_maj are in pixels; convert to arcsec
        o_emaj[i] = r_emaj[j] * (pixdeg * 3600.0)
        o_e3maj[i] = r_e3maj[j] * (pixdeg * 3600.0)

        if not have_mask or not np.isfinite(box["x_min"][j]):
            continue
        x0, x1 = int(box["x_min"][j]), int(box["x_max"][j]) + 1
        y0, y1 = int(box["y_min"][j]), int(box["y_max"][j]) + 1
        z0, z1 = int(box["z_min"][j]), int(box["z_max"][j]) + 1
        m = np.asarray(mdata[z0:z1, y0:y1, x0:x1])
        sel = m == (r_id[j] if np.isfinite(r_id[j]) else m.max())
        if not sel.any():
            sel = m > 0
        d = (np.asarray(idata[z0:z1, y0:y1, x0:x1], float)
             - np.asarray(sdata[z0:z1, y0:y1, x0:x1], float))
        f_model_mask[i] = float(np.nansum(d[sel])) / beam_area_pix * dv

    if have_mask:
        mh.close(); ih.close(); sh.close()

    # ---- output -----------------------------------------------------------
    out = Table()
    for c in truth.colnames:
        out[c] = truth[c]
    out["detected"] = det
    out["cat_id"] = np.where(det, np.where(np.isfinite(r_id[np.clip(idx, 0, None)]),
                                           r_id[np.clip(idx, 0, None)], -1), -1)
    out["sep_arcsec"] = sep_out
    out["dv_kms"] = dvel_out
    out["gate_dv_kms"] = gate
    out["rel"] = np.where(det, r_rel[np.clip(idx, 0, None)], np.nan)
    out["n_pix"] = npix
    out["Fint_injected_Jykms"] = t_f
    out["Fsum_catalogue_Jykms"] = f_sum_jy
    out["Fmodel_in_mask_Jykms"] = f_model_mask
    out["flux_capture"] = f_model_mask / t_f
    out["flux_fidelity"] = f_sum_jy / t_f
    out["noise_in_mask_Jykms"] = f_sum_jy - f_model_mask

    # ---- parameter recovery: flux is one column of this, not the whole story
    out["W50_sofia_kms"] = o_w50
    out["W20_sofia_kms"] = o_w20
    out["ell_maj_sofia_arcsec"] = o_emaj
    out["ell3s_maj_sofia_arcsec"] = o_e3maj
    tw50 = col(truth, pick(truth, ["W50_realised_kms", "W50_kms"]))
    tw20 = col(truth, pick(truth, ["W20_realised_kms", "W20_kms"]))
    tdhi = col(truth, pick(truth, ["dHI_arcsec"]))
    td3 = col(truth, pick(truth, ["d3sigma_moment_arcsec"]))
    out["W50_ratio"] = o_w50 / tw50
    out["W20_ratio"] = o_w20 / tw20
    out["ell3s_over_dHI"] = o_e3maj / tdhi
    out["ell3s_over_d3sigma"] = o_e3maj / td3
    out["ell_maj_over_dHI"] = o_emaj / tdhi

    # carry the cube geometry so temba.compare can report it per field
    # instead of NaN; channel width is the one covariate that cannot vary
    # within a field
    meta = dict(truth.meta.get("temba", {})) if truth.meta else {}
    meta.update(dv_chan_kms=float(dv), bmaj_arcsec=float(bmaj_arcsec),
                pix_arcsec=float(pixdeg * 3600.0),
                beam_area_pix=float(beam_area_pix))
    out.meta["temba"] = meta

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write(a.out, format="ascii.ecsv", overwrite=True)

    unmatched = np.setdiff1d(np.arange(len(cat)), idx[det])
    fp = Table()
    if len(unmatched):
        fp["cat_id"] = r_id[unmatched]; fp["ra"] = r_ra[unmatched]
        fp["dec"] = r_dec[unmatched]; fp["v_kms"] = r_v[unmatched]
        fp["Fint_Jykms"] = fint_cat[unmatched]; fp["rel"] = r_rel[unmatched]
        fp.write(str(a.out).replace(".ecsv", "_unmatched.ecsv"),
                 format="ascii.ecsv", overwrite=True)

    # ---- report -----------------------------------------------------------
    w = 76
    def q(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return (np.percentile(x, 16), np.median(x), np.percentile(x, 84)) if x.size \
            else (np.nan,) * 3

    print("=" * w)
    print("TEMBA MATCH")
    print("=" * w)
    print(f"  BMAJ {bmaj_arcsec:.2f}\"  beam {beam_area_pix:.1f} pix  dv {dv:.2f} km/s")
    print(f"  flux conversion: {fdesc}")
    print(f"  gate: sep <= {rmax:.1f}\"  |dv| <= max({a.dv_channels*dv:.1f}, 0.5 W20) km/s")
    print(f"  catalogue {n_cat_all} rows ({len(cat)} after rel >= {a.min_rel})")
    print()
    print(f"  injected {n}   detected {int(det.sum())}   "
          f"unmatched catalogue entries {len(unmatched)}")
    print("  (the recovered fraction depends on the injected distribution and is")
    print("   not the completeness of the field -- see thesis Table 2.2 caption)")
    print()
    print("  FLUX DENSITY RECOVERY (detected sources)")
    print("  " + "-" * (w - 4))
    for lab, v in [("capture   F_model_in_mask / F_inj", out["flux_capture"]),
                   ("fidelity  f_sum / F_inj", out["flux_fidelity"])]:
        lo, md, hi = q(v)
        print(f"  {lab:<36s} {md:8.3f}   [{lo:.3f}, {hi:.3f}]")
    lo, md, hi = q(out["noise_in_mask_Jykms"])
    print(f"  {'noise in mask [Jy km/s]':<36s} {md:8.4g}   [{lo:.4g}, {hi:.4g}]")
    print()
    print("  PARAMETER RECOVERY (SoFiA / truth)")
    print("  " + "-" * (w - 4))
    for lab, c in (("W50", "W50_ratio"), ("W20", "W20_ratio"),
                   ("ell3s_maj / d_HI", "ell3s_over_dHI"),
                   ("ell3s_maj / d_3sigma", "ell3s_over_d3sigma"),
                   ("ell_maj / d_HI", "ell_maj_over_dHI")):
        lo, md, hi = q(out[c])
        if np.isfinite(md):
            print(f"  {lab:<36s} {md:8.3f}   [{lo:.3f}, {hi:.3f}]")
    if not have_mask:
        print("\n  NOTE: capture needs --mask, --injected and --substrate together.")
        print("        Without them only fidelity (f_sum / F_inj) is measured.")
    print()
    print(f"  wrote {a.out}")

    Path(str(a.out) + ".temba.json").write_text(json.dumps(dict(
        n_injected=int(n), n_detected=int(det.sum()),
        n_catalogue=int(n_cat_all), n_unmatched=int(len(unmatched)),
        flux_conversion=fdesc,
        capture=dict(zip(("p16", "median", "p84"), map(float, q(out["flux_capture"])))),
        fidelity=dict(zip(("p16", "median", "p84"), map(float, q(out["flux_fidelity"])))),
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
