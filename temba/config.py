"""temba.config -- the survey parameter file.

One YAML file describes a whole survey: where each field's cube, SoFiA-2
mask, catalogue, noise cube and parameter file are, how many galaxies to
inject and how, and how to analyse the result. ``temba init`` writes a
commented template; this module loads it, fills in defaults, checks it, and
writes the per-field configs (``<workdir>/fields/<name>/field.yaml``) that the
campaign runner reads.

Relative paths are taken relative to the parameter file's own directory.
"""
from __future__ import annotations

import copy
import re
import zlib
from pathlib import Path

import yaml

DEFAULTS = {
    "survey": "my-survey",
    "workdir": "temba_work",
    "tools": {
        "sofia": "sofia",            # SoFiA-2 executable (path, or a name on PATH)
        "bbarolo": None,             # optional BBarolo executable
        "pythonpath": None,          # optional BBarolo source tree providing pyBBarolo
        "ld_library_path": None,     # optional extra library path (e.g. libwcs for SoFiA-2)
    },
    "cosmology": {"H0": 70.0, "Om0": 0.3},
    "injection": {
        "sources_per_realisation": 150,
        "sources_per_mvox": None,    # a density instead of a fixed count (overrides it)
        "min_sources": 50,
        "max_sources": 1200,
        "total_per_field": 600,      # injections per field, split into realisations
        "realisations": None,        # or: a fixed number of realisations per field
        "seed_base": 42,
        "field_dependent_seeds": False,
        "stamps": 6,                 # showcase stamps saved per realisation
        "design": {
            "log_snr": [-0.5, 2.0],
            "w50_kms": [40.0, 600.0],
            "d_over_beam": [0.25, 4.0],
            "inclination_deg": [15.0, 85.0],
            "sigma_hi_msun_pc2": [1.2, 30.0],
        },
        "w50_tol": 0.05,
        "flux_tol": 0.05,
    },
    "substrate": {"dilate_xy": 2, "dilate_z": 2, "seed": 42},
    "matching": {"beam_factor": 2.0, "dv_channels": 2.0},
    "run": {"null_run": True, "cleanup": True, "parallel": 1},
    "analysis": {"bootstrap": 200, "journal": "mnras", "fonts": "auto", "min_freq_mhz": {}},
    "fields": [],
}

FIELD_KEYS = {"name", "cube", "mask", "sofia_par", "catalogue", "noise_cube",
              "chan_min", "chan_max", "z", "mask_is_raw",
              "injection", "matching", "substrate", "tools", "extra"}
FILE_KEYS = ("cube", "mask", "sofia_par", "catalogue", "noise_cube")


def field_key(name):
    """Alphabetical order with numbers compared as numbers (Abell 85 before Abell 168)."""
    parts = re.split(r"(\d+(?:\.\d+)?)", str(name))
    return tuple(float(t) if i % 2 else t.lower() for i, t in enumerate(parts))


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_par(path):
    """Key = value pairs of a SoFiA-2 parameter file (comments stripped)."""
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _abs(path, root):
    if path in (None, ""):
        return None
    p = Path(str(path)).expanduser()
    return str(p if p.is_absolute() else (root / p).resolve())


def load(path):
    """Read a survey parameter file and fill in every default."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    s = _merge(DEFAULTS, raw)
    root = path.parent
    s["_path"] = str(path)
    s["workdir"] = _abs(s["workdir"], root)
    for k in ("bbarolo", "pythonpath", "ld_library_path"):
        s["tools"][k] = _abs(s["tools"].get(k), root)
    sofia = s["tools"].get("sofia")
    if sofia and ("/" in str(sofia)):
        s["tools"]["sofia"] = _abs(sofia, root)
    fields = []
    for f in s.get("fields") or []:
        f = dict(f)
        cube_from_par = f.get("cube") in (None, "", "from_par")
        if cube_from_par:
            f["cube"] = None
        for k in FILE_KEYS:
            f[k] = _abs(f.get(k), root)
        # the cube SoFiA-2 actually searched is the one to inject into
        if cube_from_par and f.get("sofia_par") and Path(f["sofia_par"]).exists():
            f["cube"] = read_par(f["sofia_par"]).get("input.data") or None
        fields.append(f)
    s["fields"] = fields
    return s


def check(s):
    """Problems with a loaded survey file, as (level, message), level 'error' or 'warn'."""
    out = []
    names = [f.get("name") for f in s["fields"]]
    if not s["fields"]:
        out.append(("error", "no fields listed under 'fields:'"))
    if len(set(names)) != len(names):
        out.append(("error", "field names must be unique"))
    inj = s["injection"]
    if inj.get("realisations") is None and not inj.get("total_per_field"):
        out.append(("error", "set injection.total_per_field or injection.realisations"))
    if int(inj.get("sources_per_realisation") or 0) < 1 and not inj.get("sources_per_mvox"):
        out.append(("error", "injection.sources_per_realisation must be at least 1"))
    for key, (lo, hi) in (inj.get("design") or {}).items():
        if not lo < hi:
            out.append(("error", f"injection.design.{key}: lower limit must be below upper limit"))
    for f in s["fields"]:
        n = f.get("name") or "?"
        unknown = set(f) - FIELD_KEYS
        if unknown:
            out.append(("warn", f"{n}: unknown keys {sorted(unknown)} (ignored)"))
        for k in ("name", "cube", "mask", "sofia_par"):
            if not f.get(k):
                out.append(("error", f"{n}: '{k}' is required"))
        for k in FILE_KEYS:
            if f.get(k) and not Path(f[k]).exists():
                out.append(("error" if k in ("cube", "mask", "sofia_par") else "warn",
                            f"{n}: {k} not found: {f[k]}"))
        if f.get("sofia_par") and Path(f["sofia_par"]).exists() and f.get("cube"):
            searched = read_par(f["sofia_par"]).get("input.data")
            if searched and Path(searched).resolve() != Path(f["cube"]).resolve():
                same = same_cube_data(f["cube"], searched)
                if same is False:
                    out.append(("error", f"{n}: cube holds different data from the one SoFiA-2 "
                                         f"searched ({searched}); inject into that one "
                                         f"(cube: from_par)"))
                elif same is None:
                    out.append(("warn", f"{n}: cube is not the file SoFiA-2 searched ({searched}), "
                                        f"and that file could not be read to compare them"))
        if f.get("sofia_par") is None or (f.get("sofia_par") and not Path(f["sofia_par"]).exists()):
            pass                      # reported above as a missing required file
        if f.get("sofia_par") and Path(f["sofia_par"]).exists() and f.get("catalogue"):
            stem = read_par(f["sofia_par"]).get("output.filename")
            if stem and not Path(f["catalogue"]).name.startswith(stem):
                out.append(("warn", f"{n}: the catalogue ({Path(f['catalogue']).name}) does not carry "
                                    f"this parameter file's output.filename ({stem}); check that "
                                    f"this parameter file made this catalogue"))
        if not f.get("catalogue"):
            out.append(("warn", f"{n}: no catalogue; coverage and mass-limit figures will skip it"))
    no_z = [f.get("name", "?") for f in s["fields"] if f.get("z") is None]
    if no_z:
        out.append(("warn", f"no redshift 'z' for {', '.join(no_z)}: mass limits at the cluster "
                            f"use a built-in value where one is known, otherwise are skipped"))
    return out


def same_cube_data(a, b, atol=0.0):
    """True if two cubes hold the same data (shape, beam, and one channel compared),
    False if they differ, None if either cannot be read."""
    try:
        import numpy as np
        from astropy.io import fits
        with fits.open(a, memmap=True) as ha, fits.open(b, memmap=True) as hb:
            da, db = ha[0].data, hb[0].data
            if da.shape != db.shape:
                return False
            for key in ("BMAJ", "BMIN"):
                if abs(float(ha[0].header.get(key, 0)) - float(hb[0].header.get(key, 0))) > 1e-9:
                    return False
            sa, sb = da.squeeze(), db.squeeze()
            k = sa.shape[0] // 2
            x, y = np.asarray(sa[k], float), np.asarray(sb[k], float)
            both = np.isfinite(x) & np.isfinite(y)
            if (np.isfinite(x) != np.isfinite(y)).mean() > 0.01:
                return False
            return bool(np.allclose(x[both], y[both], atol=atol, rtol=1e-6))
    except Exception:
        return None


def field_config(s, f):
    """The campaign config (field.yaml contents) for one field of the survey."""
    inj = _merge(s["injection"], f.get("injection"))
    design = inj.pop("design")
    work = Path(s["workdir"])
    name = f["name"]
    seed_offset = (zlib.crc32(name.encode()) % 50000) if inj.pop("field_dependent_seeds") else 0
    inj.pop("total_per_field", None)
    inj.pop("realisations", None)
    inj.update(
        log_snr_min=design["log_snr"][0], log_snr_max=design["log_snr"][1],
        w50_min=design["w50_kms"][0], w50_max=design["w50_kms"][1],
        dratio_min=design["d_over_beam"][0], dratio_max=design["d_over_beam"][1],
        inc_min=design["inclination_deg"][0], inc_max=design["inclination_deg"][1],
        sigma_hi_min=design["sigma_hi_msun_pc2"][0], sigma_hi_max=design["sigma_hi_msun_pc2"][1],
        seed_offset=seed_offset)
    if f.get("chan_min") is not None:
        inj["chan_min"] = int(f["chan_min"])
    if f.get("chan_max") is not None:
        inj["chan_max"] = int(f["chan_max"])
    if not inj.get("sources_per_mvox"):
        inj.pop("sources_per_mvox", None)
    tools = {k: v for k, v in _merge(s["tools"], f.get("tools")).items() if v}
    cfg = {
        "field": name,
        "cube": f["cube"],
        "mask": f["mask"],
        "mask_is_raw": bool(f.get("mask_is_raw", False)),
        "science_par": f["sofia_par"],
        "science_catalogue": f.get("catalogue"),
        "rms_cube": f.get("noise_cube"),
        "substrate": str(work / "substrates" / f"{name}_substrate.fits"),
        "noise_map": str(work / "substrates" / f"{name}_sigma_chan.fits"),
        "z": f.get("z"),
        "tools": tools,
        "cosmology": dict(s["cosmology"]),
        "substrate_build": _merge(s["substrate"], f.get("substrate")),
        "injection": inj,
        "matching": _merge(s["matching"], f.get("matching")),
    }
    cfg.update(f.get("extra") or {})
    return {k: v for k, v in cfg.items() if v is not None}


def total_injections(s, f):
    """Injections to request for a field, honouring injection.realisations."""
    inj = _merge(s["injection"], f.get("injection"))
    if inj.get("realisations"):
        per = int(inj["sources_per_realisation"])
        if inj.get("sources_per_mvox"):
            try:
                from astropy.io import fits
                h = fits.getheader(f["cube"])
                mvox = h["NAXIS1"] * h["NAXIS2"] * h["NAXIS3"] / 1e6
                per = int(min(max(round(inj["sources_per_mvox"] * mvox), inj["min_sources"]),
                              inj["max_sources"]))
            except Exception:
                pass
        return int(inj["realisations"]) * per
    return int(inj["total_per_field"])


def write_field_configs(s, only=None, overwrite=False):
    """Write <workdir>/fields/<name>/field.yaml for every (or the named) field.

    An existing field.yaml is kept unless overwrite=True, so that hand edits
    (and fields set up before the survey file existed) are never lost. A kept
    file that differs from what the survey file would write is reported.
    """
    out, kept = {}, []
    for f in s["fields"]:
        if only and f["name"] not in only:
            continue
        d = Path(s["workdir"]) / "fields" / f["name"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / "field.yaml"
        new = field_config(s, f)
        if p.exists() and not overwrite:
            old = yaml.safe_load(p.read_text()) or {}
            if any(old.get(k) != new.get(k) for k in ("cube", "mask", "science_par")):
                kept.append(f["name"])
        else:
            p.write_text("# written by temba from " + s["_path"] + "\n"
                         + yaml.safe_dump(new, sort_keys=False))
        out[f["name"]] = p
    write_field_configs.kept_different = kept
    return out


TEMPLATE = Path(__file__).with_name("data") / "survey_template.yaml"


def import_fields(fields_dir, workdir="."):
    """A survey dict built from existing per-field field.yaml files."""
    fields, tools = [], {}
    for p in sorted(Path(fields_dir).glob("*/field.yaml"), key=lambda q: field_key(q.parent.name)):
        c = yaml.safe_load(p.read_text()) or {}
        inj = c.get("injection") or {}
        tools = tools or {k: v for k, v in (c.get("tools") or {}).items() if v}
        fields.append({k: v for k, v in {
            "name": c.get("field", p.parent.name), "cube": c.get("cube"), "mask": c.get("mask"),
            "sofia_par": c.get("science_par"), "catalogue": c.get("science_catalogue"),
            "noise_cube": c.get("rms_cube"), "chan_min": inj.get("chan_min"),
            "chan_max": inj.get("chan_max"), "z": c.get("z"),
            "mask_is_raw": c.get("mask_is_raw") or None}.items() if v is not None})
    return {"survey": "imported", "workdir": str(workdir), "tools": tools, "fields": fields}


def _backup(path):
    path = Path(path)
    b = path.with_suffix(path.suffix + ".bak")
    b.write_text(path.read_text())
    return b


def add_field(path, entry):
    """Append one field to a survey file without disturbing the rest of it.

    When 'fields:' is the file's last top-level key (as `temba init` and
    `temba import` write it) the entry is appended as text, so comments survive;
    otherwise the file is rewritten (a .bak copy is kept).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    names = [f.get("name") for f in raw.get("fields") or []]
    if entry["name"] in names:
        raise SystemExit(f"{entry['name']} is already in {path}")
    keys = list(raw)
    if keys and keys[-1] == "fields":
        text = path.read_text()
        indent = "  " if any(l.startswith("  - name:") for l in text.splitlines()) else ""
        block = yaml.safe_dump([entry], sort_keys=False).splitlines()
        text = text.rstrip("\n") + "\n" + "\n".join(indent + l for l in block) + "\n"
        path.write_text(text)
        return None
    _backup(path)
    raw.setdefault("fields", []).append(entry)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return str(path) + ".bak"


def set_values(path, assignments):
    """Set dotted keys (injection.realisations=4) in a survey file; keeps a .bak copy."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    for a in assignments:
        if "=" not in a:
            raise SystemExit(f"expected key=value, got {a!r}")
        key, value = a.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise SystemExit(f"{key}: {k} is not a section")
        node[parts[-1]] = yaml.safe_load(value)
    b = _backup(path)
    fields = raw.pop("fields", None)          # keep the field list last, where add_field expects it
    if fields is not None:
        raw["fields"] = fields
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return str(b)
