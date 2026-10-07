"""
temba.campaign -- run realisations and whole-field campaigns.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

    temba.campaign realisation   one 150-source injection -> SoFiA -> match
    temba.campaign field         a field: substrate, null run, N realisations,
                                 combined analysis
    temba.campaign all           every field config in a directory

Design notes
------------
Every step is *resumable*. A campaign of 34 realisations across 15 fields is
hours of SoFiA, and it will be interrupted. Each step checks for its own
output and skips if it is already there, so rerunning the same command
continues rather than restarting. Use --force to redo.

The substrate and the noise map are built once per field and shared by every
realisation. That is deliberate: the aim is the selection function of *this*
cube with *these* artefacts, not an average over hypothetical noise draws.
Realisations differ only in the injection seed.

The null run (SoFiA on the substrate alone) happens once per field, before any
injection, and its detection count is the false-positive rate every subsequent
completeness number is quoted against. On Abell-194 it is zero, because the
reliability filter finally has negative detections to work with.

SoFiA exit code 8 means "no reliable sources", which is the correct answer for
a null run and must not be treated as failure.

Total injections are requested per field and split into realisations of
`sources_per_realisation` (150 by default), since 150 keeps the source density
realistic; the statistics come from repetition, not from crowding one cube.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SOFIA_OK = {0: "ok", 8: "no reliable sources"}


class FieldLock:
    """Refuse to run if another live process holds this field.

    Two concurrent campaigns on one field corrupt each other silently: the
    resumability checks see half-written outputs and skip steps, and --cleanup
    deletes files the other process is still reading. The result looks like
    data, not like a crash, which is the dangerous kind of failure.
    """

    def __init__(self, root):
        self.path = Path(root) / ".temba.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().split()[0])
            except (ValueError, IndexError):
                pid = None
            if pid and self._alive(pid):
                raise SystemExit(
                    f"\n{self.path} is held by live process {pid}.\n"
                    "Another TEMBA campaign is already running on this field. "
                    "Wait for it,\nor kill it and remove the lock file.")
            print(f"  [lock] removing stale lock from dead process {pid}")
        self.path.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass
        return False

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def load(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg["_path"] = str(cfg_path)
    return cfg


def env_for(cfg):
    """Environment for the subprocesses, from the config rather than the shell.

    Two dependencies live outside the conda env: SoFiA needs libwcs on
    LD_LIBRARY_PATH, and the injector imports pyBBarolo from the BBarolo source
    tree. Both worked interactively and then failed in a detached campaign
    because the shell that set them had gone. Recording them in field.yaml
    makes a campaign reproducible from a bare login.
    """
    e = os.environ.copy()
    tools = cfg.get("tools") or {}
    for key, var in (("ld_library_path", "LD_LIBRARY_PATH"),
                     ("pythonpath", "PYTHONPATH")):
        v = tools.get(key)
        if v:
            e[var] = str(v) + (":" + e[var] if e.get(var) else "")
    return e


def _reap(signum, frame):
    """Kill the current child before exiting.

    subprocess.run leaves its child running if the parent is killed, and the
    child is then inherited by init. A SoFiA run orphaned this way keeps a core
    busy for hours and, worse, can collide with the replacement process writing
    the same output directory -- which is how one null run was corrupted.
    """
    proc = _CURRENT.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    raise SystemExit(f"terminated by signal {signum}")


_CURRENT = {}


def run(cmd, log=None, env=None, allow=(0,)):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            proc = subprocess.Popen([str(c) for c in cmd], stdout=fh,
                                    stderr=subprocess.STDOUT, text=True, env=env)
            _CURRENT["proc"] = proc
            proc.wait()
            cp = proc
    else:
        proc = subprocess.Popen([str(c) for c in cmd], env=env)
        _CURRENT["proc"] = proc
        proc.wait()
        cp = proc
    _CURRENT["proc"] = None
    dt = time.time() - t0
    if cp.returncode not in allow:
        raise SystemExit(f"  FAILED rc={cp.returncode} after {dt:.0f}s"
                         + (f"; see {log}" if log else ""))
    print(f"    done in {dt:.0f}s", flush=True)
    return cp.returncode


def patch_par(src, dst, repl):
    import re
    out, done = [], set()
    for line in Path(src).read_text().splitlines():
        m = re.match(r"^(\s*)([A-Za-z0-9_.]+)(\s*=\s*)(.*)$", line)
        if m and m.group(2) in repl:
            out.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{repl[m.group(2)]}")
            done.add(m.group(2))
        else:
            out.append(line)
    for k, v in repl.items():
        if k not in done:
            out.append(f"{k} = {v}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(out) + "\n")


def sofia(cfg, cube, outdir, prefix, par_out, log, force=False):
    """Run SoFiA with the field's science parameters, only the paths changed."""
    cat = Path(outdir) / f"{prefix}_cat.xml"
    if cat.exists() and not force:
        print(f"    [skip] {cat}"); return cat
    Path(outdir).mkdir(parents=True, exist_ok=True)
    patch_par(cfg["science_par"], par_out, {
        "input.data": str(Path(cube).resolve()),
        "output.directory": str(Path(outdir).resolve()),
        "output.filename": prefix,
        "output.writeCatXML": "true", "output.overwrite": "true",
        "output.writeMask": "true", "output.writeRawMask": "false",
        "output.writeNoise": "false", "output.writeFiltered": "false",
        "output.writeMoments": "false", "output.writeCubelets": "false",
    })
    rc = run([cfg["tools"]["sofia"], par_out], log=log, env=env_for(cfg),
             allow=tuple(SOFIA_OK))
    if rc == 8:
        print("    SoFiA: no reliable sources. That is the right answer for "
              "the null run;\n    for an injected cube it means nothing was "
              "injected, or nothing survived\n    the reliability cut -- check "
              "inject.temba.json before blaming SoFiA.")
        return None
    return cat if cat.exists() else None


# ---------------------------------------------------------------------------

def ensure_substrate(cfg, force=False):
    sub = Path(cfg["substrate"])
    # The sidecar is written only after the build completes, so its presence
    # distinguishes a finished substrate from one killed part-way through.
    # build() pre-allocates the full file before filling it, so an interrupted
    # build leaves a full-size cube that is mostly zeros and that a bare
    # exists() check would happily accept.
    if sub.exists() and Path(str(sub) + ".temba.json").exists() and not force:
        print(f"  [skip] substrate {sub}")
        return
    sb = cfg.get("substrate_build", {}) or {}
    run([sys.executable, "-m", "temba.substrate", "build",
         "--cube", cfg["cube"], "--mask", cfg["mask"], "--out", str(sub),
         "--dilate-xy", sb.get("dilate_xy", 2), "--dilate-z", sb.get("dilate_z", 2),
         "--seed", sb.get("seed", 42)]
        + (["--rms-cube", cfg["rms_cube"]] if cfg.get("rms_cube") else []))


def ensure_null(cfg, root, force=False, skip=False):
    d = Path(root) / "null"
    if skip:
        # Measured as zero on Abell-194 and Abell-168. On a source-free cube
        # SoFiA's reliability step is at its most expensive -- the positive and
        # negative detection populations are identical, so the kernel estimate
        # fails over and takes hours -- for a number already established.
        d.mkdir(parents=True, exist_ok=True)
        (d / "null.json").write_text('{"n_false_positives": null, '
                                     '"catalogue": null, "skipped": true}\n')
        print("  [skip] null run (--skip-null)")
        return dict(n_false_positives=None, catalogue=None, skipped=True)
    summary = d / "null.json"
    if summary.exists() and not force:
        print(f"  [skip] null run {summary}")
        return json.loads(summary.read_text())
    cat = sofia(cfg, cfg["substrate"], d / "sofia", f"{cfg['field']}_null",
                d / "sofia_run.par", d / "sofia.log", force)
    n = 0
    if cat and Path(cat).exists():
        from astropy.table import Table
        n = len(Table.read(cat, format="votable"))
    d.mkdir(parents=True, exist_ok=True)
    out = dict(n_false_positives=int(n), catalogue=str(cat) if cat else None)
    summary.write_text(json.dumps(out, indent=2) + "\n")
    print(f"    false positives on the substrate: {n}")
    return out


def realisation(cfg, root, run_id, force=False, cleanup=False, n=None):
    inj_cfg = cfg.get("injection", {}) or {}
    if n is None:
        n = sources_for(cfg)[0]
    inj_cfg = cfg.get("injection") or {}
    # seed_offset != 0 gives each field its own design draws (temba.config:
    # field_dependent_seeds); 0 keeps the paired design of earlier releases
    seed = (100000 * run_id + int(inj_cfg.get("seed_base", 42))
            + int(inj_cfg.get("seed_offset", 0)))
    d = Path(root) / f"run_{run_id:04d}"
    pop = d / "injection" / "population.ecsv"
    truth = d / "injection" / "injected_galaxies_truth.ecsv"
    cube = d / "injection" / "substrate_plus_galaxies.fits"
    rec = d / "products" / "recovery.ecsv"
    prefix = f"{cfg['field']}_run_{run_id:04d}"

    print(f"\n--- {cfg['field']} realisation {run_id} (seed {seed}, n={n}) ---")

    if rec.exists() and not force:
        print(f"  [skip] already complete: {rec}")
        return rec

    if not pop.exists() or force:
        opt = []
        for k in ("log_snr_min", "log_snr_max", "w50_min", "w50_max",
                  "dratio_min", "dratio_max", "inc_min", "inc_max",
                  "sigma_hi_min", "sigma_hi_max", "chan_min", "chan_max"):
            if k in inj_cfg:
                opt += [f"--{k.replace('_','-')}", str(inj_cfg[k])]
        run([sys.executable, "-m", "temba.population",
             "--substrate", cfg["substrate"], "--out", str(pop),
             "--noise-map", cfg["noise_map"], "--n", n, "--seed", seed,
             "--field", cfg["field"], "--run-id", run_id,
             "--H0", (cfg.get("cosmology") or {}).get("H0", 70.0),
             "--Om0", (cfg.get("cosmology") or {}).get("Om0", 0.3)] + opt,
            log=d / "logs" / "population.log")

    if not cube.exists() or force:
        run([sys.executable, "-m", "temba.inject",
             "--population", str(pop), "--substrate", cfg["substrate"],
             "--bb-exe", cfg["tools"]["bbarolo"], "--outdir", str(d / "injection"),
             "--model-oversample", inj_cfg.get("model_oversample", 5),
             "--w50-tol", inj_cfg.get("w50_tol", 0.10),
             "--flux-tol", inj_cfg.get("flux_tol", 0.05)],
            log=d / "logs" / "inject.log", env=env_for(cfg))

    cat = sofia(cfg, cube, d / "sofia", prefix, d / "provenance" / "sofia_run.par",
                d / "logs" / "sofia.log", force)
    if cat is None:
        print("  SoFiA found nothing; realisation produced no catalogue.")
        return None

    mt = cfg.get("matching", {}) or {}
    mask = Path(d / "sofia" / f"{prefix}_mask.fits")
    run([sys.executable, "-m", "temba.match",
         "--truth", str(truth), "--catalogue", str(cat),
         "--injected", str(cube), "--substrate", cfg["substrate"],
         "--out", str(rec),
         "--beam-factor", mt.get("beam_factor", 2.0),
         "--dv-channels", mt.get("dv_channels", 2.0),
         "--min-rel", mt.get("min_rel", 0.0)]
        + (["--mask", str(mask)] if mask.exists() else []),
        log=d / "logs" / "match.log")

    # Cutouts for the per-source figure. Must run before cleanup: the model is
    # recovered as (injected - substrate), so both cubes have to still exist.
    if inj_cfg.get("stamps", 6) and mask.exists() and rec.exists():
        run([sys.executable, "-m", "temba.showcase", "stamps",
             "--recovery", str(rec),
             "--injected", str(cube),
             "--substrate", cfg["substrate"],
             "--mask", str(mask),
             "--out", str(d / "products" / "stamps.npz"),
             "--n", str(int(inj_cfg.get("stamps", 6)))],
            log=d / "logs" / "stamps.log")

    if cleanup:
        for p in (cube,):
            if p.exists():
                p.unlink()
        for sd in (d / "injection" / "bb_output", d / "injection" / "models"):
            shutil.rmtree(sd, ignore_errors=True)
        for p in (d / "sofia").glob("*"):
            if p.is_file() and p != Path(cat):
                p.unlink()
        print("  [cleanup] removed cubes and model products")
    return rec


def sources_for(cfg):
    """Sources per realisation, scaled to cube volume when configured.

    A fixed count sampled a 687 Mvox cube 3.7x more sparsely than the 186 Mvox
    cube the density was validated on, while costing 3.7x more per SoFiA run.
    """
    inj = cfg.get("injection") or {}
    n_per = int(inj.get("sources_per_realisation", 150))
    per_mvox = inj.get("sources_per_mvox")
    if not per_mvox:
        return n_per, None
    try:
        from astropy.io import fits
        with fits.open(cfg["substrate"], memmap=True) as h:
            nz, ny, nx = h[0].data.shape
        mvox = nx * ny * nz / 1e6
    except Exception as exc:
        print(f"  [warn] could not size the substrate ({exc}); "
              f"using sources_per_realisation = {n_per}")
        return n_per, None
    scaled = int(np.clip(round(float(per_mvox) * mvox),
                         int(inj.get("min_sources", 50)),
                         int(inj.get("max_sources", 1200))))
    return scaled, mvox


def field(cfg_path, total, outroot="runs", force=False, cleanup=False,
          start=1, analyse=True, skip_null=False):
    cfg = load(cfg_path)
    root = Path(outroot) / cfg["field"]
    # The substrate must exist before the source count can be scaled to the
    # cube volume; building it inside the lock below is too late, and the
    # count silently falls back to the fixed default.
    ensure_substrate(cfg, force)
    n_per, mvox = sources_for(cfg)
    if mvox:
        print(f"  cube {mvox:.0f} Mvox -> {n_per} sources per realisation "
              f"(density matched)")
    n_real = max(1, math.ceil(total / n_per))

    print("=" * 74)
    print(f"TEMBA CAMPAIGN  {cfg['field']}   {total} injections "
          f"= {n_real} x {n_per}")
    print("=" * 74)

    root.mkdir(parents=True, exist_ok=True)
    with FieldLock(root):
        ensure_substrate(cfg, force)
        null = ensure_null(cfg, root, force, skip_null)

        done = []
        for i in range(start, start + n_real):
            try:
                r = realisation(cfg, root, i, force, cleanup, n_per)
                if r:
                    done.append(str(r))
            except SystemExit as e:
                print(f"  realisation {i} failed: {e}")
                # A signal means stop the campaign, not just this realisation.
                # Carrying on started the next realisation with the settings
                # loaded at launch, after the code and configs had changed.
                if str(e).startswith("terminated by signal"):
                    raise

    (root / "campaign.json").write_text(json.dumps(dict(
        field=cfg["field"], requested=total, per_realisation=n_per,
        n_realisations=n_real, completed=len(done),
        false_positives=null.get("n_false_positives"),
        recovery_tables=done), indent=2) + "\n")

    if analyse and done:
        run([sys.executable, "-m", "temba.selection",
             "--recovery", *done, "--outdir", str(root / "analysis"),
             "--label", cfg["field"]])
    print(f"\n  {len(done)}/{n_real} realisations complete for {cfg['field']}")
    return done


def main():
    import signal
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _reap)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)

    common = dict()
    r = sub.add_parser("realisation", help="One realisation.")
    r.add_argument("--config", required=True); r.add_argument("--run-id", type=int, default=1)
    r.add_argument("--outroot", default="runs")

    f = sub.add_parser("field", help="A whole field.")
    f.add_argument("--config", required=True)
    f.add_argument("--total", type=int, default=5000)
    f.add_argument("--start", type=int, default=1)
    f.add_argument("--outroot", default="runs")
    f.add_argument("--no-analyse", action="store_true")

    a_ = sub.add_parser("all", help="Every field config in a directory.")
    a_.add_argument("--fields", default="fields")
    a_.add_argument("--total", type=int, default=5000)
    a_.add_argument("--outroot", default="runs")
    a_.add_argument("--only", nargs="*",
                    help="Restrict to these field names, e.g. "
                         "--only Abell-194 Abell-168.")
    a_.add_argument("--skip", nargs="*", help="Exclude these field names.")

    for p in (f, a_):
        p.add_argument("--skip-null", action="store_true",
                       help="Skip the source-free SoFiA run. Its purpose is to "
                            "measure the false-positive rate, which was zero on "
                            "both fields where it was run; on a source-free cube "
                            "SoFiA's reliability step is at its most expensive.")
    for p in (r, f, a_):
        p.add_argument("--force", action="store_true")
        p.add_argument("--cleanup", action="store_true",
                       help="Delete cubes and model products after matching.")
    a = ap.parse_args()

    if a.action == "realisation":
        cfg = load(a.config)
        root = Path(a.outroot) / cfg["field"]
        root.mkdir(parents=True, exist_ok=True)
        with FieldLock(root):
            ensure_substrate(cfg, a.force)
            ensure_null(cfg, root, a.force)
            realisation(cfg, root, a.run_id, a.force, a.cleanup)
    elif a.action == "field":
        field(a.config, a.total, a.outroot, a.force, a.cleanup, a.start,
              not a.no_analyse, getattr(a, "skip_null", False))
    else:
        cfgs = sorted(Path(a.fields).glob("*/field.yaml"))
        if a.only:
            missing = set(a.only) - {c.parent.name for c in cfgs}
            if missing:
                raise SystemExit(f"No config for: {', '.join(sorted(missing))}")
            cfgs = [c for c in cfgs if c.parent.name in a.only]
        if a.skip:
            cfgs = [c for c in cfgs if c.parent.name not in a.skip]
        if not cfgs:
            raise SystemExit(f"No field.yaml under {a.fields}/")
        print(f"{len(cfgs)} field(s): {', '.join(c.parent.name for c in cfgs)}\n")
        for c in cfgs:
            try:
                field(str(c), a.total, a.outroot, a.force, a.cleanup)
            except SystemExit as e:
                print(f"FIELD FAILED {c.parent.name}: {e}")


if __name__ == "__main__":
    main()
