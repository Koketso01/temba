"""temba -- the command line.

    temba init [survey.yaml]          write a commented survey parameter file
    temba check survey.yaml           dependencies, inputs and settings
    temba prepare survey.yaml         write the per-field campaign configs
    temba import fields/              build a survey file from existing field.yaml files
    temba run survey.yaml             substrates, null runs, injections, SoFiA-2, matching
    temba status survey.yaml          progress per field
    temba analyse survey.yaml         completeness fits, figures, examiner tables
    temba install-sofia [--prefix]    download and compile SoFiA-2
    temba banner                      the TEMBA banner

Every step is resumable: rerunning a command continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, branding, config, deps


def _load(path):
    try:
        return config.load(path)
    except FileNotFoundError:
        raise SystemExit(f"no such parameter file: {path}  (create one with `temba init`)")


def _require_fields(s, names):
    """Fail loudly when --fields names a field the survey file does not list."""
    known = [f["name"] for f in s["fields"]]
    missing = [n for n in (names or []) if n not in known]
    if missing:
        raise SystemExit(f"not in {s['_path']}: {', '.join(missing)}\n"
                         f"add an entry under 'fields:' for each (see `temba init`); "
                         f"listed now: {', '.join(known)}")


def cmd_init(a):
    branding.banner()
    dst = Path(a.path)
    if dst.exists() and not a.force:
        raise SystemExit(f"{dst} exists; use --force to overwrite")
    shutil.copy(config.TEMPLATE, dst)
    branding.ok(f"wrote {dst}: fill in the fields, then `temba check {dst}`")


def cmd_check(a):
    branding.banner()
    s = _load(a.config)
    branding.heading("Dependencies")
    hard = 0
    for name, ok, detail in deps.report(s["tools"]):
        (branding.ok if ok else branding.fail)(f"{name:24s} {detail}")
        hard += (not ok)
    branding.heading("\nParameter file")
    errors = 0
    issues = config.check(s)
    for level, msg in issues:
        (branding.fail if level == "error" else branding.warn)(msg)
        errors += level == "error"
    if not issues:
        branding.ok(f"{len(s['fields'])} field(s), no problems")
    if not a.quick:
        branding.heading("\nInputs")
        for f in s["fields"]:
            _inspect_field(f)
    print()
    if errors or hard:
        branding.fail(f"{errors} error(s) in the parameter file, {hard} missing dependenc"
                      f"{'y' if hard == 1 else 'ies'}")
        sys.exit(1)
    branding.ok("ready: `temba run " + a.config + "`")


def _inspect_field(f):
    """Header-level checks: cube and mask agree, beam and spectral axis present."""
    try:
        from astropy.io import fits
        hc = fits.getheader(f["cube"])
        shape = tuple(hc.get(f"NAXIS{i}") for i in (1, 2, 3))
        problems = []
        if not hc.get("BMAJ"):
            problems.append("cube has no BMAJ")
        ctype3 = str(hc.get("CTYPE3", ""))
        if not any(t in ctype3 for t in ("VRAD", "VOPT", "FREQ", "VELO")):
            problems.append(f"unexpected spectral axis {ctype3!r}")
        if f.get("mask") and Path(f["mask"]).exists():
            hm = fits.getheader(f["mask"])
            if tuple(hm.get(f"NAXIS{i}") for i in (1, 2, 3)) != shape:
                problems.append("mask shape differs from the cube")
        msg = (f"{f['name']:16s} {shape[0]}x{shape[1]}x{shape[2]}  "
               f"beam {3600 * float(hc.get('BMAJ', 0)):.1f}\"  {ctype3}")
        if problems:
            branding.fail(msg + "  -- " + "; ".join(problems))
        else:
            branding.ok(msg)
    except Exception as e:
        branding.warn(f"{f.get('name', '?')}: could not read headers ({e})")


def _warn_kept():
    for name in getattr(config.write_field_configs, "kept_different", []):
        branding.warn(f"{name}: existing field.yaml kept although its inputs differ from the "
                      f"survey file (use `temba prepare --overwrite` to replace it)")


def cmd_prepare(a):
    s = _load(a.config)
    _require_fields(s, a.fields)
    for name, p in config.write_field_configs(s, a.fields, overwrite=a.overwrite).items():
        branding.ok(f"{name}: {p}")
    _warn_kept()


def _field_command(s, name, cfg_path):
    f = next(f for f in s["fields"] if f["name"] == name)
    cmd = [sys.executable, "-m", "temba.campaign", "field", "--config", str(cfg_path),
           "--total", str(config.total_injections(s, f)),
           "--outroot", str(Path(s["workdir"]) / "runs")]
    if s["run"].get("cleanup"):
        cmd.append("--cleanup")
    if not s["run"].get("null_run", True):
        cmd.append("--skip-null")
    return cmd


def cmd_run(a):
    branding.banner(compact=True)
    s = _load(a.config)
    _require_fields(s, a.fields)
    bad = [m for lvl, m in config.check(s) if lvl == "error"
           and (not a.fields or any(m.startswith(n + ":") for n in a.fields))]
    if bad:
        for m in bad:
            branding.fail(m)
        raise SystemExit("fix the parameter file first (`temba check`)")
    work = Path(s["workdir"])
    (work / "logs").mkdir(parents=True, exist_ok=True)
    cfgs = config.write_field_configs(s, a.fields)
    _warn_kept()
    queue = list(cfgs.items())
    parallel = max(1, int(a.parallel or s["run"].get("parallel", 1)))
    running = []
    branding.heading(f"{len(queue)} field(s), {parallel} at a time; logs in {work / 'logs'}")
    while queue or running:
        while queue and len(running) < parallel:
            name, p = queue.pop(0)
            cmd = _field_command(s, name, p)
            if a.dry_run:
                print("  " + " ".join(cmd))
                continue
            log = open(work / "logs" / f"campaign_{name}.log", "a")
            running.append((name, subprocess.Popen(cmd, cwd=work, stdout=log, stderr=log), log))
            branding.ok(f"started {name}")
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                (branding.ok if proc.returncode == 0 else branding.fail)(
                    f"{name} finished (exit {proc.returncode})")
        if running:
            time.sleep(5)


def cmd_status(a):
    s = _load(a.config)
    runs = Path(s["workdir"]) / "runs"
    print(f"{'field':16s} {'realisations':>12s} {'injected':>9s} {'detected':>9s}  campaign")
    for f in s["fields"]:
        root = runs / f["name"]
        tabs = sorted(root.glob("run_*/products/recovery.ecsv"))
        n_inj = n_det = 0
        for t in tabs:
            try:
                from astropy.table import Table
                tt = Table.read(t, format="ascii.ecsv")
                n_inj += len(tt)
                n_det += int(sum(bool(x) for x in tt["detected"]))
            except Exception:
                pass
        camp = root / "campaign.json"
        state = "done" if camp.exists() else ("running or stopped" if tabs else "not started")
        if camp.exists():
            c = json.loads(camp.read_text())
            state = f"done ({c.get('completed')}/{c.get('n_realisations')} realisations)"
        print(f"{f['name']:16s} {len(tabs):12d} {n_inj:9d} {n_det:9d}  {state}")


def cmd_analyse(a):
    branding.banner(compact=True)
    s = _load(a.config)
    work = Path(s["workdir"])
    config.write_field_configs(s)
    _warn_kept()
    an = s["analysis"]
    runs, fields, out = str(work / "runs"), str(work / "fields"), work / "analysis"
    steps = [
        ("survey summary", [sys.executable, "-m", "temba.survey", "--runs", runs, "--fields", fields,
                            "--outdir", str(out / "survey")]),
        ("figures", [sys.executable, "-m", "temba.paperfigs", "--runs", runs, "--fields", fields,
                     "--outdir", str(out / "figures"), "--journal", an["journal"],
                     "--fonts", an["fonts"], "--bootstrap", str(an["bootstrap"])]
         + (["--min-freq"] + [f"{k}={v}" for k, v in an["min_freq_mhz"].items()]
            if an.get("min_freq_mhz") else [])),
    ]
    for label, cmd in steps:
        branding.heading(label)
        r = subprocess.run(cmd, cwd=work)
        if r.returncode:
            branding.fail(f"{label} failed (exit {r.returncode})")
    branding.heading("examiner tables")
    with open(out / "examiner.txt", "w") as fh:
        subprocess.run([sys.executable, "-m", "temba.examiner", "--runs", runs, "--fields", fields],
                       cwd=work, stdout=fh)
    branding.ok(f"results in {out}")


def cmd_import(a):
    import yaml
    sv = config.import_fields(a.fields_dir, workdir=a.workdir)
    Path(a.out).write_text("# built by `temba import` from " + str(a.fields_dir)
                           + "; settings not listed take the defaults (see `temba init`)\n"
                           + yaml.safe_dump(sv, sort_keys=False))
    branding.ok(f"{len(sv['fields'])} field(s) -> {a.out}")


def cmd_install_sofia(a):
    branding.banner()
    exe = deps.install_sofia(a.prefix, openmp=not a.no_openmp)
    branding.ok(f"SoFiA-2 built: {exe}")
    branding.ok("set `tools: sofia:` in your parameter file to this path, or add it to PATH")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="temba", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"TEMBA {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("init", help="write a commented survey parameter file")
    p.add_argument("path", nargs="?", default="temba_survey.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("check", help="check dependencies, inputs and settings")
    p.add_argument("config"); p.add_argument("--quick", action="store_true",
                                             help="skip reading FITS headers")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("prepare", help="write per-field campaign configs")
    p.add_argument("config"); p.add_argument("--fields", nargs="*")
    p.add_argument("--overwrite", action="store_true", help="replace existing field.yaml files")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("run", help="run the injection campaigns")
    p.add_argument("config"); p.add_argument("--fields", nargs="*")
    p.add_argument("--parallel", type=int); p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("status", help="progress per field")
    p.add_argument("config"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("analyse", help="completeness fits and figures")
    p.add_argument("config"); p.set_defaults(func=cmd_analyse)
    p = sub.add_parser("import", help="build a survey file from existing field.yaml files")
    p.add_argument("fields_dir"); p.add_argument("--out", default="temba_survey.yaml")
    p.add_argument("--workdir", default=".")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("install-sofia", help="download and compile SoFiA-2")
    p.add_argument("--prefix", default=str(Path.home() / ".temba"))
    p.add_argument("--no-openmp", action="store_true")
    p.set_defaults(func=cmd_install_sofia)
    p = sub.add_parser("banner", help="show the TEMBA banner")
    p.set_defaults(func=lambda a: branding.banner())
    a = ap.parse_args(argv)
    if not getattr(a, "func", None):
        branding.banner()
        ap.print_help()
        return
    a.func(a)


if __name__ == "__main__":
    main()
