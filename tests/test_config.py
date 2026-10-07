"""The survey parameter file: defaults, checks, per-field configs and the CLI."""
import subprocess
import sys

import numpy as np
import pytest
import yaml
from astropy.io import fits

from temba import config


def _fake_field(d, name, shape=(20, 16, 16)):
    h = fits.Header()
    h["BMAJ"] = 30 / 3600; h["BMIN"] = 30 / 3600; h["CTYPE3"] = "VRAD"
    h["CDELT1"] = -6 / 3600; h["CDELT2"] = 6 / 3600; h["CDELT3"] = -44100.0
    fits.PrimaryHDU(np.zeros(shape, "f4"), h).writeto(d / f"{name}.fits")
    fits.PrimaryHDU(np.zeros(shape, "i4"), h).writeto(d / f"{name}_mask.fits")
    (d / f"{name}.par").write_text("input.data = x\nscfind.threshold = 3.5\n")


@pytest.fixture
def survey(tmp_path):
    for n in ("A", "B"):
        _fake_field(tmp_path, n)
    s = {"survey": "test", "workdir": "work",
         "injection": {"sources_per_realisation": 120, "realisations": 3,
                       "field_dependent_seeds": True},
         "fields": [{"name": n, "cube": f"{n}.fits", "mask": f"{n}_mask.fits",
                     "sofia_par": f"{n}.par", "z": 0.05} for n in ("A", "B")]}
    s["fields"][1]["injection"] = {"sources_per_realisation": 300}
    p = tmp_path / "survey.yaml"
    p.write_text(yaml.safe_dump(s))
    return p


def test_template_loads_and_has_every_default():
    s = config.load(config.TEMPLATE)
    assert s["injection"]["sources_per_realisation"] == 150
    assert s["fields"][0]["name"] == "Field-A"


def test_defaults_and_paths(survey):
    s = config.load(survey)
    assert s["workdir"] == str(survey.parent / "work")
    assert s["fields"][0]["cube"] == str(survey.parent / "A.fits")
    assert s["matching"]["beam_factor"] == 2.0          # default filled in
    assert not [m for lvl, m in config.check(s) if lvl == "error"]


def test_realisations_and_per_field_override(survey):
    s = config.load(survey)
    a, b = s["fields"]
    assert config.total_injections(s, a) == 3 * 120
    assert config.total_injections(s, b) == 3 * 300


def test_field_config_matches_campaign_schema(survey):
    s = config.load(survey)
    cfg = config.field_config(s, s["fields"][1])
    for key in ("field", "cube", "mask", "science_par", "substrate", "noise_map",
                "tools", "cosmology", "substrate_build", "injection", "matching"):
        assert key in cfg
    inj = cfg["injection"]
    assert inj["sources_per_realisation"] == 300
    assert (inj["log_snr_min"], inj["log_snr_max"]) == (-0.5, 2.0)
    assert inj["seed_offset"] != 0                      # field-dependent seeds on
    assert "design" not in inj and "realisations" not in inj


def test_seed_offsets_differ_between_fields(survey):
    s = config.load(survey)
    a, b = (config.field_config(s, f)["injection"]["seed_offset"] for f in s["fields"])
    assert a != b


def test_missing_input_is_an_error(survey):
    s = config.load(survey)
    s["fields"][0]["cube"] = "/nonexistent.fits"
    assert any(lvl == "error" and "cube not found" in m for lvl, m in config.check(s))


def test_cli_init_check_and_dry_run(survey, tmp_path):
    run = lambda *a: subprocess.run([sys.executable, "-m", "temba", *a], cwd=tmp_path,
                                    capture_output=True, text=True)
    r = run("init", "new.yaml")
    assert r.returncode == 0 and (tmp_path / "new.yaml").exists()
    r = run("prepare", str(survey))
    assert r.returncode == 0
    assert (tmp_path / "work" / "fields" / "A" / "field.yaml").exists()
    r = run("run", str(survey), "--dry-run")
    assert r.returncode == 0 and "temba.campaign field" in r.stdout
    assert "--total 360" in r.stdout and "--total 900" in r.stdout


def test_cube_from_par_and_provenance_warning(survey, tmp_path):
    data = fits.getdata(tmp_path / "B.fits"); data[10] += 1.0
    fits.writeto(tmp_path / "B.fits", data, fits.getheader(tmp_path / "B.fits"), overwrite=True)
    (tmp_path / "A.par").write_text(f"input.data = {tmp_path / 'B.fits'}\n")
    s = config.load(survey)
    assert any("different data from the one SoFiA-2 searched" in m for _, m in config.check(s))
    raw = yaml.safe_load(survey.read_text())
    raw["fields"][0]["cube"] = "from_par"
    survey.write_text(yaml.safe_dump(raw))
    s = config.load(survey)
    assert s["fields"][0]["cube"] == str(tmp_path / "B.fits")


def test_import_round_trip(survey, tmp_path):
    s = config.load(survey)
    config.write_field_configs(s)
    sv = config.import_fields(tmp_path / "work" / "fields")
    assert [f["name"] for f in sv["fields"]] == ["A", "B"]
    assert sv["fields"][0]["sofia_par"] == str(tmp_path / "A.par")


def test_existing_field_yaml_is_kept(survey, tmp_path):
    s = config.load(survey)
    p = config.write_field_configs(s)["A"]
    p.write_text("field: A\ncube: hand-edited.fits\n")
    config.write_field_configs(s)
    assert "hand-edited" in p.read_text()
    assert "A" in config.write_field_configs.kept_different


def test_unknown_field_is_an_error(survey, tmp_path):
    r = subprocess.run([sys.executable, "-m", "temba", "run", str(survey), "--fields", "Nope",
                        "--dry-run"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0 and "not in" in (r.stdout + r.stderr)


def test_same_data_under_another_path_is_accepted(survey, tmp_path):
    import shutil
    shutil.copy(tmp_path / "A.fits", tmp_path / "A_copy.fits")
    (tmp_path / "A.par").write_text(f"input.data = {tmp_path / 'A_copy.fits'}\n")
    s = config.load(survey)
    assert not [m for _, m in config.check(s) if m.startswith("A:") and "searched" in m]
    data = fits.getdata(tmp_path / "A_copy.fits"); data[10] += 1.0
    fits.writeto(tmp_path / "A_copy.fits", data, fits.getheader(tmp_path / "A_copy.fits"), overwrite=True)
    assert any(lvl == "error" and "different data" in m for lvl, m in config.check(config.load(survey)))
