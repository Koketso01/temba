# Changelog

## 1.0.2 (2026-10-07)

- `temba add-field` and `temba set` edit a survey file from the command line,
  so no text editor is needed.
- `temba check` warns when a catalogue does not carry its parameter file's
  `output.filename`, i.e. when the parameter file may not be the one that made it.
- Repository moved to github.com/koketso01/temba.

## 1.0.1 (2026-10-07)

- `temba run` and `temba prepare` stop with a clear message when `--fields`
  names a field the survey file does not list (it used to finish silently).
- `temba check` compares the data, not just the path, of the cube against the
  one SoFiA-2 searched: a copy is accepted, different data is an error.
- Missing redshifts are reported in one line.

## 1.0.0 (2026-10-07)

First public version, as used for the MGCLS-HI completeness analysis.

- One survey parameter file (`temba init`) for every field's inputs, the
  number of sources per realisation and in total, the design ranges, and
  the analysis settings; `temba import` converts existing field.yaml files.
- `temba check`: dependencies, inputs, and whether each cube is the one the
  catalogue's SoFiA-2 run searched.
- `temba run` (resumable, fields in parallel), `temba status`, `temba analyse`.
- Optional field-dependent seeds (`injection.field_dependent_seeds`).
- Figures for MNRAS, A&A or a thesis page (`analysis.journal`).
- Installer (`install.sh`), conda environment and Dockerfile with SoFiA-2,
  3D-Barolo (pyBBarolo) and their libraries.
