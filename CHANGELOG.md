# Changelog

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
