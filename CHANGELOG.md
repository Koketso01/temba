# Changelog

## 1.0.12 (2026-10-08)

- fig03 (thresholds) is ordered by increasing S50, lowest at the top; every
  other figure and table keeps the alphabetical field order.

## 1.0.11 (2026-10-08)

- fig09: the pair of colour bars is centred under the figure, closer to the
  panels above.

## 1.0.10 (2026-10-08)

- fig09 (showcase): two fields on a thesis page and three on a journal page
  by default (`--showcase-fields` chooses them), the best and worst example of each stacked as rows across the full
  width, so every map is about twice as large; colour bars sit under the
  columns they describe.
- fig07: one shared x and y label for the grid (per-panel labels collided
  around the centred last row).
- Thicker colour bars in fig06, fig07 and fig09.

## 1.0.9 (2026-10-08)

- One field order everywhere (figures, tables, survey summary, examiner
  tables, `temba status`, `temba import`): alphabetical, with numbers compared
  as numbers (Abell 85 before Abell 168). Colours and markers stay attached to
  each field.
- Grids of per-field panels (fig07, fig10, fig14) centre a short last row;
  fig07 has its own colour-bar axis.
- fig01 wording: precise technical labels; the resampling condition is given
  as G > 6 or d < d_min (velocity gradient and size floor).

## 1.0.8 (2026-10-08)

- Figures can be made from any directory: relative paths in field configs
  are resolved against the campaign folder (the released catalogues and
  noise maps were silently skipped before).
- fig01: shows that draws 3D-Barolo cannot model are redrawn and failed
  builds are dropped.
- fig10 and fig14: one shared axis label, the S90 / SNR90 value in each
  panel title, the legend in its own strip.
- fig04: taller, with more room between the panels.

## 1.0.7 (2026-10-07)

- Figures: the font-timestamp warnings that matplotlib's PDF backend logs
  for every embedded Computer Modern font are silenced.
- Fields without a catalogue are skipped quietly in the coverage and
  mass-limit figures.

## 1.0.6 (2026-10-07)

- Substrate files are padded to whole 2880-byte FITS blocks. MGCLS cube sizes
  are exact multiples, so earlier results are unaffected; other cube shapes
  gave astropy's "file may have been truncated" warning.
- The demo injects two realisations of 40 galaxies, enough for the
  survey-level fits and figures (`temba analyse`).
- The survey summary, figures and examiner tables stop with a plain message
  when no field has enough injections, instead of a traceback.

## 1.0.5 (2026-10-07)

- `temba check` and `temba run` also look for the BBarolo executable the
  injector uses (tools.bbarolo, pyBBarolo's own, or BBarolo on PATH).
- `temba demo --tools-from survey.yaml` reuses the tools of a working survey.

## 1.0.4 (2026-10-07)

- `temba demo` finds SoFiA-2 (PATH, ~/.temba, ~/SoFiA-2) and the conda
  environment's libraries, and takes `--sofia PATH`.
- `temba run` stops before starting when SoFiA-2 or pyBBarolo is missing.
- `temba status` marks a field whose realisations all failed; `temba analyse`
  says so instead of failing when nothing has finished.

## 1.0.3 (2026-10-07)

- `temba demo`: a tiny synthetic survey that tests an installation end to end.
- `temba selftest` and `temba report` (a text file to send back with feedback);
  TESTING.md and GitHub issue templates for testers.
- No BBarolo executable needs configuring: pip-installed pyBBarolo ships one.
- `temba install-sofia` clones SoFiA-2 from GitLab (its GitHub repository is a pointer).
- `temba status` shows fields in progress.

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
