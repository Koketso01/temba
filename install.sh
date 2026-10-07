#!/usr/bin/env bash
# TEMBA installer: Three-dimensional Emission Mock-models for Blind-survey Assessments
#
#   ./install.sh                 conda environment "temba" with everything
#   ./install.sh --env myenv     a different environment name
#   ./install.sh --no-sofia      skip building SoFiA-2 (you already have it)
#
# Needs conda (Miniforge, Miniconda or Anaconda) and git.
set -euo pipefail

ENV_NAME="temba"
BUILD_SOFIA=1
PREFIX="${HOME}/.temba"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --no-sofia) BUILD_SOFIA=0; shift ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

c() { printf '\033[1;38;2;%sm%s\033[0m\n' "$1" "$2"; }
NAVY="46;64;87"; TEAL="0;121;140"; AMBER="237;174;73"; ROSE="209;73;91"
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  c "$NAVY" " ████████╗ ███████╗ ███╗   ███╗ ██████╗   █████╗ "
  c "$NAVY" " ╚══██╔══╝ ██╔════╝ ████╗ ████║ ██╔══██╗ ██╔══██╗"
  c "$NAVY" "    ██║    █████╗   ██╔████╔██║ ██████╔╝ ███████║"
  c "$TEAL" "    ██║    ██╔══╝   ██║╚██╔╝██║ ██╔══██╗ ██╔══██║"
  c "$TEAL" "    ██║    ███████╗ ██║ ╚═╝ ██║ ██████╔╝ ██║  ██║"
  c "$TEAL" "    ╚═╝    ╚══════╝ ╚═╝     ╚═╝ ╚═════╝  ╚═╝  ╚═╝"
  c "$AMBER" "   ▁▁▁▂▅█▇▆▆▇█▅▂▁▁▁   ▁▁▁▅█▇▆▆▇█▅▂▁▁▁"
  c "$NAVY" "   Three-dimensional Emission Mock-models for Blind-survey Assessments"
  echo "   with 3D-Barolo and SoFiA-2"
  echo
else
  echo "TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments"
fi

step() { c "$TEAL" "==> $1"; }
die() { c "$ROSE" "error: $1"; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found; install Miniforge first (https://github.com/conda-forge/miniforge)"
command -v git >/dev/null 2>&1 || die "git not found"
HERE="$(cd "$(dirname "$0")" && pwd)"

step "1/4  conda environment '${ENV_NAME}' (Python, astropy, wcslib, cfitsio, fftw, compilers)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${HERE}/environment.yml" --prune
else
  conda env create -n "${ENV_NAME}" -f "${HERE}/environment.yml"
fi
RUN="conda run --no-capture-output -n ${ENV_NAME}"

step "2/4  TEMBA and pyBBarolo (3D-Barolo)"
${RUN} python -m pip install --upgrade pip >/dev/null
${RUN} python -m pip install -e "${HERE}[models,dev]"

if [[ ${BUILD_SOFIA} -eq 1 ]]; then
  step "3/4  SoFiA-2 (compiled into ${PREFIX})"
  ${RUN} temba install-sofia --prefix "${PREFIX}"
  SOFIA="${PREFIX}/SoFiA-2/sofia"
else
  step "3/4  SoFiA-2: skipped (--no-sofia)"
  SOFIA="sofia"
fi

step "4/4  tests"
${RUN} python -m pytest -q "${HERE}/tests" || die "tests failed"

echo
c "$TEAL" "TEMBA is installed."
echo "  conda activate ${ENV_NAME}"
echo "  temba init my_survey.yaml        # then set tools.sofia: ${SOFIA}"
echo "  temba check my_survey.yaml"
echo "  temba run my_survey.yaml"
