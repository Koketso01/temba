"""temba.paperfigs -- every figure for the paper, in one consistent style.

TEMBA: Three-dimensional Emission Mock-models for Blind-survey Assessments
       (3D-Barolo + SoFiA-2)

    python -m temba.paperfigs --runs runs --fields fields --outdir analysis/paper

Writes, into --outdir, one PDF (vector, fonts embedded) and one PNG preview
per figure, plus figures.tex (LaTeX figure environments with draft captions
whose numbers are filled in from the data):

  fig01_pipeline        schematic of the whole method
  fig02_completeness    completeness against flux and against SNR, every field
  fig03_thresholds      S10 / S50 / S90 per field, in flux and in SNR
  fig04_noise_config    S50, S90 against field noise; SNR50 against minSNR
  fig05_flux_recovery   capture and fidelity against SNR
  fig06_size            SoFiA 3-sigma size against SNR, with point-source curves
  fig07_position_maps   detection probability relative to SNR, across each field
  fig08_radius_freq     the same against radius and against observed frequency
  fig09_showcase        best and worst recovery in six representative fields

Style: STIX (Times-compatible, as MNRAS typesets) text and maths at 8 pt,
nothing below 6.5 pt; MNRAS widths of 84 and 174 mm by default (--journal aa
for A&A's 88 and 180 mm); and one colour, marker and line style per field that never
changes from figure to figure, chosen to stay distinguishable in greyscale.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle

from .compare import collect
from .selection import (fit_logistic, logistic, binned_completeness,
                        equal_count_bins, wilson)
from .survey import fit_fields, par_settings, point_source_size, col

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

JOURNALS = {"aa": dict(col=88.0, full=180.0), "mnras": dict(col=84.0, full=174.0),
            "thesis": dict(col=100.0, full=146.5)}
MM = 1.0 / 25.4
W = dict(JOURNALS["mnras"])

# Fixed for the whole paper. Order is alphabetical, so a field keeps its
# colour whichever subset of fields a figure shows. Paul Tol's "muted" set
# plus three high-contrast additions; each field also has its own marker and
# line style, so nothing relies on colour alone.
FIELD_ORDER = ["Abell-168", "Abell-194", "Abell-3562", "Abell-4038", "Abell-548",
               "J0314.3-4525", "J0351.1-8212", "J0431.4-6126", "J0540.1-4050",
               "J0600.8-5835", "J0757.7-5315", "J2319.2-6750",
               # added later: appended, so the colours of the fields above never change
               "Abell-85", "Abell-3376"]
PALETTE = ["#332288", "#56B4E9", "#44AA99", "#8C510A", "#999933", "#117733",
           "#CC6677", "#882255", "#AA4499", "#0077BB", "#EE7733", "#000000",
           "#EE3377", "#DDAA33"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*", "p", "d", "H"]
LINES = ["-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1))]

INK = "#222222"
GREY = "#8a8a8a"
LIGHT = "#d9d9d9"
TOOL = dict(temba=("#e8eef5", "#3b5b7a"), bbarolo=("#fbe6d4", "#b5651d"),
            sofia=("#dcefe9", "#2a7f6f"), data=("#ffffff", "#555555"))
F0_MHZ = 1420.405751768
C_KMS = 299792.458
BAND_CUTS = {}          # field -> lowest observed frequency used (MHz), from --min-freq


FONT_SCALE = 1.0   # set by use_style: every hard-coded size is quoted at an 8 pt base
BASE_PT = {"mnras": 9.0, "aa": 9.0, "thesis": 10.0}


def pt(x):
    """A font size quoted for an 8 pt base, scaled to the journal's base size."""
    return round(float(x) * FONT_SCALE, 2)


PCT = "%"          # becomes \% when LaTeX typesets the text
USETEX = False
APOS = "\u2019"    # typographic apostrophe, where the font has one

# Times clones, in order of preference. MNRAS sets its text in Times (the
# newtx fonts of its LaTeX template are built on TeX Gyre Termes / URW Times).
TIMES = ["TeX Gyre Termes", "Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman",
         "Times", "STIXGeneral"]


def _quiet_font_warnings():
    import logging
    # set on the emitting logger itself: fontTools adjusts its parent loggers' levels
    for name in ("fontTools", "fontTools.ttLib", "fontTools.ttLib.tables._h_e_a_d",
                 "fontTools.subset"):
        logging.getLogger(name).setLevel(logging.ERROR)


def use_style(base=None, fonts="auto", journal="mnras"):
    """MNRAS typography.

    fonts="auto": the first Times clone installed for text, STIX (a Times
    design) for maths; nothing needs LaTeX. fonts="latex": LaTeX typesets every
    label with newtxtext/newtxmath, the fonts of the MNRAS template (mathptmx
    if newtx is not installed), so figure text matches the paper exactly.
    """
    global PCT, USETEX, FONT_SCALE
    _quiet_font_warnings()
    base = float(base or BASE_PT.get(journal, 9.0))
    FONT_SCALE = base / 8.0
    import shutil
    import subprocess
    import matplotlib.font_manager as fm
    global APOS
    installed = {f.name for f in fm.fontManager.ttflist}
    if journal == "thesis":
        # the thesis is set in Latin Modern: Computer Modern text and maths.
        # cmr10 ships with matplotlib, so this works everywhere.
        cm = ["CMU Serif", "Latin Modern Roman", "cmr10"]
        text_font = next((f for f in cm if f in installed), "cmr10")
        rc = {"font.family": "serif", "font.serif": [text_font, "cmr10", "DejaVu Serif"],
              "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
              "text.usetex": False}
        if text_font == "cmr10":
            APOS = "'"                       # cmr10 has no U+2019; its ' is curly
        used = f"{text_font} (text), Computer Modern (maths)"
    else:
        text_font = next((f for f in TIMES if f in installed), "DejaVu Serif")
        rc = {
            "font.family": "serif",
            "font.serif": [text_font] + [f for f in TIMES if f != text_font] + ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
        }
        used = f"{text_font} (text), STIX (maths)"
    if fonts == "latex":
        if shutil.which("latex") is None:
            print("  --fonts latex: no LaTeX found; using the Times fonts instead")
        else:
            def have(sty):
                r = subprocess.run(["kpsewhich", sty], capture_output=True, text=True)
                return bool(r.stdout.strip())
            if journal == "thesis":
                pre = r"\usepackage{lmodern}"
                serif = ["Computer Modern Roman"]
            else:
                pre = (r"\usepackage{newtxtext,newtxmath}" if have("newtxtext.sty")
                       else r"\usepackage{mathptmx}")
                serif = ["Times"]          # matplotlib's own preamble then selects ptm
            rc.update({"text.usetex": True, "text.latex.preamble": pre,
                       "font.serif": serif})
            PCT, USETEX = r"\%", True
            used = "LaTeX, " + pre.split("{")[1].rstrip("}")
    print(f"  fonts: {used}")
    rc.update({
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 1.0, "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.columnspacing": 1.0,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    plt.rcParams.update(rc)


def bold(s):
    """Bold text in either typesetting mode (LaTeX ignores fontweight)."""
    return r"\textbf{%s}" % s if USETEX else s


from .config import field_key  # noqa: E402  (one display order everywhere)


def ordered(fields):
    """Fields in the display order used by every figure and table."""
    return sorted(fields, key=field_key)


def centred_axes(fig, n, ncol, sharex=True, sharey=True, **gs_kw):
    """n panels in rows of ncol, with a short last row centred under the others.

    Returns the axes and, for each, whether it is first or last in its row and
    whether it shows x tick labels (it does when nothing sits directly below it).
    """
    nrow = int(np.ceil(n / ncol))
    gs = fig.add_gridspec(nrow, 2 * ncol, **gs_kw)
    axes, first, last, span = [], [], [], []
    ref = None
    for i in range(n):
        r, c = divmod(i, ncol)
        in_row = ncol if r < nrow - 1 else n - ncol * (nrow - 1)
        c0 = (ncol - in_row) + 2 * c
        ax = fig.add_subplot(gs[r, c0:c0 + 2], sharex=ref if sharex else None,
                             sharey=ref if sharey else None)
        ref = ref or ax
        axes.append(ax); first.append(c == 0); last.append(c == in_row - 1)
        span.append((r, c0, c0 + 2))
    below = []
    for (r, a, b) in span:
        covered = any(r2 == r + 1 and a2 < b and b2 > a for (r2, a2, b2) in span)
        below.append(not covered)
    for ax, show in zip(axes, below):
        if sharex and not show:
            ax.tick_params(labelbottom=False)
    for ax, f in zip(axes, first):
        if sharey and not f:
            ax.tick_params(labelleft=False)
    return axes, first, last, below, nrow


def fstyle(field):
    i = FIELD_ORDER.index(field) if field in FIELD_ORDER else \
        len(FIELD_ORDER) + abs(hash(field)) % 7
    return dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                ls=LINES[i % len(LINES)])


def label(field):
    """Field name as typeset in the paper."""
    if field.startswith("Abell-"):
        return "Abell " + field.split("-", 1)[1]
    return field.replace("-", "$-$")


def field_legend(fig, fields, ncol=None, y=-0.02, markers=False, both=False):
    # callers ask for 6 columns at journal width; narrower pages get fewer
    ncol = min(ncol or 6, max(3, int(W["full"] / (29.0 * FONT_SCALE))))
    h = []
    for f in ordered(fields):
        s = fstyle(f)
        h.append(Line2D([], [], color=s["color"], ls="" if markers and not both else s["ls"],
                        marker=s["marker"] if (markers or both) else None, ms=3.8,
                        mfc="white" if both else s["color"], mec=s["color"], mew=0.8,
                        lw=1.2, label=label(f)))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=plt.rcParams["legend.fontsize"],
               handlelength=1.2 if (markers and not both) else 3.2)


def panel_tag(ax, s, x=0.03, y=0.95):
    ax.text(x, y, bold(s), transform=ax.transAxes, ha="left", va="top",
            fontsize=plt.rcParams["font.size"], fontweight="bold")


def save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    try:
        fig.savefig(outdir / f"{name}.png", dpi=200)
    except (RuntimeError, FileNotFoundError) as e:     # LaTeX mode without dvipng
        print(f"  ({name}: PDF written; no PNG preview: {str(e).splitlines()[0][:70]})")
    plt.close(fig)
    return str(outdir / f"{name}.pdf")


# ---------------------------------------------------------------------------
# fig 1: pipeline schematic
# ---------------------------------------------------------------------------

def fig_pipeline_boxes(outdir):
    """The method in ten boxes: a title and one short line each."""
    wfull = W["full"]
    s = wfull / 180.0                     # layout drawn for 180 mm, then scaled
    H = 54.0
    fig = plt.figure(figsize=(wfull * MM, H * s * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, wfull); ax.set_ylim(0, H * s); ax.axis("off")
    bw, bh = 30.0, 16.0

    def box(x, y, kind, title, sub, dashed=False):
        fc, ec = TOOL[kind]
        ax.add_patch(FancyBboxPatch((x * s, y * s), bw * s, bh * s,
                                    boxstyle="round,pad=0,rounding_size=%.2f" % (1.8 * s),
                                    fc=fc, ec=ec, lw=0.9, ls="--" if dashed else "-"))
        ax.text((x + bw / 2) * s, (y + bh - 3.6) * s, bold(title), ha="center", va="center",
                fontsize=pt(8.0), fontweight="bold", color=ec)
        ax.text((x + bw / 2) * s, (y + bh / 2 - 2.4) * s, sub, ha="center", va="center",
                fontsize=pt(7.0), color=INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1, ls="-", rad=0.0, color=INK):
        ax.annotate("", xy=(x1 * s, y1 * s), xytext=(x0 * s, y0 * s),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2",
                                    lw=0.9, color=color, ls=ls,
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=0, shrinkB=0))

    xs = [3.0, 39.0, 75.0, 111.0, 147.0]
    y1, y2 = 27.0, 5.5
    row1 = [("data", "Science data", "cube, mask,\npar file"),
            ("temba", "Substrate", "sources removed,\nnoise kept"),
            ("temba", "Population", "SNR, $W_{50}$, $d/\\theta$, $i$\nLatin hypercube"),
            ("temba", "Disc model", "exponential disc,\narctan $v(r)$"),
            ("bbarolo", "Model cube", "tilted rings,\nbeam-convolved")]
    row2 = [("temba", "Survey", "all fields\ncompared"),
            ("temba", "Selection", "$C(S)$, $C(\\mathrm{SNR})$\n$S_{50}$, $S_{90}$"),
            ("temba", "Match", "capture,\nfidelity"),
            ("sofia", "Source finding", "catalogue\nsettings"),
            ("temba", "Inject", "scale, add,\nverify")]
    for x, (k, t, sub) in zip(xs, row1):
        box(x, y1, k, t, sub, dashed=(k == "data"))
    for x, (k, t, sub) in zip(xs, row2):
        box(x, y2, k, t, sub)

    for xa, xb in zip(xs[:-1], xs[1:]):
        arrow(xa + bw, y1 + bh / 2, xb, y1 + bh / 2)          # left to right
        arrow(xb, y2 + bh / 2, xa + bw, y2 + bh / 2)          # right to left
    arrow(xs[4] + bw / 2, y1, xs[4] + bw / 2, y2 + bh)        # model down to injection
    # W50 loop between the disc model and 3D-Barolo
    arrow(xs[4] + 10, y1 + bh, xs[3] + bw - 10, y1 + bh, rad=0.35,
          color=TOOL["bbarolo"][1])
    ax.text(0.5 * (xs[3] + bw + xs[4]) * s, (y1 + bh + 6.0) * s,
            "iterate until $W_{50}$ matches", ha="center", va="bottom",
            fontsize=pt(6.5), color=TOOL["bbarolo"][1])
    # the catalogue's own settings go straight to the source finder
    yc = 0.5 * (y1 + y2 + bh)
    dash = (0, (2.2, 1.6))
    xd, xsof = xs[0] + bw / 2, xs[3] + bw / 2
    ax.plot([xd * s, xd * s, xsof * s], [y1 * s, yc * s, yc * s], color=GREY, lw=0.8, ls=dash)
    arrow(xsof, yc, xsof, y2 + bh, ls=dash, color=GREY)

    # legend for box colours
    lx = 3.0
    for kind, name in (("temba", "TEMBA"), ("bbarolo", "3D-Barolo"), ("sofia", "SoFiA-2"),
                       ("data", "input")):
        fc, ec = TOOL[kind]
        ax.add_patch(FancyBboxPatch((lx * s, 0.8 * s), 4.0 * s, 2.6 * s,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=fc, ec=ec, lw=0.8, ls="--" if kind == "data" else "-"))
        ax.text((lx + 5.5) * s, 2.1 * s, name, va="center", fontsize=pt(6.5))
        lx += 26.0
    return save(fig, outdir, "fig01_pipeline")


def fig_pipeline_column(outdir):
    """The method top to bottom, one column wide: nine plain boxes in three bands.

    Each box carries a coloured edge for the program doing the step, a bold name
    and one short line. The amber arc is the W50 loop; the dashed line on the
    right carries the catalogue's own SoFiA-2 settings straight to the source
    finder.
    """
    from matplotlib.patches import Rectangle, FancyArrowPatch
    wd = W["col"]
    NAVY, RED, TEAL, AMBER = "#2E4057", "#D1495B", "#00798C", "#EDAE49"
    steps = [("data", "Input data", "cube, SoFiA-2 mask and parameters"),
             ("temba", "Substrate", "sources masked and refilled with noise"),
             ("temba", "Population", r"Latin hypercube in SNR, $W_{50}$, $d_{\rm HI}/\theta$, $i$"),
             ("temba", "Disc model", "exponential disc, arctan rotation curve"),
             ("bbarolo", "Model cube", "tilted-ring model, beam-convolved"),
             ("temba", "Injection", "flux scaling, addition, verification"),
             ("sofia", "Source finding", "catalogue SoFiA-2 parameters"),
             ("temba", "Cross-matching", "capture and flux fidelity"),
             ("temba", "Completeness", r"$C(S)$ and $C({\rm SNR})$ for each field")]
    ring = dict(data=GREY, temba=NAVY, bbarolo=RED, sofia=TEAL)
    bands = [(0, 1, "#EEF1F5", "source-free cube"), (2, 5, "#FBF0EC", "model galaxies"),
             (6, 8, "#E6F2F3", "recovery")]
    bh, gap = 9.0, 3.6
    pitch = bh + gap
    bx, bw = 9.8, 50.6          # room on the right for the side branches
    legend_h, top = 9.0, 3.0
    H = top + 9 * pitch - gap + 4.0 + legend_h
    fig = plt.figure(figsize=(wd * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, wd); ax.set_ylim(0, H); ax.axis("off")
    k = wd / 84.0
    bx, bw = bx * k, bw * k
    ys = [H - top - 2.0 - bh - i * pitch for i in range(9)]       # box bottoms
    cx = bx + bw / 2

    for n, (i0, i1, tint, name) in enumerate(bands):
        y0, y1 = ys[i1] - 1.4, ys[i0] + bh + 1.4
        ax.add_patch(FancyBboxPatch((bx - 2.2, y0), bw + 4.4, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=2.2",
                                    fc=tint, ec="none", zorder=0))
        ym = 0.5 * (y0 + y1)
        ax.add_patch(Circle((4.6 * k, y1 - 3.0), 1.7, fc=INK, ec="none", zorder=2))
        ax.text(4.6 * k, y1 - 3.05, bold(str(n + 1)), ha="center", va="center", fontsize=pt(6.0),
                color="white", fontweight="bold", zorder=3)
        # a two-box band is too short for its name on one line beside the badge
        shown = name.replace(" ", "\n", 1) if (i1 - i0) < 2 else name
        ax.text(4.6 * k, 0.5 * (y0 + y1 - 5.0), shown, ha="center", va="center", rotation=90,
                fontsize=min(pt(7.0), 8.0), style="italic", color=INK, linespacing=1.1,
                multialignment="center")

    boxes = []
    for (kind, name, sub), y in zip(steps, ys):
        b = FancyBboxPatch((bx, y), bw, bh, boxstyle="round,pad=0,rounding_size=1.4",
                           fc="white", ec=ring[kind] if kind != "data" else GREY,
                           lw=0.7, ls="--" if kind == "data" else "-", zorder=3)
        ax.add_patch(b)
        if kind != "data":
            bar = Rectangle((bx, y), 1.7, bh, fc=ring[kind], ec="none", zorder=4)
            bar.set_clip_path(b)
            ax.add_patch(bar)
        ax.text(bx + 4.2, y + bh - 3.0, bold(name), ha="left", va="center", fontsize=pt(7.5),
                fontweight="bold", color=INK, zorder=5)
        ax.text(bx + 4.2, y + 2.7, sub, ha="left", va="center", fontsize=min(pt(6.5), 7.6),
                color="#555555", zorder=5)
        boxes.append(b)

    for i in range(8):
        ax.annotate("", xy=(cx, ys[i + 1] + bh), xytext=(cx, ys[i]),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15",
                                    lw=0.8, color=INK, shrinkA=0, shrinkB=0), zorder=6)

    # W50 loop: from 3D-Barolo back up to the disc model, on the right
    xr = bx + bw
    ax.add_patch(FancyArrowPatch((xr, ys[4] + 0.5 * bh), (xr, ys[3] + 0.5 * bh),
                                 connectionstyle="arc3,rad=0.55", arrowstyle="-|>",
                                 mutation_scale=6, lw=0.9, color=AMBER, zorder=6))
    ax.text(xr + 5.6, ys[3] + 0.5 * bh - 0.5 * pitch, r"iterated to $W_{50}$", rotation=90,
            ha="center", va="center", fontsize=pt(6.0), color="#B07A18")
    # galaxies 3D-Barolo cannot model never reach it: the population redraws any
    # draw whose projected velocity gradient exceeds 6 channels per pixel or whose
    # disc is smaller than 2 kpc or 5 arcsec; builds that still fail are dropped
    NOTE = "#6b7480"
    yp = ys[2]
    ax.add_patch(FancyArrowPatch((xr, yp + 0.22 * bh), (xr, yp + 0.78 * bh),
                                 connectionstyle="arc3,rad=1.1", arrowstyle="-|>",
                                 mutation_scale=5, lw=0.8, color=NOTE, zorder=6))
    ax.text(xr + 6.4, yp + 0.5 * bh, "resampled\n" + r"if $G>6$" + "\n" + r"or $d<d_{\rm min}$",
            rotation=90, ha="center",
            va="center", fontsize=pt(5.6), color=NOTE, style="italic", linespacing=1.05,
            multialignment="center")
    yi = ys[5] + 0.5 * bh
    ax.annotate("", xy=(xr + 2.6, yi), xytext=(xr, yi),
                arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", lw=0.8,
                                color=NOTE, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xr + 3.0, yi, r"$\times$", ha="left", va="center", fontsize=pt(7.0), color=RED)
    ax.text(xr + 8.9, yi, "failed models\nexcluded", rotation=90, ha="center", va="center",
            fontsize=pt(5.6), color=NOTE, style="italic", linespacing=1.05,
            multialignment="center")
    # the catalogue's own settings go straight to the source finder
    xg = wd - 4.6 * k
    dash = (0, (2.2, 1.6))
    y_from, y_to = ys[0] + 0.5 * bh, ys[6] + 0.5 * bh
    ax.plot([xr, xg, xg], [y_from, y_from, y_to], color=GREY, lw=0.8, ls=dash, zorder=2)
    ax.annotate("", xy=(xr, y_to), xytext=(xg, y_to),
                arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", lw=0.8,
                                color=GREY, ls=dash, shrinkA=0, shrinkB=0), zorder=2)
    ax.text(xg + 1.9, 0.5 * (y_from + y_to), "identical SoFiA-2 parameters",
            rotation=90, ha="center", va="center", fontsize=pt(6.0), color="#6b7480",
            style="italic")

    # legend: edge colours
    lx, ly = 3.0 * k, 2.6
    for name, col_, dashed in (("TEMBA", NAVY, False), ("3D-Barolo", RED, False),
                               ("SoFiA-2", TEAL, False), ("input data", GREY, True)):
        b = FancyBboxPatch((lx, ly), 5.0, 3.2, boxstyle="round,pad=0,rounding_size=0.7",
                           fc="white", ec=col_, lw=0.7, ls="--" if dashed else "-")
        ax.add_patch(b)
        if not dashed:
            bar = Rectangle((lx, ly), 1.1, 3.2, fc=col_, ec="none")
            bar.set_clip_path(b)
            ax.add_patch(bar)
        ax.text(lx + 6.3, ly + 1.6, name, va="center", fontsize=pt(6.3), color=INK)
        lx += 20.5 * k
    return save(fig, outdir, "fig01_pipeline")


def fig_pipeline_flow(outdir):
    """The method as one flow of nine nodes, each with a small drawing of what it does.

    Three tinted bands group the steps into building a signal-free sky, making
    mock galaxies and recovering them. Node rings are coloured by the program
    that does the work; the dashed arc carries the catalogue's own SoFiA-2 settings
    going straight to the source finder, and the short arc is the W50 loop.
    """
    from matplotlib.patches import Polygon, Ellipse, Rectangle, FancyArrowPatch
    wd = W["full"]
    H = 54.0
    fig = plt.figure(figsize=(wd * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, wd); ax.set_ylim(0, H); ax.axis("off")
    k = wd / 174.0                                    # layout drawn for 174 mm
    NAVY, RED, TEAL, AMBER, SAGE = "#2E4057", "#D1495B", "#00798C", "#EDAE49", "#66A182"
    rng = np.random.default_rng(3)
    r = 6.6
    cy = 31.0
    xs = [10.4 + 19.2 * i for i in range(9)]
    xs = [x * k for x in xs]

    # phase bands
    bands = [(0, 1, "#EEF1F5", "signal-free sky"),
             (2, 5, "#FBF0EC", "mock galaxies"),
             (6, 8, "#E6F2F3", "recovery")]
    for n, (i0, i1, tint, name) in enumerate(bands):
        x0, x1 = xs[i0] - r - 3.2, xs[i1] + r + 3.2
        ax.add_patch(FancyBboxPatch((x0, 11.0), x1 - x0, 33.0,
                                    boxstyle="round,pad=0,rounding_size=3.0",
                                    fc=tint, ec="none", zorder=0))
        xm = 0.5 * (x0 + x1)
        ax.add_patch(Circle((xm - 0.5 * len(name) * 1.18 - 2.6, 6.6), 1.9, fc=INK, ec="none"))
        ax.text(xm - 0.5 * len(name) * 1.18 - 2.6, 6.55, bold(str(n + 1)), ha="center", va="center",
                fontsize=pt(6.5), color="white", fontweight="bold")
        ax.text(xm - 0.5 * len(name) * 1.18, 6.6, name, ha="left", va="center",
                fontsize=pt(7.5), color=INK, style="italic")

    def node(i, ring, dashed=False):
        c = Circle((xs[i], cy), r, fc="white", ec=ring, lw=1.4, ls="--" if dashed else "-",
                   zorder=3)
        ax.add_patch(c)
        return c

    def clip_circle(i):
        return Circle((xs[i], cy), r - 0.9, transform=ax.transData)

    def labels(i, name, sub):
        ax.text(xs[i], cy - r - 3.2, bold(name), ha="center", va="center", fontsize=pt(7.5),
                fontweight="bold", color=INK, zorder=4)
        ax.text(xs[i], cy - r - 7.0, sub, ha="center", va="center", fontsize=pt(6.5),
                color="#555555", zorder=4)

    def texture(i, alpha=1.0, blob=None, cmap="Greys", vmax=1.0):
        n = 9
        img = rng.normal(0.5, 0.18, (n, n))
        if blob is not None:
            yy, xx = np.mgrid[:40, :40] / 39.0 - 0.5
            img = np.kron(img, np.ones((5, 5)))[:40, :40] * 0.55
            img = img + blob * np.exp(-0.5 * (xx ** 2 + yy ** 2) / 0.12 ** 2)
        im = ax.imshow(img, extent=[xs[i] - r, xs[i] + r, cy - r, cy + r], cmap=cmap,
                       vmin=0, vmax=vmax, interpolation="nearest", alpha=alpha, zorder=3.5)
        im.set_clip_path(clip_circle(i))
        return im

    # 1 science data: a cube
    node(0, GREY, dashed=True)
    a = 4.4
    cx = xs[0]
    T, UR, C, UL = (cx, cy + a), (cx + .866 * a, cy + .5 * a), (cx, cy), (cx - .866 * a, cy + .5 * a)
    LR, B, LL = (cx + .866 * a, cy - .5 * a), (cx, cy - a), (cx - .866 * a, cy - .5 * a)
    for pts, fc in (([T, UR, C, UL], "#C9D3E0"), ([UL, C, B, LL], "#8C9BB0"),
                    ([C, UR, LR, B], "#5B6E89")):
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="white", lw=0.5, zorder=4))
    labels(0, "Science data", "cube, mask, par")

    # 2 substrate: noise, sources removed
    node(1, NAVY)
    texture(1)
    labels(1, "Substrate", "noise kept")

    # 3 population: Latin hypercube
    node(2, NAVY)
    cx, h = xs[2], 4.2
    for t in np.linspace(-h, h, 6):
        ax.plot([cx - h, cx + h], [cy + t, cy + t], color="#D5DBE4", lw=0.4, zorder=4)
        ax.plot([cx + t, cx + t], [cy - h, cy + h], color="#D5DBE4", lw=0.4, zorder=4)
    perm = [3, 0, 4, 1, 2]
    cells = np.linspace(-h, h, 6)
    for j, p in enumerate(perm):
        ax.plot(cx + 0.5 * (cells[j] + cells[j + 1]), cy + 0.5 * (cells[p] + cells[p + 1]),
                "o", ms=2.6, color=NAVY, zorder=5)
    labels(2, "Population", "Latin hypercube")

    # 4 disc model: an inclined exponential disc with a rotation arrow
    node(3, NAVY)
    cx = xs[3]
    for j, s_ in enumerate(np.linspace(1.0, 0.25, 6)):
        ax.add_patch(Ellipse((cx, cy - 0.6), 9.6 * s_, 3.8 * s_, angle=12, fc=NAVY,
                             ec="none", alpha=0.16 + 0.05 * j, zorder=4))
    ax.add_patch(FancyArrowPatch((cx + 3.9, cy + 1.9), (cx - 3.9, cy + 2.4),
                                 connectionstyle="arc3,rad=0.45", arrowstyle="-|>",
                                 mutation_scale=5, lw=0.7, color=NAVY, zorder=5))
    labels(3, "Disc model", "exponential disc")

    # 5 3D-Barolo: tilted rings populated with clouds
    node(4, RED)
    cx = xs[4]
    for j, (s_, pa) in enumerate(((1.0, 18), (0.76, 12), (0.52, 6), (0.28, 0))):
        ax.add_patch(Ellipse((cx, cy), 10.0 * s_, 4.4 * s_, angle=pa, fc="none", ec=RED,
                             lw=0.6, zorder=4))
        th = rng.uniform(0, 2 * np.pi, 7 - j)
        e, f_ = 5.0 * s_, 2.2 * s_
        c_, s2 = np.cos(np.radians(pa)), np.sin(np.radians(pa))
        px, py = e * np.cos(th), f_ * np.sin(th)
        ax.plot(cx + px * c_ - py * s2, cy + px * s2 + py * c_, "o", ms=1.3, color=RED,
                zorder=5)
    labels(4, "Model cube", "tilted rings")

    # 6 inject: a galaxy added to the noise
    node(5, NAVY)
    texture(5, blob=1.4, cmap="cividis", vmax=1.5)
    labels(5, "Inject", "scale, add, verify")

    # 7 SoFiA-2: a source with its mask outline
    node(6, TEAL)
    cx = xs[6]
    yy, xx = np.mgrid[-1:1:60j, -1:1:60j]
    blob = np.exp(-0.5 * ((xx / 0.38) ** 2 + (yy / 0.26) ** 2)) + 0.12 * rng.normal(size=xx.shape)
    from scipy.ndimage import gaussian_filter
    blob = gaussian_filter(blob, 2.0)
    im = ax.imshow(blob, extent=[cx - r, cx + r, cy - r, cy + r], cmap="Greys", vmin=-0.1,
                   vmax=1.4, zorder=3.5, interpolation="bilinear")
    im.set_clip_path(clip_circle(6))
    cs = ax.contour(np.linspace(cx - r, cx + r, 60), np.linspace(cy - r, cy + r, 60), blob,
                    levels=[0.28], colors=[TEAL], linewidths=1.0, zorder=5)
    for coll in getattr(cs, "collections", [cs]):
        coll.set_clip_path(clip_circle(6))
    labels(6, "Source finding", "catalogue settings")

    # 8 match: truth and catalogue paired, one false detection left over
    node(7, NAVY)
    cx = xs[7]
    pts = [(-3.0, 2.0), (1.6, 3.2), (-1.2, -2.4), (3.0, -1.4)]
    for j, (u, v) in enumerate(pts[:3]):
        ax.plot(cx + u, cy + v, "o", ms=3.4, mfc="white", mec=NAVY, mew=0.8, zorder=5)
        ax.plot([cx + u, cx + u + 1.1], [cy + v, cy + v - 0.9], color=NAVY, lw=0.6, zorder=4)
        ax.plot(cx + u + 1.1, cy + v - 0.9, "o", ms=2.2, color=TEAL, zorder=5)
    ax.plot(cx + pts[3][0], cy + pts[3][1], "o", ms=2.2, color=TEAL, zorder=5)
    labels(7, "Match", "capture, fidelity")

    # 9 completeness: curves for several fields
    node(8, NAVY)
    cx = xs[8]
    xx = np.linspace(-3.8, 3.8, 60)
    ax.plot([cx - 3.8, cx - 3.8, cx + 3.8], [cy + 3.6, cy - 3.2, cy - 3.2], color=INK, lw=0.5,
            zorder=4)
    for x0, cc in ((-1.2, PALETTE[0]), (0.0, PALETTE[6]), (1.3, PALETTE[9])):
        ax.plot(cx + xx, cy - 3.2 + 6.6 / (1 + np.exp(-(xx - x0) / 0.55)), color=cc, lw=0.9,
                zorder=5)
    labels(8, "Completeness", r"$C(S)$, $C$(SNR)")

    # flow arrows
    for i in range(8):
        ax.annotate("", xy=(xs[i + 1] - r - 0.5, cy), xytext=(xs[i] + r + 0.5, cy),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16",
                                    lw=0.8, color=INK, shrinkA=0, shrinkB=0), zorder=6)
    # W50 loop between disc model and 3D-Barolo
    ax.add_patch(FancyArrowPatch((xs[4] - 2.2, cy + r + 0.2), (xs[3] + 2.2, cy + r + 0.2),
                                 connectionstyle="arc3,rad=0.55", arrowstyle="-|>",
                                 mutation_scale=6, lw=0.9, color=AMBER, zorder=6))
    ax.text(0.5 * (xs[3] + xs[4]), cy + r + 6.3, r"iterate to $W_{50}$", ha="center",
            va="bottom", fontsize=pt(6.5), color="#B07A18")
    # the catalogue's own settings go straight to the source finder
    ax.add_patch(FancyArrowPatch((xs[0] + 1.2, cy + r + 0.2), (xs[6] - 1.2, cy + r + 0.2),
                                 connectionstyle="arc3,rad=-0.17", arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=GREY, ls=(0, (2.4, 1.8)),
                                 zorder=2))
    ax.text(0.5 * (xs[0] + xs[6]), 48.6, "the same settings as the catalogue",
            ha="center", va="bottom", fontsize=pt(6.5), color="#6b7480", style="italic")

    # legend: ring colours
    lx = wd - 64 * k
    for name, col_, ls in (("TEMBA", NAVY, "-"), ("3D-Barolo", RED, "-"), ("SoFiA-2", TEAL, "-"),
                           ("input", GREY, "--")):
        ax.add_patch(Circle((lx, 50.3), 1.5, fc="white", ec=col_, lw=1.1, ls=ls))
        ax.text(lx + 2.6, 50.3, name, va="center", fontsize=pt(6.5), color=INK)
        lx += 16.0 * k
    return save(fig, outdir, "fig01_pipeline")


# ---------------------------------------------------------------------------
# data-driven figures
# ---------------------------------------------------------------------------

def order(res):
    return ordered(res)


def fig_completeness(res, outdir, bins=9):
    fig, axes = plt.subplots(1, 2, figsize=(W["full"] * MM, 0.42 * W["full"] * MM),
                             sharey=True)
    for f in order(res):
        d, st = res[f], fstyle(f)
        for ax, x, fit in ((axes[0], d["logflux"], d["fit_flux"]),
                           (axes[1], d["logsnr"], d["fit_snr"])):
            xs, N, K, p, lo, hi = binned_completeness(x, d["det"], equal_count_bins(x, bins, 8))
            ax.errorbar(xs, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                        fmt=st["marker"], ms=3.4, mfc="white", mec=st["color"],
                        mew=0.8, ecolor=st["color"], elinewidth=0.6, capsize=0, lw=0, zorder=4)
            xx = np.linspace(x.min() - 0.2, x.max() + 0.2, 400)
            ax.plot(xx, logistic(xx, fit["x0"], fit["k"]), color=st["color"],
                    ls=st["ls"], lw=1.35, zorder=3)
    axes[0].set_xlabel(r"$\log_{10}(S_{\rm int}/{\rm Jy\,km\,s^{-1}})$")
    axes[1].set_xlabel(r"$\log_{10}\,{\rm SNR}_{\rm int}$")
    axes[0].set_ylabel("completeness")
    axes[0].set_xlim(-1.8, 0.8); axes[1].set_xlim(0.0, 1.9)
    from matplotlib.ticker import MultipleLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    for ax, t in zip(axes, ("(a)", "(b)")):
        ax.set_ylim(-0.03, 1.05)
        for lv in (0.5, 0.9):
            ax.axhline(lv, color=LIGHT, lw=0.6, ls=":", zorder=0)
            ax.text(0.012, lv - 0.012, "%d%s" % (100 * lv, PCT), transform=ax.get_yaxis_transform(),
                    ha="left", va="top", fontsize=plt.rcParams["legend.fontsize"] - 1,
                    color=GREY)
        panel_tag(ax, t)
    fig.subplots_adjust(wspace=0.05)
    field_legend(fig, res, ncol=6, y=-0.04, both=True)
    return save(fig, outdir, "fig02_completeness")


def fig_thresholds(res, outdir):
    # the one figure ordered by result rather than by name: increasing S50, top to bottom
    fs = sorted(res, key=lambda f: res[f]["fit_flux"]["logS50"])
    fig, axes = plt.subplots(1, 2, figsize=(W["full"] * MM, 0.34 * W["full"] * MM),
                             sharey=True)
    y = np.arange(len(fs))[::-1]
    for ax, key, xl in ((axes[0], "fit_flux", r"$\log_{10}(S/{\rm Jy\,km\,s^{-1}})$"),
                        (axes[1], "fit_snr", r"$\log_{10}\,{\rm SNR}$")):
        for yi, f in zip(y, fs):
            fit, st = res[f][key], fstyle(f)
            ax.plot([fit["logS10"], fit["logS90"]], [yi, yi], color=st["color"], lw=1.6,
                    solid_capstyle="butt", alpha=0.35)
            for k in ("logS10", "logS90"):
                ax.errorbar(fit[k], yi, xerr=fit.get("err_" + k, np.nan), fmt="|",
                            color=st["color"], ms=5, mew=0.9, elinewidth=0.7, capsize=1.2)
            ax.errorbar(fit["logS50"], yi, xerr=fit["err_logS50"], fmt=st["marker"],
                        ms=4, color=st["color"], mec=st["color"], mfc=st["color"],
                        elinewidth=0.8, capsize=1.5)
        ax.set_xlabel(xl)
        ax.yaxis.set_minor_locator(plt.NullLocator())
    axes[0].set_yticks(y); axes[0].set_yticklabels([label(f) for f in fs])
    panel_tag(axes[0], "(a)", x=0.90, y=0.97); panel_tag(axes[1], "(b)", x=0.90, y=0.97)
    fig.subplots_adjust(wspace=0.06)
    return save(fig, outdir, "fig03_thresholds")


def fig_noise_config(res, pars, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(W["full"] * MM, 0.47 * W["full"] * MM))
    ax = axes[0]
    sig = {f: np.log10(1e3 * res[f]["sigma_med"]) for f in res}
    for key, filled in (("logS50", True), ("logS90", False)):
        ys = []
        for f in res:
            st, fit = fstyle(f), res[f]["fit_flux"]
            ax.errorbar(sig[f], fit[key], yerr=fit["err_" + key], fmt=st["marker"], ms=5,
                        color=st["color"], mec=st["color"],
                        mfc=st["color"] if filled else "white", mew=0.9,
                        elinewidth=0.8, capsize=1.5)
            ys.append(fit[key] - sig[f])
        off = np.median(ys)
        xx = np.array([min(sig.values()) - 0.05, max(sig.values()) + 0.05])
        ax.plot(xx, xx + off, color=GREY, lw=0.7, ls="--", zorder=0)
        xm_ = xx[0] + 0.62 * (xx[1] - xx[0])
        ax.text(xm_, xm_ + off + 0.03, r"$S_{50}\propto\sigma$" if filled
                else r"$S_{90}\propto\sigma$", fontsize=plt.rcParams["legend.fontsize"],
                color=GREY, ha="center", va="bottom")
    ax.set_xlabel(r"$\log_{10}(\sigma_{\rm chan}/{\rm mJy\,beam^{-1}})$, field median")
    ax.set_ylabel(r"$\log_{10}(S/{\rm Jy\,km\,s^{-1}})$")
    ax.text(0.97, 0.04, r"filled: $S_{50}$;  open: $S_{90}$",
            transform=ax.transAxes, fontsize=plt.rcParams["legend.fontsize"], color=GREY,
            ha="right", va="bottom")
    panel_tag(ax, "(a)")

    ax = axes[1]
    have = [f for f in res if "reliability.minSNR" in pars.get(f, {})]
    for f in have:
        p, fit, st = pars[f], res[f]["fit_snr"], fstyle(f)
        rel = p.get("reliability.threshold", 0.75)
        scf = p.get("scfind.threshold", 3.0)
        x = p["reliability.minSNR"] + 0.04 * (FIELD_ORDER.index(f) % 5 - 2 if f in FIELD_ORDER else 0)
        ax.errorbar(x, fit["logS50"], yerr=fit["err_logS50"],
                    fmt="s" if rel >= 0.85 else "o", ms=5.2, color=st["color"],
                    mec=st["color"], mfc="white" if scf >= 3.5 else st["color"],
                    mew=0.9, elinewidth=0.8, capsize=1.5)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=INK, ms=4.5, label="rel. threshold 0.75"),
        Line2D([], [], marker="s", ls="", color=INK, ms=4.5, label="rel. threshold 0.90"),
        Line2D([], [], marker="o", ls="", mfc="white", color=INK, ms=4.5,
               label="scfind threshold 4.0")], loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=3 if W["full"] >= 170 else 2, handletextpad=0.3, columnspacing=0.9,
        borderaxespad=0.2,
        fontsize=plt.rcParams["legend.fontsize"] - 0.5)
    ax.set_xlabel("reliability.minSNR")
    ax.set_ylabel(r"$\log_{10}\,{\rm SNR}_{50}$")
    panel_tag(ax, "(b)")
    fig.subplots_adjust(wspace=0.28 * FONT_SCALE ** 2, top=0.86)
    field_legend(fig, res, ncol=6, y=-0.04, markers=True)
    return save(fig, outdir, "fig04_noise_config")


def running(x, y, nbins=10, min_n=20):
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if len(x) < min_n:
        running.err = np.array([])
        return [np.array([])] * 4
    edges = np.unique(np.percentile(x, np.linspace(0, 100, nbins + 1)))
    xm, md, lo, hi, er = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() < min_n:
            continue
        q = np.percentile(y[m], [16, 50, 84])
        xm.append(np.median(x[m])); lo.append(q[0]); md.append(q[1]); hi.append(q[2])
        er.append(1.2533 * 0.5 * (q[2] - q[0]) / np.sqrt(m.sum()))
    running.err = np.array(er)
    return np.array(xm), np.array(md), np.array(lo), np.array(hi)


def fig_flux_recovery(res, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(W["full"] * MM, 0.36 * W["full"] * MM),
                             sharex=True)
    for ax, c, yl, ref in ((axes[0], "flux_capture", r"capture $F_{\rm mask}/F_{\rm inj}$", 1.0),
                           (axes[1], "flux_fidelity", r"fidelity $f_{\rm sum}/F_{\rm inj}$", 1.0)):
        allx, ally, lows, highs = [], [], [], []
        for f in order(res):
            d, st = res[f], fstyle(f)
            x, yv = d["logsnr"][d["det"]], col(d["t"], c)[d["det"]]
            allx.append(x); ally.append(yv)
            xm, md, lo, hi = running(x, yv, nbins=8, min_n=25)
            ax.plot(xm, md, color=st["color"], ls=st["ls"], lw=0.8)
            if len(md):
                lows.append(md.min()); highs.append(md.max())
        xm, md, lo, hi = running(np.concatenate(allx), np.concatenate(ally), nbins=12, min_n=60)
        ax.fill_between(xm, lo, hi, color=LIGHT, lw=0, zorder=0)
        if len(lo):
            ymin = min(np.min(lo), min(lows, default=1.0))
            ymax = max(np.max(hi), max(highs, default=1.0))
            pad = 0.06 * (ymax - ymin + 0.05)
            ax.set_ylim(min(ymin - pad, ref - 0.05), max(ymax + pad, ref + 0.05))
        ax.plot(xm, md, color=INK, lw=1.8, zorder=5)
        ax.errorbar(xm, md, yerr=running.err, fmt="o", ms=2.6, color=INK, elinewidth=0.8,
                    capsize=1.2, zorder=6)
        ax.axhline(ref, color=GREY, lw=0.6, ls=":")
        ax.set_ylabel(yl); ax.set_xlabel(r"$\log_{10}\,{\rm SNR}_{\rm int}$")
    panel_tag(axes[0], "(a)"); panel_tag(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.22 * FONT_SCALE ** 2)
    field_legend(fig, res, ncol=6, y=-0.04)
    return save(fig, outdir, "fig05_flux_recovery")


def fig_size(res, outdir):
    xs, ys, cs = [], [], []
    for f in res:
        t, det = res[f]["t"], res[f]["det"]
        e3 = col(t, "ell3s_maj_sofia_arcsec")
        beam = col(t, "dHI_arcsec") / col(t, "dHI_over_beam")
        g = det & np.isfinite(e3) & np.isfinite(beam) & (beam > 0)
        xs.append(col(t, "snr_design")[g]); ys.append((e3 / beam)[g])
        cs.append(np.log10(col(t, "dHI_over_beam")[g]))
    x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
    if not len(x):
        return None
    fig, ax = plt.subplots(figsize=(W["col"] * MM, 0.86 * W["col"] * MM))
    sc = ax.scatter(x, y, c=c, s=0.8, cmap="cividis", lw=0, alpha=0.6,
                    rasterized=True, vmin=np.log10(0.2), vmax=np.log10(4.0))
    # SoFiA's ell3s is an ellipse fitted to the 3-sigma region, not the isophotal
    # diameter itself, so it runs below the point-source curve by a constant
    # factor. Measure that factor on unresolved sources and show both curves.
    dr = 10 ** c
    ref = point_source_size(x, 3)
    u = (dr < 0.5) & (x > 5) & (ref > 0)
    kfac = float(np.nanmedian(y[u] / ref[u])) if u.sum() > 20 else 1.0
    xx = np.geomspace(3.01, max(150.0, np.nanmax(x)), 400)
    ax.plot(xx, point_source_size(xx, 3), color="#c1440e", lw=0.8, ls="--",
            label=r"point source, $3\sigma$ isophote")
    ax.plot(xx, kfac * point_source_size(xx, 3), color="#c1440e", lw=1.1,
            label=r"$\times\,%.2f$ (SoFiA ellipse)" % kfac)
    ax.set_xscale("log")
    ax.set_xlim(2.0, max(120.0, np.nanpercentile(x, 99.5)))
    ax.set_ylim(0, min(4.0, np.nanpercentile(y, 99.5) * 1.05))
    ax.set_xlabel(r"${\rm SNR}_{\rm int}$")
    ax.set_ylabel(r"SoFiA $3\sigma$ major axis / $\theta_{\rm beam}$")
    cb = fig.colorbar(sc, ax=ax, pad=0.015, aspect=18)          # thicker bar
    cb.set_label(r"$\log_{10}\,d_{\rm HI}/\theta_{\rm beam}$")
    cb.solids.set_alpha(1)
    ax.legend(loc="upper left")
    fig_size.kfac = kfac
    return save(fig, outdir, "fig06_size")


# --- detectability against position and frequency --------------------------

def geometry(fields_dir, field):
    """Pointing centre (pixels) and pixel size (arcsec) from the substrate header."""
    try:
        import yaml
        from astropy.io import fits
        cfg = field_cfg(fields_dir, field)
        h = fits.getheader(cfg["substrate"])
        return (float(h["CRPIX1"]) - 1.0, float(h["CRPIX2"]) - 1.0,
                3600.0 * abs(float(h["CDELT1"])))
    except Exception:
        return None


def residuals(res, fields_dir):
    """Per field: offsets (arcmin), radius, frequency, detected - P(SNR), P."""
    out = {}
    for f in res:
        d, fit = res[f], res[f]["fit_snr"]
        t = d["t"]
        x, y = col(t, "xpix"), col(t, "ypix")
        g = geometry(fields_dir, f)
        if g is None:
            xc, yc, pix = np.nanmedian(x), np.nanmedian(y), 6.0
        else:
            xc, yc, pix = g
        dx, dy = (x - xc) * pix / 60.0, (y - yc) * pix / 60.0
        p = logistic(d["logsnr"], fit["x0"], fit["k"])
        out[f] = dict(dx=dx, dy=dy, r=np.hypot(dx, dy),
                      freq=F0_MHZ * (1.0 - col(t, "vsys_kms") / C_KMS),
                      res=d["det"].astype(float) - p, p=p)
    return out


def binned(v, r, p, edges, min_n=25):
    xc, mu, er = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (v >= a) & (v < b) & np.isfinite(r)
        if m.sum() < min_n:
            continue
        xc.append(np.median(v[m])); mu.append(r[m].mean())
        er.append(np.sqrt(np.sum(p[m] * (1 - p[m]))) / m.sum())
    return np.array(xc), np.array(mu), np.array(er)


def fig_position_maps(rr, outdir, nb=8, min_n=12, vlim=0.2):
    fs = ordered(rr)
    R = max(np.nanpercentile(rr[f]["r"], 99.5) for f in fs)
    edges = np.linspace(-R, R, nb + 1)
    ncol = 4
    nrow = int(np.ceil(len(fs) / ncol))
    fig = plt.figure(figsize=(W["full"] * MM, 0.215 * W["full"] * MM * nrow))
    axes, first, last, below, _ = centred_axes(fig, len(fs), ncol, wspace=0.12, hspace=0.32)
    im = None
    for ax, f in zip(axes, fs):
        d = rr[f]
        m = np.full((nb, nb), np.nan); z = np.full((nb, nb), np.nan)
        ix = np.clip(np.digitize(d["dx"], edges) - 1, 0, nb - 1)
        iy = np.clip(np.digitize(d["dy"], edges) - 1, 0, nb - 1)
        for j in range(nb):
            for i in range(nb):
                s = (ix == i) & (iy == j)
                if s.sum() >= min_n:
                    m[j, i] = d["res"][s].mean()
                    z[j, i] = m[j, i] / (np.sqrt(np.sum(d["p"][s] * (1 - d["p"][s]))) / s.sum())
        ok = np.isfinite(z)
        im = ax.imshow(m, origin="lower", extent=[-R, R, -R, R], cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim, interpolation="nearest")
        ax.set_title(label(f), fontsize=min(plt.rcParams["font.size"] - 0.5, pt(8.0)), pad=2)
        ax.text(0.03, 0.97, r"$\chi^2_\nu$ %.2f" % (np.sum(z[ok] ** 2) / max(ok.sum(), 1)),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=plt.rcParams["legend.fontsize"] - 0.5, color=INK)
        ax.set_aspect("equal")
        ax.minorticks_off()
    # one label per axis for the whole grid: with a centred last row, per-panel
    # labels collided with the tick labels of the row above
    fig.supxlabel(r"$\Delta x$ [arcmin]", fontsize=plt.rcParams["axes.labelsize"],
                  x=0.48, y=0.0, va="bottom")
    fig.supylabel(r"$\Delta y$ [arcmin]", fontsize=plt.rcParams["axes.labelsize"],
                  x=0.02, ha="left")
    # a colour bar of its own, right of the grid, so it never covers a panel
    fig.subplots_adjust(left=0.09, right=0.845, bottom=0.10)
    cax = fig.add_axes([0.868, 0.18, 0.032, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\Delta P_{\rm det}$ = detected $-$ $P_{\rm det}$(SNR)")
    return save(fig, outdir, "fig07_position_maps")


def fig_radius_freq(rr, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(W["full"] * MM, 0.34 * W["full"] * MM),
                             sharey=True)
    for ax, key, edges_n in ((axes[0], "r", 10), (axes[1], "freq", 18)):
        allv = np.concatenate([rr[f][key] for f in rr])
        allr = np.concatenate([rr[f]["res"] for f in rr])
        allp = np.concatenate([rr[f]["p"] for f in rr])
        g = np.isfinite(allv)
        edges = np.linspace(np.nanmin(allv[g]), np.nanmax(allv[g]), edges_n + 1)
        for f in ordered(rr):
            st = fstyle(f)
            xc, mu, er = binned(rr[f][key], rr[f]["res"], rr[f]["p"],
                                np.linspace(np.nanmin(rr[f][key]), np.nanmax(rr[f][key]), 7),
                                min_n=30)
            ax.plot(xc, mu, color=st["color"], ls=st["ls"], lw=0.6, alpha=0.8)
        xc, mu, er = binned(allv, allr, allp, edges, min_n=60)
        ax.errorbar(xc, mu, yerr=er, fmt="o", ms=3.2, color=INK, elinewidth=0.8,
                    capsize=1.5, zorder=5)
        if key == "freq" and len(mu):
            i = int(np.argmin(mu))
            fig_radius_freq.dip = (float(mu[i]), float(xc[i]))
        ax.axhline(0, color=GREY, lw=0.6, ls=":")
    axes[0].set_xlabel("distance from centre [arcmin]")
    axes[1].set_xlabel("frequency [MHz]")
    axes[0].set_ylabel(r"$\Delta P_{\rm det}$")
    lim = 0.05
    for ax in axes:
        for ln in ax.lines:
            yd = np.asarray(ln.get_ydata(), float)
            if yd.size and np.isfinite(yd).any():
                lim = max(lim, np.nanmax(np.abs(yd)))
    axes[0].set_ylim(-1.15 * lim, 1.15 * lim)
    top = axes[1].secondary_xaxis(
        "top", functions=(lambda fq: C_KMS * (1 - fq / F0_MHZ) / 1e3,
                          lambda v: F0_MHZ * (1 - 1e3 * v / C_KMS)))
    top.set_xlabel(r"$cz_{\rm radio}$ [10$^3$ km s$^{-1}$]")
    panel_tag(axes[0], "(a)"); panel_tag(axes[1], "(b)")
    axes[0].text(0.97, 0.05, "black: all fields\nlines: individual fields",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=plt.rcParams["legend.fontsize"], color=GREY)
    fig.subplots_adjust(wspace=0.05 + 0.08 * (FONT_SCALE - 1) * 4)
    field_legend(fig, rr, ncol=6, y=-0.05)
    return save(fig, outdir, "fig08_radius_freq")


# --- showcase: best and worst recovery in six fields --------------------------

def load_stamps(runs):
    out = {}
    for p in sorted(glob.glob(str(Path(runs) / "*" / "run_*" / "products" / "stamps.npz"))):
        f = Path(p).parts[len(Path(runs).parts)]
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            d = json.loads(str(z[k]))
            if not d.get("neighbour"):
                out.setdefault(f, []).append(d)
    return out


def score(d):
    """Distance from a perfect recovery: |ln capture| + |ln fidelity|."""
    c, fi = d.get("capture", np.nan), d.get("fidelity", np.nan)
    if not (np.isfinite(c) and np.isfinite(fi)) or c <= 0 or fi <= 0:
        return np.nan
    return abs(np.log(c)) + abs(np.log(fi))


def pick_fields(res, stamps, n=6, wanted=None):
    have = [f for f in order(res) if len(stamps.get(f, [])) >= 2]
    if wanted:
        return [f for f in wanted if f in have]
    if len(have) <= n:
        return have
    return [have[int(round(i))] for i in np.linspace(0, len(have) - 1, n)]


ISO_COLOUR = "#EE7733"
RESID_NSIG = 3.0        # residual colour scale: +-3 sigma of the moment-0 noise


def centred_velocity(d):
    v = np.array(d["vel"], float)
    sp = np.clip(np.array(d["spec_inj"], float), 0, None)
    vc = np.sum(v * sp) / np.sum(sp) if np.sum(sp) > 0 else np.median(v)
    return v - vc, sp


def fsz(size):
    """Showcase type: the scaled size, a little smaller on narrow pages."""
    return pt(size) if W["full"] >= 170 else round(pt(size) * 0.92, 2)


def showcase_panels(fig, gs, row, col0, d, first_row, last_row, vlim):
    pix, bmaj = d["pix"], d["bmaj"]
    m0, r0, rs = (np.array(d["model_mom0"]), np.array(d["recov_mom0"]),
                  np.array(d["resid_mom0"]))
    ny, nx = m0.shape
    ext = np.array([-nx / 2, nx / 2, -ny / 2, ny / 2]) * pix / 60.0     # arcmin
    vmax = np.nanmax(np.abs(m0)) or 1.0
    sig0 = d["sigma_chan"] * d["dv"] * np.sqrt(max(d["nchan"], 1))
    ok0 = np.isfinite(sig0) and sig0 > 0
    xx = (np.arange(nx) - nx / 2 + 0.5) * pix / 60.0
    yy = (np.arange(ny) - ny / 2 + 0.5) * pix / 60.0
    rnorm = sig0 if ok0 else vmax                 # residual unit: sigma_mom0
    rmax = RESID_NSIG if ok0 else 0.6
    for j, (img, cmap, lim) in enumerate(((m0 / vmax, "cividis", (0, 1)),
                                          (r0 / vmax, "cividis", (0, 1)),
                                          (rs / rnorm, "RdBu_r", (-rmax, rmax)))):
        ax = fig.add_subplot(gs[row, col0 + j])
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1],
                  interpolation="nearest")
        if j < 2 and ok0:
            # isophotes of the noiseless model, drawn on both maps
            for n, ls_ in ((1, (0, (2.0, 1.4))), (3, "-")):
                if np.nanmax(m0) > n * sig0:
                    ax.contour(xx, yy, m0, levels=[n * sig0], colors=[ISO_COLOUR],
                               linewidths=0.7, linestyles=[ls_])
        ax.add_patch(Circle((ext[0] + 0.75 * bmaj / 60, ext[2] + 0.75 * bmaj / 60),
                            0.5 * bmaj / 60, fc="none", ec="white" if j < 2 else "#555555",
                            lw=0.5))
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_linewidth(0.4)
        if first_row:
            ax.set_title(("injected", "recovered", "residual")[j], fontsize=fsz(8.0), pad=2.5)
    ax = fig.add_subplot(gs[row, col0 + 3])
    dv, sp = centred_velocity(d)
    rec = np.array(d["spec_rec"], float)
    ax.plot(dv, sp * 1e3, color="#0077BB", lw=0.9)
    ax.plot(dv, rec * 1e3, color="#EE7733", lw=0.9, ls="--")
    ax.axhline(0, color=LIGHT, lw=0.4)
    top = 1e3 * max(np.nanmax(sp), np.nanmax(rec))
    bot = 1e3 * min(0.0, np.nanmin(rec))
    ax.set_ylim(bot - 0.05 * (top - bot), bot + 1.55 * (top - bot))
    ax.set_xlim(-vlim, vlim)
    ax.tick_params(labelsize=fsz(7.5), length=2)
    ax.minorticks_off()
    ax.yaxis.tick_right()
    if not last_row:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r"$v-v_{\rm c}$ [km s$^{-1}$]", fontsize=fsz(8.0), labelpad=1.5)
    ax.text(0.03, 0.95, "SNR %.0f, $d/\\theta$ %.1f\n$C$ %.2f, $\\mathcal{F}$ %.2f" % (
        d["snr"], d["dratio"], d["capture"], d["fidelity"]),
            transform=ax.transAxes, va="top", ha="left", fontsize=fsz(7.2), color=INK,
            linespacing=1.25)
    if first_row:
        ax.set_title("spectrum [mJy]", fontsize=fsz(8.0), pad=2.5)
    return ax


def fig_showcase(res, runs, outdir, wanted=None, n=3, worst_by="score"):
    stamps = load_stamps(runs)
    fs = pick_fields(res, stamps, n, wanted)
    if not fs:
        print("  (no showcase: no stamps.npz files)")
        return None
    rows, used = [], set()
    key = lambda d: (round(d["snr"], 1), round(d["dratio"], 2), round(d.get("w50", 0)))
    for f in fs:
        cand = [d for d in stamps[f] if np.isfinite(score(d)) and key(d) not in used]
        if len(cand) < 2:
            continue
        best = min(cand, key=lambda d: score(d) - 0.05 * min(d["dratio"], 2.0))
        if worst_by == "capture":
            worst = min(cand, key=lambda d: d["capture"])
        elif worst_by == "fidelity":
            worst = max(cand, key=lambda d: abs(np.log(d["fidelity"])))
        else:
            worst = max(cand, key=score)
        used.update({key(best), key(worst)})
        rows.append((f, best, worst))
    vlim = 0.0
    for _, b_, w_ in rows:
        for d in (b_, w_):
            dv, sp = centred_velocity(d)
            if sp.max() > 0:
                vlim = max(vlim, np.max(np.abs(dv[sp > 0.01 * sp.max()])))
    vlim = 100.0 * np.ceil(1.1 * vlim / 100.0) if vlim > 0 else 500.0
    # Layout: one row per example, the best above the worst for each field, with
    # four columns (injected, recovered, residual, spectrum) across the page.
    # This gives every map about twice the area of the side-by-side layout.
    nf = len(rows)
    order_ = []                                   # (gridspec row, field, case, record)
    hr = []
    for k, (f, best, worst) in enumerate(rows):
        if k:
            hr.append(0.22)                       # gap between fields
        order_.append((len(hr), f, "best", best)); hr.append(1.0)
        order_.append((len(hr), f, "worst", worst)); hr.append(1.0)
    ws_ = 2.0                                     # spectrum width, in map widths
    wr = [1, 1, 1, ws_]
    left, right, wsp, hsp = 0.095, 0.925, 0.06, 0.10
    head = 6.0 * FONT_SCALE                       # column titles
    two_rows = W["full"] < 170                    # the legend needs two rows on narrow pages
    strip = (25.5 + (5.0 if two_rows else 0.0)) * FONT_SCALE   # axis label, colour bars, legend
    hmax = 180.0 if W["full"] < 170 else 205.0    # leave room for the caption on the page
    width = W["full"]
    for _ in range(3):                            # shrink the width if the page is too short
        u = width * (right - left) / (sum(wr) + wsp * (len(wr) - 1) * np.mean(wr))
        grid = u * (sum(hr) + hsp * (len(hr) - 1) * np.mean(hr))
        H = grid + head + strip
        if H <= hmax:
            break
        width *= (hmax - head - strip) / grid
    fig = plt.figure(figsize=(width * MM, H * MM))
    gs = fig.add_gridspec(len(hr), 4, width_ratios=wr, height_ratios=hr, wspace=wsp, hspace=hsp,
                          left=left, right=right, top=1 - head / H, bottom=strip / H)
    last = max(r for r, *_ in order_)
    for r, f, case, d in order_:
        showcase_panels(fig, gs, r, 0, d, r == 0, r == last, vlim)
        bb = fig.axes[-4].get_position()
        fig.text(0.058, 0.5 * (bb.y0 + bb.y1), case, rotation=90, ha="center", va="center",
                 fontsize=fsz(7.5), style="italic", color=GREY)
        if case == "best":
            bw_ = bb
        else:
            fig.text(0.022, 0.5 * (bw_.y1 + bb.y0), label(f), rotation=90, ha="center",
                     va="center", fontsize=fsz(8.5))
    # colour bars under the columns they describe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    # the pair of colour bars centred under the figure, the moment-0 bar wider
    yb, hb = (strip - 13.0 * FONT_SCALE) / H, 2.8 / H     # bars just under the bottom row
    w_mom, w_res, gap = 0.34, 0.20, 0.07
    x_mom = 0.5 - 0.5 * (w_mom + gap + w_res)
    for x0, wd, cmap, lim, lab in (
            (x_mom, w_mom, "cividis", (0, 1), "moment 0 / injected peak"),
            (x_mom + w_mom + gap, w_res, "RdBu_r", (-RESID_NSIG, RESID_NSIG),
             r"residual / $\sigma_{\rm mom0}$")):
        cax = fig.add_axes([x0, yb, wd, hb])
        cb = fig.colorbar(ScalarMappable(Normalize(*lim), cmap=cmap), cax=cax,
                          orientation="horizontal")
        cb.set_ticks([0, 0.5, 1] if lim[0] == 0 else [lim[0], 0, lim[1]])
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=fsz(7.5), length=2.5, width=0.5, pad=1.5)
        cb.ax.minorticks_off()
        cb.set_label(lab, fontsize=fsz(8.0), labelpad=2)
    fig.legend(handles=[Line2D([], [], color="#0077BB", lw=1.0, label="injected"),
                        Line2D([], [], color="#EE7733", lw=1.0, ls="--", label="recovered (in mask)"),
                        Line2D([], [], color=ISO_COLOUR, lw=0.8, ls=(0, (2.0, 1.4)),
                               label=r"model $1\sigma$ isophote"),
                        Line2D([], [], color=ISO_COLOUR, lw=0.8, label=r"model $3\sigma$ isophote"),
                        Line2D([], [], color=GREY, lw=0, marker="o", mfc="none", ms=4.5,
                               label="beam")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3 if two_rows else 5,
               fontsize=fsz(7.5), handlelength=2.0, columnspacing=1.0)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig09_showcase.pdf", bbox_inches=None)
    try:
        fig.savefig(out / "fig09_showcase.png", dpi=200, bbox_inches=None)
    except (RuntimeError, FileNotFoundError):
        print("  (fig09_showcase: PDF written; no PNG preview without dvipng)")
    plt.close(fig)
    return str(out / "fig09_showcase.pdf"), rows


# ---------------------------------------------------------------------------
# figures added for the thesis examiners' points (see docstring of each)
# ---------------------------------------------------------------------------

# Cluster redshifts from thesis Table 2.1 (two decimals there). A field.yaml
# key "z" overrides them.
CLUSTER_Z = {"Abell-194": 0.02, "Abell-4038": 0.03, "Abell-548": 0.04, "Abell-168": 0.05,
             "Abell-3562": 0.05, "J2319.2-6750": 0.03, "J0600.8-5835": 0.04,
             "J0757.7-5315": 0.04, "J0351.1-8212": 0.06, "J0314.3-4525": 0.07,
             "J0431.4-6126": 0.06, "J0540.1-4050": 0.04,
             "Abell-85": 0.055, "Abell-3376": 0.046}
W_REF = 200.0                                   # km/s, reference linewidth for limits
S_LABEL = r"$\log_{10}(S_{\rm int}/{\rm Jy\,km\,s^{-1}})$"


FILE_KEYS = ("cube", "mask", "science_par", "science_catalogue", "rms_cube",
             "substrate", "noise_map")


def field_cfg(fields_dir, f):
    """A field's config, with relative paths made absolute.

    Campaign configs may give paths relative to the campaign's working folder
    (the parent of fields/); resolving them here lets the figures be made from
    any directory.
    """
    try:
        import yaml
        cfg = yaml.safe_load(open(Path(fields_dir) / f / "field.yaml")) or {}
    except Exception:
        return {}
    base = Path(fields_dir).resolve().parent
    for k in FILE_KEYS:
        v = cfg.get(k)
        if v and not Path(str(v)).is_absolute():
            cfg[k] = str(base / str(v))
    return cfg


def cluster_z(fields_dir, f):
    return float(field_cfg(fields_dir, f).get("z", CLUSTER_Z.get(f, np.nan)))


def chan_width(res, f):
    m = getattr(res[f]["t"], "meta", {}) or {}
    return abs(float((m.get("temba") or {}).get("dv_chan_kms", 44.11)))


def s_point(snr, sigma_jy, dv, w50=W_REF):
    """Integrated flux of an unresolved source at a given integrated SNR."""
    return snr * sigma_jy * np.sqrt(dv * w50)


def lum_dist(z):
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=70.0, Om0=0.3).luminosity_distance(np.asarray(z, float)).value


def fig_per_field(res, outdir, axis="flux", edges=None):
    """Per-field completeness diagnostics, replacing thesis Figs 2.28-2.32.

    axis="flux" (fig10) or axis="snr" (fig14, completeness against integrated
    SNR at each source's local noise, as Examiner A asks). Every panel uses the
    same axis range and the same fixed bins for the histograms and the
    completeness points, so each point sits on its own histogram bar. The
    curve is the unbinned logistic fit; the 50% and 90% points are marked and
    labelled. In flux, the grey dotted lines are the fluxes of an unresolved
    source with W50 = 200 km/s at integrated SNR 1 and 3 in the field's median
    noise, S = n sigma sqrt(dv W50).
    """
    snr = axis == "snr"
    if edges is None:
        edges = (np.arange(-0.5, 2.0 + 1e-9, 0.125) if snr
                 else np.arange(-2.0, 1.2 + 1e-9, 0.2))
    fs = ordered(res)
    ncol = 4
    nrow = int(np.ceil(len(fs) / ncol))
    fig = plt.figure(figsize=(W["full"] * MM, (0.27 * W["full"] * nrow + 30 * FONT_SCALE) * MM))
    axes, first, last, below, _ = centred_axes(fig, len(fs), ncol)
    xc = 0.5 * (edges[:-1] + edges[1:])
    sym = r"{\rm SNR}" if snr else "S"
    for k, (ax, f) in enumerate(zip(axes, fs)):
        d, st = res[f], fstyle(f)
        fit = d["fit_snr"] if snr else d["fit_flux"]
        x, det = (d["logsnr"] if snr else d["logflux"]), d["det"]
        n_i = np.histogram(x, edges)[0]
        n_r = np.histogram(x[det], edges)[0]
        tw = ax.twinx()
        w = 0.92 * (edges[1] - edges[0])
        tw.bar(xc, n_i, width=w, color=LIGHT, edgecolor="none", zorder=0)
        tw.bar(xc, n_r, width=w, color=st["color"], alpha=0.25, edgecolor="none", zorder=0)
        tw.set_ylim(0, 2.4 * max(n_i.max(), 1))
        tw.tick_params(axis="y", labelsize=pt(6), colors=GREY, length=2)
        tw.minorticks_off()
        if not last[k]:
            tw.set_yticklabels([])
        ok = n_i > 0
        p, lo, hi = wilson(n_r[ok], n_i[ok])
        ax.errorbar(xc[ok], p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                    fmt="o", ms=2.6, color=INK, elinewidth=0.6, capsize=0, zorder=4)
        xx = np.linspace(edges[0], edges[-1], 400)
        ax.plot(xx, logistic(xx, fit["x0"], fit["k"]), color=st["color"], lw=1.1, zorder=3)
        for lv in (0.5, 0.9):
            ax.axhline(lv, color=LIGHT, lw=0.5, ls=":", zorder=1)
        ax.axvline(fit["logS50"], color=st["color"], lw=0.8, ls="-", zorder=2)
        ax.axvline(fit["logS90"], color=st["color"], lw=0.8, ls="--", zorder=2)
        if not snr:
            sig = d["sigma_med"]
            for n in (1, 3):
                ax.axvline(np.log10(s_point(n, sig, chan_width(res, f))), color=GREY, lw=0.6,
                           ls=(0, (1, 1.5)), zorder=2)
        ax.set_zorder(tw.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_title(label(f) + "\n" + r"$%s_{90}$ = %.*f" % (
                         sym, 1 if snr else 2, 10 ** fit["logS90"]),
                     fontsize=plt.rcParams["font.size"] - 0.5, pad=2, linespacing=1.15)
        ax.set_ylim(-0.03, 1.05)
    for ax, f0 in zip(axes, first):
        if f0:
            ax.set_ylabel("completeness")
    handles = [
        Line2D([], [], marker="o", ls="", color=INK, ms=3, label="recovered / injected (68" + PCT + " Wilson)"),
        Line2D([], [], color=GREY, lw=1.0, label="logistic fit (unbinned)"),
        Line2D([], [], color=GREY, lw=0.8, label=r"$%s_{50}$" % sym),
        Line2D([], [], color=GREY, lw=0.8, ls="--", label=r"$%s_{90}$" % sym),
        Line2D([], [], color=LIGHT, lw=5, label="injected (right axis)")]
    if not snr:
        handles.insert(4, Line2D([], [], color=GREY, lw=0.6, ls=(0, (1, 1.5)),
                       label=r"SNR$_{\rm int}$ = 1 and 3 (unresolved)"))
    H = fig.get_size_inches()[1] * 25.4                     # figure height, mm
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3 if W["full"] >= 170 else 2, fontsize=plt.rcParams["legend.fontsize"],
               columnspacing=1.0, handlelength=1.6)
    # rows of panels, then the shared axis label, then the legend underneath
    fig.subplots_adjust(wspace=0.22 * FONT_SCALE, hspace=0.50 * FONT_SCALE,
                        bottom=(30 * FONT_SCALE) / H, top=1 - 10 * FONT_SCALE / H)
    fig.supxlabel(r"$\log_{10}\,{\rm SNR}_{\rm int}$" if snr else S_LABEL,
                  fontsize=plt.rcParams["axes.labelsize"], y=(18.5 * FONT_SCALE) / H,
                  va="bottom")
    return save(fig, outdir, "fig14_per_field_snr" if snr else "fig10_per_field")


def binned_fits(x, det, v, edges, min_n=150):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (v >= a) & (v < b)
        if m.sum() < min_n or det[m].sum() < 10 or (~det[m]).sum() < 10:
            continue
        fit = fit_logistic(x[m], det[m], n_boot=100)
        out.append((np.median(v[m]), fit["logS50"], fit["err_logS50"], fit["k"], int(m.sum())))
    return np.array(out)


def fig_width_size(res, outdir):
    """Does completeness depend on linewidth and size beyond what SNR includes?

    (a) and (b): SNR50 from logistic fits in bins of W50 and of d_HI/theta,
    pooling every field (SNR already uses each source's local noise). Flat
    means the integrated SNR captures that dependence. (c): the flux view
    that HIPASS and ALFALFA use, S50 divided by each field's median channel
    noise, against W50, with the W50^(1/2) scaling of a matched filter.
    """
    lsnr = np.concatenate([res[f]["logsnr"] for f in res])
    det = np.concatenate([res[f]["det"] for f in res])
    w50 = np.concatenate([col(res[f]["t"], "W50_kms") for f in res])
    drat = np.concatenate([col(res[f]["t"], "dHI_over_beam") for f in res])
    lfn = np.concatenate([res[f]["logflux"] - np.log10(res[f]["sigma_med"]) for f in res])
    fig, axes = plt.subplots(1, 3, figsize=(W["full"] * MM, 0.32 * W["full"] * MM))
    for ax, v, xl, logx in ((axes[0], w50, r"$W_{50}$ [km s$^{-1}$]", True),
                            (axes[1], drat, r"$d_{\rm HI}/\theta_{\rm beam}$", True)):
        e = np.quantile(v[np.isfinite(v)], np.linspace(0, 1, 9))
        b = binned_fits(lsnr, det, v, e)
        if len(b):
            ax.errorbar(b[:, 0], 10 ** b[:, 1], yerr=[10 ** b[:, 1] - 10 ** (b[:, 1] - b[:, 2]),
                        10 ** (b[:, 1] + b[:, 2]) - 10 ** b[:, 1]], fmt="o", ms=3.5, color=INK,
                        elinewidth=0.8, capsize=1.5)
            ax.axhline(10 ** np.median(b[:, 1]), color=GREY, lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_xlabel(xl)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    for ax, ticks in ((axes[0], [60, 100, 200, 400]), (axes[1], [0.3, 1, 3])):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(r"${\rm SNR}_{50}$ (all fields)")
    e = np.quantile(w50[np.isfinite(w50)], np.linspace(0, 1, 9))
    b = binned_fits(lfn, det, w50, e)
    ax = axes[2]
    if len(b):
        ax.errorbar(b[:, 0], 10 ** b[:, 1], yerr=[10 ** b[:, 1] - 10 ** (b[:, 1] - b[:, 2]),
                    10 ** (b[:, 1] + b[:, 2]) - 10 ** b[:, 1]], fmt="s", ms=3.5, color=INK,
                    elinewidth=0.8, capsize=1.5)
        ww = np.geomspace(b[:, 0].min() * 0.8, b[:, 0].max() * 1.2, 80)
        low = b[:, 0] < 200
        ref = 10 ** np.median((b[:, 1] - 0.5 * np.log10(b[:, 0]))[low if low.any() else slice(None)])
        ax.plot(ww, ref * ww ** 0.5, color="#c1440e", lw=0.9, ls="--",
                label=r"$\propto W_{50}^{1/2}$")
        # ALFALFA's 50% limit has this shape (Haynes et al. 2011): W^1/2 below
        # log W50 = 2.5, W^1 above; drawn here with our normalisation
        wb = 10 ** 2.5
        alf = np.where(ww < wb, ref * ww ** 0.5, ref * wb ** 0.5 * (ww / wb))
        ax.plot(ww, alf, color=GREY, lw=0.9, ls=":", label="ALFALFA")
        ax.legend(loc="lower right", fontsize=plt.rcParams["legend.fontsize"] - 0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$W_{50}$ [km s$^{-1}$]")
    ax.set_ylabel(r"$S_{50}/\sigma_{\rm chan}$ [km s$^{-1}$]")
    axes[2].xaxis.set_major_locator(FixedLocator([60, 100, 200, 400]))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].yaxis.set_major_locator(FixedLocator([300, 500, 1000, 2000, 3000]))
    for a_, t, x in zip(axes, ("(a)", "(b)", "(c)"), (0.88, 0.88, 0.03)):
        panel_tag(a_, t, x=x)
    fig.subplots_adjust(wspace=0.35 * FONT_SCALE ** 2)
    return save(fig, outdir, "fig11_width_size")


_NOISE_REPORTED = set()


def read_release(fields_dir, f, sigma_ref=None):
    """The field's released SoFiA catalogue in physical units, or None.

    Besides flux, W50, velocity and SoFiA's 3-sigma size, it returns
      drat    d_HI / beam, with d_HI from the H I size-mass relation of
              Wang et al. (2016), log D_HI = 0.506 log M_HI - 3.293 (kpc, Msun);
      snr_ps  the point-source integrated SNR, S / (sigma sqrt(dv W50)), with
              sigma the local channel noise at the source position.
    The local noise is the field's noise map scaled so that its median equals
    sigma_ref, the median sigma (Jy/beam) the injections saw. That makes the
    result independent of the units the noise map happens to be stored in.
    """
    if not (field_cfg(fields_dir, f) or {}).get("science_catalogue"):
        return None                      # no catalogue given for this field: nothing to compare
    try:
        from astropy.table import Table
        from astropy.io import fits
        from .match import velocity_axis, fsum_to_jykms, pick
        cfg = field_cfg(fields_dir, f)
        cat = Table.read(cfg["science_catalogue"], format="votable")
        h = fits.getheader(cfg["substrate"])
        dv = abs(velocity_axis(h)[1])
        bmaj, bmin = float(h["BMAJ"]), float(h.get("BMIN", h["BMAJ"]))
        if bmaj < 1.0:
            bmaj, bmin = bmaj * 3600, bmin * 3600
        pix = 3600 * 0.5 * (abs(float(h["CDELT1"])) + abs(float(h["CDELT2"])))
        barea = np.pi / (4 * np.log(2)) * bmaj * bmin / pix ** 2
        s_, _ = fsum_to_jykms(cat, pick(cat, ["f_sum"], True), dv, barea)
        v = np.asarray(cat[pick(cat, ["v_rad", "v_opt", "v_app"], True)], float)
        if np.nanmedian(np.abs(v)) > 1e5:
            v = v / 1e3
        w = np.asarray(cat[pick(cat, ["w50"], True)], float)
        if np.nanmedian(np.abs(w)) > 1e4:
            w = w / 1e3
        e3 = np.asarray(cat[pick(cat, ["ell3s_maj"], True)], float) * pix / bmaj
        z = 1.0 / (1.0 - v / C_KMS) - 1.0
        dl = lum_dist(z)
        m = 2.356e5 * dl ** 2 * s_
        d_kpc = 10 ** (0.506 * np.log10(np.clip(m, 1.0, None)) - 3.293)
        d_as = d_kpc / (1e3 * dl / (1 + z) ** 2) * 206264.8
        drat = d_as / bmaj
        snr = np.full(len(cat), np.nan)
        try:
            sig = np.asarray(fits.getdata(cfg["noise_map"]), float).squeeze()
            if sig.ndim == 3 and sig.shape[0] == 2:
                sig = sig[0]                  # TEMBA's cache: [sigma_chan, valid fraction]
            elif sig.ndim == 3:
                sig = np.nanmedian(sig, axis=0)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
            med = np.nanmedian(sig)
            scale = 1.0
            if sigma_ref and not 0.5 < med / sigma_ref < 2.0:
                scale = sigma_ref / med       # stored in other units: put it in Jy/beam
            if f not in _NOISE_REPORTED:
                _NOISE_REPORTED.add(f)
                print(f"  {f}: noise map median {med * 1e3:.3f} mJy/beam; injections "
                      f"{(sigma_ref or med) * 1e3:.3f}" + ("" if scale == 1.0 else
                      f"; map rescaled by {scale:.3g}"))
            x = np.clip(np.round(np.asarray(cat[pick(cat, ["x"], True)], float)).astype(int),
                        0, sig.shape[1] - 1)
            y = np.clip(np.round(np.asarray(cat[pick(cat, ["y"], True)], float)).astype(int),
                        0, sig.shape[0] - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = s_ / (scale * sig[y, x] * np.sqrt(dv * w))
        except Exception as e:
            print(f"  ({f}: no SNR for released sources: {e})")
        return dict(S=s_, w50=w, e3=e3, z=z, M=m, v=v, drat=drat, snr_ps=snr)
    except Exception as e:
        print(f"  ({f}: release catalogue not read: {e})")
        return None


def fig_mass_limits(res, fields_dir, outdir):
    """HI mass detection limits, the theoretical limit for thesis Fig. 2.11.

    For each field, the HI mass of an unresolved W50 = 200 km/s source at the
    field's SNR90 and median noise, against distance, with the cluster's own
    distance marked; the released detections are shown behind.
    """
    fig, ax = plt.subplots(figsize=(W["col"] * MM, 0.85 * W["col"] * MM))
    zz = np.linspace(0.003, 0.08, 200)
    dl = lum_dist(zz)
    for f in ordered(res):
        rel = read_release(fields_dir, f, res[f]["sigma_med"])
        if rel is not None:
            g = np.isfinite(rel["M"]) & (rel["M"] > 0)
            ax.plot(lum_dist(rel["z"][g]), np.log10(rel["M"][g]), ".", ms=1.6, color=LIGHT,
                    zorder=0, rasterized=True)
    for f in ordered(res):
        st, d = fstyle(f), res[f]
        s90 = s_point(10 ** d["fit_snr"]["logS90"], d["sigma_med"], chan_width(res, f))
        ax.plot(dl, np.log10(2.356e5 * dl ** 2 * s90), color=st["color"], ls=st["ls"], lw=0.8)
        zc = cluster_z(fields_dir, f)
        if np.isfinite(zc):
            dc = lum_dist(zc)
            ax.errorbar(dc, np.log10(2.356e5 * dc ** 2 * s90),
                        yerr=d["fit_snr"].get("err_logS90", np.nan), fmt=st["marker"],
                        color=st["color"], ms=4, mec="white", mew=0.4, elinewidth=0.7,
                        capsize=1.2, zorder=5)
    ax.set_xlabel(r"luminosity distance [Mpc]")
    ax.set_ylabel(r"$\log_{10}(M_{\rm HI}/{\rm M}_\odot)$")
    ax.set_xlim(0, lum_dist(0.08)); ax.set_ylim(7.0, 10.6)
    ax.text(0.97, 0.05, "curves: 90%s limit, unresolved,\n$W_{50}$ = %d km s$^{-1}$; markers: cluster"
            % (PCT, W_REF), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=plt.rcParams["legend.fontsize"] - 0.5, color=GREY)
    h = [Line2D([], [], color=fstyle(f)["color"], ls=fstyle(f)["ls"], lw=0.8,
                marker=fstyle(f)["marker"], ms=3.2, label=label(f))
         for f in ordered(res)]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=plt.rcParams["legend.fontsize"] - 0.5, handlelength=2.4)
    return save(fig, outdir, "fig12_mass_limits")


def fig_coverage(res, fields_dir, outdir):
    """Do the injected sources span the real detections?

    Six quantities: integrated flux, W50, velocity, integrated SNR, d_HI/beam
    and SoFiA's 3-sigma size. Injected and recovered use the true values of
    all injected sources and of the detected ones (SoFiA's measurement for the
    3-sigma size, which has no true value); the released catalogue uses its
    measured values. The SNR panel uses the point-source form,
    S / (sigma sqrt(dv W50)), for all three populations, with sigma the local
    channel noise, because it needs no size: the design SNR's size term would
    need a true d_HI that real galaxies do not have. The released histogram
    carries Poisson errors.
    """
    inj = {k: [] for k in ("S", "w50", "v", "snr_ps", "drat")}
    rec = {k: [] for k in ("S", "w50", "v", "snr_ps", "drat", "e3")}
    for f in res:
        t, det = res[f]["t"], res[f]["det"]
        vals = dict(S=col(t, "Fint_injected_Jykms"), w50=col(t, "W50_kms"),
                    v=col(t, "vsys_kms"), drat=col(t, "dHI_over_beam"))
        vals["snr_ps"] = vals["S"] / (col(t, "sigma_chan_Jybeam")
                                      * np.sqrt(chan_width(res, f) * vals["w50"]))
        for k, a_ in vals.items():
            inj[k].append(a_); rec[k].append(a_[det])
        beam = col(t, "dHI_arcsec") / col(t, "dHI_over_beam")
        rec["e3"].append((col(t, "ell3s_maj_sofia_arcsec") / beam)[det])
    inj = {k: np.concatenate(v) for k, v in inj.items()}
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    rel = [read_release(fields_dir, f, res[f]["sigma_med"]) for f in res]
    rel = [r for r in rel if r is not None]
    keys = ("S", "w50", "v", "snr_ps", "drat", "e3")
    real = {k: np.concatenate([r[k] for r in rel]) for k in keys} if rel else None
    fig, axes = plt.subplots(2, 3, figsize=(W["full"] * MM, 0.52 * W["full"] * MM))
    spec = (("S", "log", S_LABEL),
            ("w50", "log", r"$\log_{10}(W_{50}/{\rm km\,s^{-1}})$"),
            ("v", "lin1e3", r"$cz_{\rm radio}$ [10$^3$ km s$^{-1}$]"),
            ("snr_ps", "log", r"$\log_{10}\,[S/(\sigma\sqrt{\Delta v\,W_{50}})]$"),
            ("drat", "log", r"$\log_{10}(d_{\rm HI}/\theta_{\rm beam})$"),
            ("e3", "lin", r"SoFiA $3\sigma$ size / $\theta_{\rm beam}$"))
    for ax, (k, kind, xl), tg in zip(axes.flat, spec, "abcdef"):
        def tr(a_):
            a_ = np.asarray(a_, float)
            a_ = a_[np.isfinite(a_)]
            if kind == "log":
                return np.log10(a_[a_ > 0])
            return a_ / 1e3 if kind == "lin1e3" else a_
        parts = [tr(rec[k])] + ([tr(inj[k])] if k in inj else [])
        if real is not None:
            parts.append(tr(real[k]))
        allv = np.concatenate([p_ for p_ in parts if p_.size])
        lo, hi = np.percentile(allv, [0.3, 99.7])
        bins = np.linspace(lo, hi, 26)
        if k in inj:
            ax.hist(tr(inj[k]), bins, density=True, histtype="step", color=GREY, lw=0.9,
                    ls="--", label="injected")
        ax.hist(tr(rec[k]), bins, density=True, histtype="stepfilled", color=PALETTE[9],
                alpha=0.35, label="injected and recovered")
        if real is not None:
            rv = tr(real[k])
            if rv.size:
                n, _ = np.histogram(rv, bins)
                dens = n / (rv.size * np.diff(bins))
                ax.hist(rv, bins, density=True, histtype="step", color=INK, lw=1.2,
                        label="released detections")
                ax.errorbar(0.5 * (bins[:-1] + bins[1:]), dens,
                            yerr=np.sqrt(n) / (rv.size * np.diff(bins)), fmt="none",
                            ecolor=INK, elinewidth=0.6, capsize=0)
        ax.set_xlabel(xl); ax.set_yticks([])
        ax.set_ylim(0, 1.22 * ax.get_ylim()[1])           # headroom for the panel tag
        panel_tag(ax, "(%s)" % tg, x=0.88)
    for ax in axes[:, 0]:
        ax.set_ylabel("normalised count")
    hh, ll = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.10 * FONT_SCALE ** 2, hspace=0.42 * FONT_SCALE, top=0.90)
    fig.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=3,
               fontsize=plt.rcParams["legend.fontsize"])
    return save(fig, outdir, "fig13_coverage")


def write_tables(outdir, res, fields_dir):
    """Per-field limits: flux and SNR thresholds, noise, and HI mass limits."""
    rows = []
    for f in ordered(res):
        d = res[f]; ff, fs_ = d["fit_flux"], d["fit_snr"]
        dv = chan_width(res, f); zc = cluster_z(fields_dir, f)
        dc = lum_dist(zc) if np.isfinite(zc) else np.nan
        m50 = 2.356e5 * dc ** 2 * s_point(10 ** fs_["logS50"], d["sigma_med"], dv)
        m90 = 2.356e5 * dc ** 2 * s_point(10 ** fs_["logS90"], d["sigma_med"], dv)
        rows.append((label(f), zc, 1e3 * d["sigma_med"], ff["logS50"], ff["err_logS50"],
                     ff["logS90"], 10 ** ff["logS90"], 10 ** fs_["logS50"], 10 ** fs_["logS90"],
                     np.log10(m50), np.log10(m90), d["n_real"], len(d["t"])))
    tex = ["%% Generated by temba.paperfigs. Mass limits: unresolved source, W50 = %d km/s." % W_REF,
           r"\begin{table*}", r"\centering",
           r"\caption{Completeness limits per field. $\sigma$ is the median channel noise at the "
           r"injected positions; $S_{50}$ and $S_{90}$ come from unbinned logistic fits in integrated "
           r"flux, SNR$_{50}$ and SNR$_{90}$ from the same fits in integrated SNR. $M_{50}$ and $M_{90}$ "
           r"are the H\,{\sc i} masses of an unresolved source with $W_{50}=%d$\,km\,s$^{-1}$ at SNR$_{50}$ and "
           r"SNR$_{90}$ in that noise, at the cluster redshift $z$.}" % W_REF,
           r"\label{tab:limits}",
           r"\begin{tabular}{lcccccccccr}", r"\hline",
           r"Field & $z$ & $\sigma$ & $\log_{10}S_{50}$ & $\log_{10}S_{90}$ & $S_{90}$ & SNR$_{50}$ "
           r"& SNR$_{90}$ & $\log_{10}M_{50}$ & $\log_{10}M_{90}$ & $N_{\rm inj}$ \\",
           r" & & (mJy\,beam$^{-1}$) & \multicolumn{2}{c}{($S$ in Jy\,km\,s$^{-1}$)} & "
           r"(Jy\,km\,s$^{-1}$) & & & \multicolumn{2}{c}{($M$ in M$_\odot$)} & \\", r"\hline"]
    for r in rows:
        tex.append("%s & %.2f & %.3f & $%.2f\\pm%.2f$ & $%.2f$ & %.2f & %.1f & %.1f & %.2f & %.2f & %d \\\\"
                   % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[12]))
    tex += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    p = Path(outdir) / "tables.tex"
    p.write_text("\n".join(tex))
    return str(p)


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------

def write_tex(outdir, res, pars, rr, showcase_rows, journal, schematic="column"):
    lf = np.array([res[f]["fit_flux"]["logS50"] for f in res])
    ls = np.array([res[f]["fit_snr"]["logS50"] for f in res])
    nf, ninj = len(res), sum(len(res[f]["t"]) for f in res)
    smin, smax = 10 ** lf.min(), 10 ** lf.max()
    figure = "figure*" if journal in ("aa", "mnras") else "figure"
    def env(name, cap, wide=True):
        e = figure if wide else "figure"
        width = r"\textwidth" if wide else r"\columnwidth"
        return (f"\\begin{{{e}}}\n  \\centering\n  \\includegraphics[width={width}]{{{name}}}\n"
                f"  \\caption{{{cap}}}\n  \\label{{fig:{name.split('_', 1)[1]}}}\n\\end{{{e}}}\n\n")
    sc = ", ".join(label(f) for f, _, _ in (showcase_rows or []))
    tex = "% Generated by temba.paperfigs. Captions are drafts: edit freely.\n\n"
    tex += env("fig01_pipeline",
               "The TEMBA method. The science cube, with its catalogued sources removed and the "
               "gaps refilled with its own noise, forms the substrate. Mock galaxies drawn on a "
               "Latin hypercube in integrated SNR, linewidth $W_{50}$, size $d/\\theta$ and "
               "inclination $i$ are built as exponential discs with arctan rotation curves, "
               "modelled with 3D-Barolo (iterating until the realised $W_{50}$ matches the "
               "design), injected, and recovered with SoFiA-2 using exactly the settings that "
               "produced the catalogue (dashed). Matching gives capture and fidelity; logistic "
               "fits give the "
               "completeness in flux and SNR. Colours mark the program doing each step; "
               "the bands group the steps into building a signal-free sky, making mock "
               "galaxies and recovering them.", wide=(schematic != "column"))
    tex += env("fig02_completeness",
               f"Completeness of the {nf} fields against (a) integrated flux and (b) integrated "
               f"SNR, from {ninj} injections. Points are binned detection fractions with 68\\% "
               f"Wilson intervals; lines are binomial-likelihood logistic fits. $S_{{50}}$ ranges "
               f"from {smin:.2f} to {smax:.2f}\\,Jy\\,km\\,s$^{{-1}}$ ({10 ** np.ptp(lf):.1f}$\\times$); "
               f"in SNR the spread is {10 ** np.ptp(ls):.2f}$\\times$."
               + ("" if not BAND_CUTS else " " + "; ".join(
                   f"{label(f)} uses only injections above {m:.0f}\\,MHz"
                   for f, m in sorted(BAND_CUTS.items()))
                  + ", excluding an artefact-dominated band edge where detection fails "
                    "whatever the SNR."))
    tex += env("fig03_thresholds",
               "Completeness thresholds per field, ordered by $S_{50}$, in (a) flux and (b) SNR. "
               "Bars span $S_{10}$ to $S_{90}$; points mark $S_{50}$ with bootstrap errors.")
    tex += env("fig04_noise_config",
               "(a) $S_{50}$ (filled) and $S_{90}$ (open) against each field's median channel "
               "noise; dashed lines are proportional to the noise. (b) SNR$_{50}$ against the "
               "SoFiA-2 \\texttt{reliability.minSNR} used for each field; marker shape gives "
               "\\texttt{reliability.threshold}, open markers \\texttt{scfind.threshold}\\,=\\,4.")
    tex += env("fig05_flux_recovery",
               "Flux recovery of detected sources against SNR: (a) capture, the fraction of the "
               "injected flux inside the SoFiA-2 mask; (b) fidelity, the catalogued $f_{\\rm sum}$ "
               "over the injected flux. Thin lines are running medians per field; the thick line "
               "and grey band are the median and 16th--84th percentiles over all fields.")
    tex += env("fig06_size",
               "SoFiA-2 $3\\sigma$ major axis in beams against SNR for detected sources, coloured "
               "by true $d_{\\rm HI}/\\theta_{\\rm beam}$. The dashed curve is the $3\\sigma$ "
               "isophotal diameter of a point source, $\\sqrt{\\ln({\\rm SNR}/3)/\\ln 2}$; the "
               "solid curve is the same scaled by %.2f, the ratio SoFiA-2's ellipse fit gives for "
               "unresolved sources. Unresolved sources follow it whatever their true size."
               % getattr(fig_size, "kfac", 1.0), wide=False)
    tex += env("fig07_position_maps",
               "Detection probability relative to the SNR expectation, $\\Delta P_{\\rm det}$, "
               "across each field (offsets from the pointing centre). SNR already includes the "
               "local noise, so structure here is selection that SNR does not explain; "
               "$\\chi^2$/dof compares each map with zero.")
    tex += env("fig08_radius_freq",
               "$\\Delta P_{\\rm det}$ against (a) distance from the pointing centre and (b) "
               "observed frequency (top axis: radio-convention velocity), for all fields "
               "together (points, binomial errors) and individually (lines). The deepest "
               "deficit, $\\Delta P_{\\rm det}=%.2f$, is at %.0f\\,MHz."
               % getattr(fig_radius_freq, "dip", (np.nan, np.nan)))
    tex += env("fig09_showcase",
               f"The best (left) and worst (right) recovered injections in six representative "
               f"fields ({sc}), chosen by $|\\ln C|+|\\ln \\mathcal{{F}}|$ for capture $C$ and "
               f"fidelity $\\mathcal{{F}}$. Columns: injected and recovered moment-0 on a common "
               f"scale with the noiseless model's $1\\sigma$ (dashed) and $3\\sigma$ (solid) "
               f"isophotes, their difference on a $\\pm3\\sigma$ scale of the moment-0 noise, "
               f"and the spectra. Maps are normalised to the injected model's peak and residuals "
               f"to the moment-0 noise, so the colour bars apply to every row. The circle is the "
               f"beam.")
    tex += env("fig10_per_field",
               "Per-field completeness against integrated flux. Points are recovered over "
               "injected in fixed 0.2\\,dex bins (68\\% Wilson intervals), drawn over the "
               "injected (grey) and recovered (coloured) histograms of the same bins; curves are "
               "unbinned logistic fits with $S_{50}$ (solid) and $S_{90}$ (dashed) marked. Dotted "
               "lines: the flux of an unresolved source with $W_{50}=200$\\,km\\,s$^{-1}$ at "
               "integrated SNR 1 and 3 in the field's median channel noise $\\sigma$, "
               "$S=n\\,\\sigma\\sqrt{\\Delta v\\,W_{50}}$.")
    tex += env("fig14_per_field_snr",
               "As Fig.~\\ref{fig:per_field}, against integrated SNR, "
               "${\\rm SNR}_{\\rm int}=S/[\\sigma\\sqrt{\\Delta v\\,W_{50}}\\,"
               "\\max(1,d_{\\rm HI}/\\theta_{\\rm beam})]$, with $\\sigma$ the local channel "
               "noise at each injected position; bins of 0.125\\,dex.")
    tex += env("fig11_width_size",
               "Dependence of completeness on linewidth and size beyond SNR. (a), (b): SNR$_{50}$ "
               "from logistic fits in bins of $W_{50}$ and of $d_{\\rm HI}/\\theta_{\\rm beam}$, "
               "all fields pooled. (c): $S_{50}$ divided by each field's median channel noise "
               "against $W_{50}$, with the $W_{50}^{1/2}$ scaling of a matched filter and the shape "
               "of the ALFALFA 50\\% limit.")
    tex += env("fig12_mass_limits",
               "H\\,{\\sc i} mass of an unresolved $W_{50}=200$\\,km\\,s$^{-1}$ source at each "
               "field's SNR$_{90}$ and median noise, against luminosity distance; markers (with "
               "bootstrap errors) at each cluster's distance. Grey points: released detections.",
               wide=False)
    tex += env("fig13_coverage",
               "Coverage of the injections against the released detections: all injected sources "
               "(dashed), those recovered (filled; true values) and the released catalogue "
               "(black, measured values, Poisson errors). (d) uses the point-source integrated "
               "SNR, $S/(\\sigma\\sqrt{\\Delta v\\,W_{50}})$, with the local channel noise for "
               "all three. (e) uses the true $d_{\\rm HI}$ for injections and, for released "
               "detections, $d_{\\rm HI}$ from the H\\,{\\sc i} size--mass relation. (f) is "
               "SoFiA-2's measurement for both.")
    tex += env("fig13_coverage",
               "Coverage of the injection design against the released detections in (a) "
               "integrated flux, (b) linewidth, (c) velocity, (d) integrated SNR and (e) "
               "H\\,{\\sc i} size in beams. Dashed: all injected sources; filled: those recovered "
               "(true values); solid: released detections, with SNR computed the same way from "
               "the local noise and $d_{\\rm HI}$ estimated from the H\\,{\\sc i} size--mass relation "
               "(Wang et al. 2016).")
    p = Path(outdir) / "figures.tex"
    p.write_text(tex)
    return str(p)


# ---------------------------------------------------------------------------

FIGS = ("fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07", "fig08", "fig09",
        "fig10", "fig11", "fig12", "fig13", "fig14")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fields", default="fields")
    ap.add_argument("--outdir", default="analysis/paper")
    ap.add_argument("--journal", choices=sorted(JOURNALS), default="mnras")
    ap.add_argument("--fonts", choices=("auto", "latex"), default="auto",
                    help="auto: Times clone + STIX maths, no LaTeX needed; latex: typeset "
                         "with newtxtext/newtxmath, the MNRAS template fonts")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--only", nargs="*", choices=FIGS)
    ap.add_argument("--showcase-fields", nargs="*",
                    help="the six fields for fig09; default spans the S50 range")
    ap.add_argument("--schematic", choices=("column", "flow", "boxes"), default="column",
                    help="fig01 style: top to bottom in one column (default), an "
                         "illustrated left-to-right flow, or ten boxes in two rows")
    ap.add_argument("--min-freq", nargs="*", default=[], metavar="FIELD=MHZ",
                    help="leave out injections below this observed frequency in a field, "
                         "e.g. J0351.1-8212=1310 for an artefact-dominated band edge")
    ap.add_argument("--worst", choices=("score", "capture", "fidelity"), default="score",
                    help="fig09 'worst': furthest from perfect (score), most flux lost "
                         "to the mask (capture), or most noise in the mask (fidelity)")
    a = ap.parse_args()

    W.update(JOURNALS[a.journal])
    use_style(fonts=a.fonts, journal=a.journal)
    want = set(a.only or FIGS)
    made = []
    if "fig01" in want:
        made.append(dict(column=fig_pipeline_column, flow=fig_pipeline_flow,
                         boxes=fig_pipeline_boxes)[a.schematic](a.outdir))
    if want - {"fig01"}:
        fields = collect(a.runs)
        if not fields:
            raise SystemExit(f"No recovery tables under {a.runs}/*/run_*/products/")
        for item in a.min_freq:
            f, mhz = item.split("=")
            if f not in fields:
                raise SystemExit(f"--min-freq: no field {f}")
            t, nr = fields[f]
            keep = F0_MHZ * (1.0 - np.asarray(t["vsys_kms"], float) / C_KMS) >= float(mhz)
            print(f"  {f}: keeping {keep.sum()} of {len(t)} injections above {float(mhz):.0f} MHz")
            fields[f] = (t[keep], nr)
            BAND_CUTS[f] = float(mhz)
        res = fit_fields(fields, a.bootstrap)
        if not res:
            raise SystemExit("no field has enough injections to fit (at least 60 usable); "
                             "inject more: injection.total_per_field or injection.realisations")
        pars = {f: par_settings(a.fields, f) for f in res}
        rr = residuals(res, a.fields)
        rows = None
        if "fig02" in want: made.append(fig_completeness(res, a.outdir))
        if "fig03" in want: made.append(fig_thresholds(res, a.outdir))
        if "fig04" in want: made.append(fig_noise_config(res, pars, a.outdir))
        if "fig05" in want: made.append(fig_flux_recovery(res, a.outdir))
        if "fig06" in want: made.append(fig_size(res, a.outdir))
        if "fig07" in want: made.append(fig_position_maps(rr, a.outdir))
        if "fig08" in want: made.append(fig_radius_freq(rr, a.outdir))
        if "fig09" in want:
            # a thesis page is shorter than a journal page: two fields keep the maps large
            out = fig_showcase(res, a.runs, a.outdir, a.showcase_fields, worst_by=a.worst,
                               n=2 if a.journal == "thesis" else 3)
            if out:
                made.append(out[0]); rows = out[1]
        if "fig10" in want: made.append(fig_per_field(res, a.outdir))
        if "fig11" in want: made.append(fig_width_size(res, a.outdir))
        if "fig12" in want: made.append(fig_mass_limits(res, a.fields, a.outdir))
        if "fig13" in want: made.append(fig_coverage(res, a.fields, a.outdir))
        if "fig14" in want: made.append(fig_per_field(res, a.outdir, axis="snr"))
        if not a.only:
            made.append(write_tex(a.outdir, res, pars, rr, rows, a.journal, a.schematic))
            made.append(write_tables(a.outdir, res, a.fields))
    for m in made:
        if m:
            print(f"  {m}")


if __name__ == "__main__":
    main()
