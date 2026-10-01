"""Shared design system for the redesigned manuscript figures (126_Figure_Redesign_20260909).

Print-first, journal-style (Nature/Cell family conventions):
  * Helvetica (Arial fallback), 7 pt body text, 6 pt small annotations, 8 pt bold panel letters.
  * Column widths: single 88 mm, one-and-a-half 120 mm, double 180 mm; maximum height 225 mm.
  * Thin, recessive axes (0.5 pt), outward 2 pt ticks, no top/right spines, no in-figure titles.
  * Okabe-Ito based categorical palette, fixed assignment by entity (never by rank).
  * Diverging maps = two hues + neutral midpoint, always symmetric about zero.
  * Sequential maps = one hue, light -> dark.
Every script must import this module and use only these tokens.
"""
from __future__ import annotations
import json, hashlib, os, datetime, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

MM = 1 / 25.4
W_SINGLE, W_15, W_DOUBLE, H_MAX = 88, 120, 180, 225   # mm

# ---- typography ---------------------------------------------------------
FS = dict(body=7, small=6, tiny=5.5, panel=8, axis=7)
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS["body"], "axes.labelsize": FS["axis"], "axes.titlesize": FS["body"],
    "xtick.labelsize": FS["body"], "ytick.labelsize": FS["body"], "legend.fontsize": FS["small"],
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
    "text.color": "#111111", "axes.labelcolor": "#111111",
    "lines.linewidth": 0.9, "lines.markersize": 4, "patch.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 100, "savefig.dpi": 600, "savefig.facecolor": "white",
    "axes.unicode_minus": True, "mathtext.default": "regular",
})

# ---- color tokens (Okabe-Ito based; validated with the dataviz palette validator) --------
C = dict(
    tumor="#D55E00", normal="#0072B2",                 # tissue pair (vermilion / blue)
    healthy="#009E73", adenoma="#E69F00",              # extra tissue groups when needed
    cohort={"CMCBSN": "#0072B2", "SNUH": "#E69F00", "ASAN": "#009E73"},   # Korean arrays (fixed order)
    local="#0072B2", public="#D55E00",                 # local PSQ vs public array models
    clinical="#6E6E6E", methylation="#0072B2", combined="#D55E00",         # recurrence blocks
    event="#D55E00", noevent="#B8B8B8",
    ink="#111111", ink2="#444444", muted="#8A8A8A", grid="#E6E6E6", rule="#BDBDBD",
    na="#EEEEEE", na_mark="#9E9E9E", sig="#111111",
    box_fill=["#EAF1F8", "#FBEFE6", "#EAF5F0", "#F4F4F4"], box_edge="#8FA6BF",
)

# Diverging: blue - neutral grey-white - vermilion (two hues + neutral midpoint)
CMAP_DIV = LinearSegmentedColormap.from_list("div_bwr", ["#1F5F99", "#6AA0CF", "#F3F3F1", "#E48B5C", "#A63603"])
# Correlation (all values expected positive but scale kept honest): white -> deep red
CMAP_CORR = LinearSegmentedColormap.from_list("corr_seq", ["#FBF8F5", "#F2C9B0", "#E48B5C", "#C2452D", "#7A1A0F"])
# Sequential single hue for unsigned statistics (Kruskal-Wallis H)
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F1F5FA", "#B9CFE7", "#6A9BD1", "#2E6DB3", "#0B3D78"])

GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
COHORTS_KR = ["CMCBSN", "SNUH", "ASAN"]

ROOT = pathlib.Path(os.environ.get("KPU_PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[1]))  # project root (folder containing the numbered analysis folders); override with KPU_PROJECT_ROOT
# Manuscript analysis folder (holds figures/ and results/). The redesign was written against
# 124_Integrated_Revision_20260909; the submission archive ships the same tree as
# 129_Submission_Revised_20260909. Override with KPU_MANUSCRIPT_DIR, otherwise take the first
# candidate that exists under ROOT.
_MANUSCRIPT_CANDIDATES = ("129_Submission_Revised_20260909", "124_Integrated_Revision_20260909")
MANUSCRIPT_DIR = (pathlib.Path(os.environ["KPU_MANUSCRIPT_DIR"]) if os.environ.get("KPU_MANUSCRIPT_DIR")
                  else next((ROOT / name for name in _MANUSCRIPT_CANDIDATES if (ROOT / name).is_dir()),
                            ROOT / _MANUSCRIPT_CANDIDATES[0]))
F124 = MANUSCRIPT_DIR / "figures"   # name kept for the make_fig*.py scripts
R124 = MANUSCRIPT_DIR / "results"
OUT = ROOT / "126_Figure_Redesign_20260909" / "output"

# ---- helpers ------------------------------------------------------------
def figure(width_mm: float, height_mm: float):
    assert width_mm <= W_DOUBLE + 0.01 and height_mm <= H_MAX + 0.01, "exceeds journal page"
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))

def panel_label(fig, x_mm: float, y_mm: float, letter: str):
    """Bold 8 pt panel letter at absolute position (mm from left, mm from top)."""
    w, h = fig.get_size_inches()
    fig.text(x_mm * MM / w, 1 - y_mm * MM / h, letter, fontsize=FS["panel"], fontweight="bold", ha="left", va="top")

def ax_mm(fig, left, top, width, height):
    """Add axes by mm box measured from the top-left corner of the figure."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left * MM / w, 1 - (top + height) * MM / h, width * MM / w, height * MM / h])

def italic(gene: str) -> str:
    """DEPRECATED - mathtext leaves digits upright (ZNF568 -> ZNF*568*).

    No figure script uses this any more: gene symbols are drawn as plain text with
    ``fontstyle="italic"`` / ``style="italic"`` so that letters *and* digits slant.
    Kept only so that older scripts keep importing.
    """
    return f"$\\it{{{gene}}}$"

def despine(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)

def light_grid(ax, axis="x"):
    ax.grid(True, axis=axis, color=C["grid"], linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

def na_cell(ax, x, y, w=1.0, h=1.0):
    """Non-estimable cell: light grey fill with a small grey cross."""
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=C["na"], edgecolor="none", zorder=1))
    ax.text(x, y, "×", ha="center", va="center", fontsize=FS["tiny"], color=C["na_mark"], zorder=2)

def sig_dot(ax, x, y, size=1.6):
    ax.plot(x, y, marker="o", ms=size, mfc=C["sig"], mec="none", zorder=3, linestyle="none")

def gene_bracket(ax, y0, y1, x, label, side="left", lw=0.6):
    """DEPRECATED - thin vertical bracket with a mathtext gene label.

    Uses :func:`italic`, so the digits stay upright. Figures 4-8 draw the bracket
    and a real italic label themselves (``rows77.draw_gene_brackets``).
    """
    ax.plot([x, x], [y0, y1], color=C["ink2"], lw=lw, clip_on=False, solid_capstyle="butt")
    ax.text(x - 0.35 if side == "left" else x + 0.35, (y0 + y1) / 2, italic(label), ha="right" if side == "left" else "left",
            va="center", fontsize=FS["body"], color=C["ink"], clip_on=False)

def colorbar(fig, mappable, left, top, width, height, label, ticks=None, orientation="horizontal"):
    cax = ax_mm(fig, left, top, width, height)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=1.5, labelsize=FS["small"], pad=1)
    if ticks is not None: cb.set_ticks(ticks)
    cb.set_label(label, fontsize=FS["small"], labelpad=2)
    return cb

def sha256(path) -> str:
    h = hashlib.sha256(); h.update(pathlib.Path(path).read_bytes()); return h.hexdigest()

def save(fig, name: str, sources: list, notes: str = ""):
    """Write PDF (vector), SVG (editable text) and 600-dpi PNG plus a JSON manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches=None, pad_inches=0, metadata=None if ext == "png" else {"Title": name})
    w, h = fig.get_size_inches()
    man = {"figure": name, "width_mm": round(w / MM, 1), "height_mm": round(h / MM, 1),
           "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources], "notes": notes,
           "outputs": {ext: sha256(OUT / f"{name}.{ext}") for ext in ("pdf", "svg", "png")}}
    (OUT / f"{name}_manifest.json").write_text(json.dumps(man, indent=1))
    plt.close(fig)
    return man
