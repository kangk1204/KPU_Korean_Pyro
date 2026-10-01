"""Portable, deterministic figure exports at the specified printed size."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
INK = "#20272C"
BLUE = "#276187"
GRAY = "#5A646B"
LIGHT = "#CED3D7"
GOLD = "#997325"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2, "ytick.major.size": 2, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "crc-132-figures",
    "figure.dpi": 100, "savefig.dpi": 600, "savefig.facecolor": "white",
})
DIV = LinearSegmentedColormap.from_list("paired_difference", ["#1F5F99", "#6AA0CF", "#F3F3F1", "#E48B5C", "#A63603"])
CORR = LinearSegmentedColormap.from_list("positive_correlation", ["#FBF8F5", "#F2C9B0", "#E48B5C", "#C2452D", "#7A1A0F"])

def figure(width=180, height=100):
    return plt.figure(figsize=(width * MM, height * MM))

def axes(fig, left, top, width, height):
    w, h = fig.get_size_inches() / MM
    return fig.add_axes([left/w, 1-(top+height)/h, width/w, height/h])

def label(fig, letter, left, top):
    w, h = fig.get_size_inches() / MM
    fig.text(left/w, 1-top/h, letter, ha="left", va="top", fontsize=9, fontweight="bold")

def text(fig, value, left, top, **kwargs):
    w, h = fig.get_size_inches() / MM
    return fig.text(left/w, 1-top/h, value, ha=kwargs.pop("ha", "left"), va=kwargs.pop("va", "top"), **kwargs)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(fig, out, name, inputs, notes, details=None):
    """Fixed metadata and SVG identifiers allow identical repeat exports."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    from matplotlib.text import Text
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        if item.get_fontsize() < 7:
            raise ValueError(f"Text below 7 pt: {item.get_text()}")
        box = item.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1+.5 or box.y1 > fig.bbox.y1+.5:
            outside.append(item.get_text())
    if outside:
        raise ValueError(f"Text outside figure: {outside}")
    fig.savefig(out/f"{name}.svg", bbox_inches=None, pad_inches=0,
                metadata={"Title": name, "Date": None})
    fig.savefig(out/f"{name}.pdf", bbox_inches=None, pad_inches=0,
                metadata={"Title": name, "CreationDate": None, "ModDate": None})
    fig.savefig(out/f"{name}.png", dpi=600, bbox_inches=None, pad_inches=0,
                metadata={"Title": name, "Software": "matplotlib"})
    w, h = fig.get_size_inches()/MM
    record = {"figure": name, "width_mm": round(float(w), 6), "height_mm": round(float(h), 6),
              "minimum_font_pt": 7, "png_dpi": 600, "backend": "matplotlib",
              "matplotlib_version": matplotlib.__version__,
              "inputs": [{"file": Path(p).name, "sha256": sha256(p)} for p in inputs],
              "notes": notes, "details": details or {},
              "outputs": {ext: sha256(out/f"{name}.{ext}") for ext in ("svg", "pdf", "png")}}
    (out/f"{name}_manifest.json").write_text(json.dumps(record, indent=2)+"\n")
    plt.close(fig)
    return record
