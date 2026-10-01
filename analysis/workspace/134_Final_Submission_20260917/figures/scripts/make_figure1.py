#!/usr/bin/env python3
"""Draw the study overview from aggregate counts and selection metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, Rectangle

import style as s


def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

BASE = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 180, 144


def render(source, output):
    data = json.loads(source.read_text())
    genes = data["genes"]
    local, public = data["local"], data["public"]
    require(genes == s.GENES, 'Integrity check failed: genes == s.GENES')
    require(local['patients'] == 87 and local['specimens'] == 2 * local['patients'], "Integrity check failed: local['patients'] == 87 and local['specimens'] == 2 * local['patients']")
    require(local['recurrence_patients'] == 82 and local['recurrence_events'] == 14, "Integrity check failed: local['recurrence_patients'] == 82 and local['recurrence_events'] == 14")
    require(sum(public['korean_pairs'].values()) == 373, "Integrity check failed: sum(public['korean_pairs'].values()) == 373")
    require((public['measured_korean_cpgs'], public['source_candidate_cpgs']) == (56, 77), "Integrity check failed: (public['measured_korean_cpgs'], public['source_candidate_cpgs']) == (56, 77)")
    require(data['selection']['recurrence_used_for_selection'] is False, "Integrity check failed: data['selection']['recurrence_used_for_selection'] is False")
    fig = s.figure(WIDTH, HEIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    checks = []

    def txt(text, x, top, bold=False, italic=False, color=s.INK, size=7):
        return ax.text(x, top, text, va="top", ha="left", fontsize=size,
                       fontweight="bold" if bold else "normal",
                       fontstyle="italic" if italic else "normal", color=color,
                       linespacing=1.35)

    def box(x, top, width, height, title, lines, fill):
        ax.add_patch(Rectangle((x, top), width, height, facecolor=fill,
                               edgecolor="#A9B4BA", lw=.55))
        artists = [txt(title, x+2, top+1.6, bold=True)]
        if lines:
            artists.append(txt("\n".join(lines), x+2, top+6.1))
        checks.append((x, top, width, height, artists))

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                    mutation_scale=6, lw=.65, color=s.GRAY,
                                    shrinkA=1, shrinkB=1))

    def panel(letter, title, x, top, width):
        txt(letter, x, top, bold=True, size=9)
        txt(title, x+6, top+.3, bold=True, size=8)
        ax.plot([x, x+width], [top+5, top+5], lw=.5, color="#BDC5C9")

    panel("A", "Candidate selection", 4, 3, 172)
    box(4, 12, 68, 21, "Original screen and literature review",
        [f"{len(data['selection']['datasets'])} public HumanMethylation450 datasets",
         "Prior evidence in colorectal neoplasia"], "#F2F3F4")
    box(82, 12, 94, 21, "Ten fixed candidate genes", [], "#F2F3F4")
    for idx, gene in enumerate(genes):
        txt(gene, 84+(idx % 5)*18, 20+(idx//5)*5, italic=True)
    arrow(72, 22.5, 82, 22.5)
    txt("Original selection details and limitations are reported in Methods and Supplementary Analyses.",
        4, 36, color=s.GRAY)

    panel("B", "Korean pyrosequencing", 4, 44, 83)
    panel("C", "Public CpG analyses", 93, 44, 83)
    blue, green = "#ECF1F5", "#EDF4F0"
    box(4, 53, 83, 15, f"{local['patients']} patients",
        [f"Tumor and adjacent mucosa, {local['specimens']} specimens",
         "Ten gene assays"], blue)
    box(4, 73, 83, 21, "Paired methylation differences",
        ["Mean differences with 95% confidence intervals",
         "Correlations between genes",
         "Principal component analysis"], blue)
    box(4, 99, 83, 17, "Exploratory tissue classification",
        ["Ridge logistic regression",
         "Patient groups retained during cross-validation"], blue)
    box(4, 121, 83, 19, "Exploratory recurrence prediction",
        [f"{local['recurrence_patients']} stage I–III patients, {local['recurrence_events']} events",
         "Clinical, methylation and combined ridge Cox models",
         "Patients receiving palliative treatment excluded"], blue)
    for a,b in [(68,73),(94,99),(116,121)]:
        arrow(45.5,a,45.5,b)

    box(93, 53, 83, 15, f"{public['source_candidate_cpgs']} candidate CpGs",
        ["Fixed list retained for the ten candidate genes",
         "Individual CpGs analyzed separately"], green)
    pairs = public["korean_pairs"]
    box(93, 73, 83, 21, "Three Korean MethylationEPIC cohorts",
        [f"CMCBSN {pairs['CMCBSN']}, SNUH {pairs['SNUH']}, ASAN {pairs['ASAN']} pairs",
         f"{sum(pairs.values())} patient pairs in total",
         f"{public['measured_korean_cpgs']} of {public['source_candidate_cpgs']} candidate CpGs measured"], green)
    box(93, 99, 83, 17, "Tissue and molecular context",
        [f"{public['lesion_cohorts']} public tissue and lesion cohorts",
         "Expression, stromal score and CMS analyses"], green)
    box(93, 121, 83, 19, "Public tissue classification",
        [f"Colonomics, {public['colonomics_pairs']} patient pairs",
         f"GSE119526, {public['gse119526_pairs']} patient pairs",
         "Ridge logistic regression"], green)
    for a,b in [(68,73),(94,99),(116,121)]:
        arrow(134.5,a,134.5,b)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for x, top, width, height, artists in checks:
        bounds = ax.transData.transform([[x,top],[x+width,top+height]])
        for artist in artists:
            bb = artist.get_window_extent(renderer)
            if bb.x0 < bounds[0,0] or bb.x1 > bounds[1,0] or bb.y1 > bounds[0,1] or bb.y0 < bounds[1,1]:
                raise ValueError(f"Text outside its box: {artist.get_text()}")
    return s.save(fig, output, "Figure_1", [source],
                  "Original selection counts condensed. Current patient counts and analysis roles retained. No scientific estimates changed.",
                  {"sample_counts_checked":True,"text_boxes_checked":True,
                   "gene_symbols_italic":True,"historical_count_branches_shown":False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=BASE/"source_data/Figure1_study_flow.json")
    parser.add_argument("--output-dir",type=Path,default=BASE)
    args=parser.parse_args()
    print(json.dumps(render(args.input,args.output_dir),indent=2))


if __name__ == "__main__":
    main()
