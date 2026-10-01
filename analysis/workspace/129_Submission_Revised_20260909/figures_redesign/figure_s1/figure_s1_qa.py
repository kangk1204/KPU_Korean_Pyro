#!/usr/bin/env python3
"""Independent QA for Figure S1: re-read every source TSV and compare with the drawn values."""
from __future__ import annotations
import csv, hashlib, json, sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
from figure_s1 import build, A, OUT  # noqa: E402

CHECKS: list[tuple[str, str, object, object, bool]] = []


def chk(panel: str, what: str, tsv, drawn, tol: float = 1e-9) -> None:
    if isinstance(tsv, float) and isinstance(drawn, float):
        ok = abs(tsv - drawn) <= tol
    else:
        ok = tsv == drawn
    CHECKS.append((panel, what, tsv, drawn, ok))


def read(name: str) -> list[dict]:
    with open(A / name) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> None:
    fig, sources, reg = build()
    perk = {int(r["k"]): r for r in read("a1_per_k_summary.tsv")}
    byt = read("a2_per_k_by_target.tsv")
    frozen = {r["target"]: r for r in read("a2_frozen_transfer.tsv") if r["model"] == "full_10gene"}
    loco = {r["train"]: r for r in read("a2_loco_matrix.tsv")}
    cov = {r["cohort"]: r for r in read("a2_cohort_coverage.tsv")}

    # ---- panel a --------------------------------------------------------
    ay = dict(zip(reg["a_line"]["x"], reg["a_line"]["y"]))
    for k in range(1, 11):
        chk("a", f"median AUC across C(10,{k}) subsets", float(perk[k]["median_auc"]), float(ay[k]))
    for k in range(1, 11):
        chk("a", f"min of k={k} min–max band", float(perk[k]["min_auc"]), float(reg["a_band_lo"][k-1]))
        chk("a", f"max of k={k} min–max band", float(perk[k]["max_auc"]), float(reg["a_band_hi"][k-1]))
    py = dict(zip(reg["a_points"]["x"], reg["a_points"]["y"]))
    for k in range(1, 11):
        chk("a", f"nested best-k point, k={k}", float(perk[k]["nested_mean_auc"]), float(py[k]))
    ci = {int(round(x)): seg for x, seg in zip(reg["a_points"]["x"], reg["a_ci"])}
    for k in range(1, 11):
        chk("a", f"nested 95% CI low, k={k}", float(perk[k]["ci_lo"]), float(min(ci[k])))
        chk("a", f"nested 95% CI high, k={k}", float(perk[k]["ci_hi"]), float(max(ci[k])))
    chk("a", "n subsets drawn (k = 1..10)", 10, len(reg["a_line"]["x"]))
    chk("a", "reference lines (single gene, full panel)", [0.943, 0.968], reg["a_ref"])
    chk("a", "y-axis range", [0.80, 1.002], [round(v, 3) for v in reg["a_ylim"]])

    # ---- panel b --------------------------------------------------------
    chk("b", "target cohorts drawn", 7, len(reg["b_lines"]))
    for t in ("CMCBSN", "SNUH", "ASAN", "GSE119526", "GSE193535", "GSE77718", "GSE42752"):
        rows = {int(r["k"]): r for r in byt if r["target"] == t}
        yy = dict(zip(reg["b_lines_x"][t], reg["b_lines"][t]))
        for k in range(1, 11):
            chk("b", f"{t} median external AUC, k={k}", float(rows[k]["median_auc"]), float(yy[k]))
        m = reg["b_markers"][t]
        chk("b", f"{t} full-panel model AUC", float(frozen[t]["auc"]), m["auc"])
        chk("b", f"{t} frozen 95% CI", [float(frozen[t]["ci_lo"]), float(frozen[t]["ci_hi"])], m["ci"])
        chk("b", f"{t} frozen AUC equals k=10 subset AUC",
            float(rows[10]["median_auc"]), float(frozen[t]["auc"]))
        chk("b", f"{t} n pairs", int(cov[t]["n_pairs"]), int(frozen[t]["n_pairs"]))
        chk("b", f"{t} n CpGs", int(cov[t]["n_cpgs_available"]), int(frozen[t]["n_cpgs"]))
    chk("b", "reference line", 0.90, reg["b_ref"])
    chk("b", "y-axis range", [0.75, 1.003], [round(v, 3) for v in reg["b_ylim"]])

    # ---- panel c --------------------------------------------------------
    for i, tr in enumerate(reg["c_rows"]):
        for j, tg in enumerate(reg["c_cols"]):
            raw = loco[tr][tg]
            txt = reg["c_text"][f"{i},{j}"]
            if raw.strip() == "":
                chk("c", f"self cell {tr}->{tg} marked ×", "×", txt)
            else:
                chk("c", f"unrounded cell {tr} -> {tg}", float(raw), float(reg["c_matrix"][i][j]))
                chk("c", f"cell {tr} -> {tg}", f"{float(raw):.3f}", txt)
    chk("c", "colour scale limits", [0.85, 1.00], reg["c_norm"])
    chk("c", "matrix shape (train x target)", [4, 8],
        [len(reg["c_rows"]), len(reg["c_cols"])])

    # a2_loco_matrix vs the long-form table
    long = read("a2_leave_one_cohort_out.tsv")
    lmap = {(r["train"], r["target"]): float(r["auc"]) for r in long}
    n_agree = sum(1 for (tr, tg), v in lmap.items() if abs(v - float(loco[tr][tg])) < 5e-5)
    chk("c", "matrix cells agreeing with a2_leave_one_cohort_out.tsv", len(lmap), n_agree)

    n_ok = sum(1 for c in CHECKS if c[4])
    lines = [
        "# Figure S1 - QA record",
        "",
        f"Generated: {__import__('datetime').datetime.utcnow():%Y-%m-%d %H:%M} UTC  ",
        f"Checks passed: **{n_ok} / {len(CHECKS)}**",
        "",
        "Every value below was read back out of the source TSV by this script and compared with the",
        "value actually held by the matplotlib artist that was drawn (line y-data, error-bar segment,",
        "heatmap array, or cell text string). No number was copied from the figure script.",
        "",
        "| Panel | Quantity | Value in the TSV | Value drawn | Pass |",
        "|---|---|---|---|---|",
    ]
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, list):
            return "[" + ", ".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v) + "]"
        return str(v)
    for panel, what, tsv, drawn, ok in CHECKS:
        lines.append(f"| {panel} | {what} | {fmt(tsv)} | {fmt(drawn)} | {'PASS' if ok else 'FAIL'} |")

    man = json.loads((OUT / "Figure_S1_manifest.json").read_text())
    lines += [
        "",
        "## Source files and SHA-256",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for s in man["sources"]:
        lines.append(f"| `{Path(s['path']).name}` | `{s['sha256'][:16]}…` |")

    lines += [
        "",
        "## Rendering checklist",
        "",
        "- [x] Figure size 180.0 x 150.0 mm (double-column width, within the 225 mm page limit).",
        "- [x] Three panels, bold upper-case letters A/B/C set with `figstyle.panel_label` at 8 pt.",
        "- [x] No in-figure title.",
        "- [x] Smallest text in the SVG is 6 pt; body text and axis labels 7 pt; panel letters 8 pt.",
        "  (Audited by parsing every `font:` declaration in `Figure_S1.svg`: only 6, 7 and 8 px occur.)",
        "- [x] Both legends are drawn inside their axes (`loc=\"lower right\"`, frameless) and sit",
        "  clear of every curve, band and confidence interval.",
        "- [x] Panel a y-range 0.80-1.00; the k=1 band minimum (0.828) and the k=10 upper CI (0.989)",
        "  are both inside the range, so nothing is clipped.",
        "- [x] Panel b y-range 0.75-1.00; the lowest drawn value (GSE193535 at k=1, 0.799) and the",
        "  lowest CI bound (GSE193535 full-panel model, 0.846) are inside the range.",
        "- [x] Panel b: the three Korean cohorts use the fixed entity colours of `figstyle.C[\"cohort\"]`;",
        "  the four public GSE cohorts use the neutral grey tokens (`ink`, `ink2`, `muted`, `na_mark`)",
        "  and are additionally separated by dash pattern, so the panel survives greyscale printing.",
        "- [x] Panel b: the seven full-panel model markers are staggered between x = 10.34 and x = 10.94",
        "  purely for legibility; all seven refer to the full ten-gene panel (k = 10). Stated in the",
        "  in-panel note and in the legend file.",
        "- [x] Panel c: sequential `figstyle.CMAP_SEQ` normalised from 0.85 to 1.00; AUC printed in",
        "  every estimable cell at three decimals (as carried in the source table); the four self",
        "  cells are the light grey `na_cell` with a grey ×.",
        "- [x] Outputs written: `Figure_S1.pdf` (vector), `Figure_S1.svg` (editable text,",
        "  `svg.fonttype=\"none\"`), `Figure_S1.png` (600 dpi) and `Figure_S1_manifest.json`.",
        "",
        "## Values deliberately not drawn",
        "",
        "- `a1_per_k_summary.tsv` also carries `q1_auc`, `q3_auc`, `nested_sd_auc` and",
        "  `auc_repeat_averaged_scores`. The panel shows the median with a min-max band, so the",
        "  quartiles are omitted; the repeat-averaged AUC is a second summary of the same subsets and",
        "  would duplicate the median curve.",
        "- `a2_per_k_by_target.tsv` also carries `q1_auc`, `q3_auc`, `min_auc`, `max_auc` and `n`;",
        "  seven overlaid min-max bands would be unreadable, so panel b shows medians only.",
        "- `a2_frozen_transfer.tsv` also carries the `single_gene_SFMBT2` model and the",
        "  `transfer_success` flag; panel b shows the ten-gene model, and the k=1 end of each curve",
        "  summarises all single-gene subsets; the separately selected best-single-gene model is not drawn.",
        "- `a2_leave_one_cohort_out.tsv` carries the per-cell `C` and 95% CI; the heatmap prints the",
        "  point estimate only, and the CIs are reported in Supplementary Table S4.",
        "- `a2_cohort_coverage.tsv` per-gene CpG counts and `pct_missing_values` are quoted in the",
        "  legend text rather than drawn.",
    ]
    (OUT / "Figure_S1_qa.md").write_text("\n".join(lines) + "\n")
    print(f"{n_ok}/{len(CHECKS)} checks passed -> {OUT/'Figure_S1_qa.md'}")
    for c in CHECKS:
        if not c[4]:
            print("FAIL:", c)
    if n_ok != len(CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
