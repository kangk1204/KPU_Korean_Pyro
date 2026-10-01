#!/usr/bin/env python3
"""Table S2 - associations of the shared methylation axis with clinical and molecular labels.

Built entirely from the read-only Part-B result tables of
127_Exploratory_PanelSize_Coordination_20260909/B_coordination_axis/results/.
Writes tables/Table_S2.csv (plain ASCII) and tables/Table_S2.md (typeset).
"""
from __future__ import annotations
import csv, hashlib, json, math, datetime
import os
from pathlib import Path

B = Path(str(Path(os.environ.get("KPU_PROJECT_ROOT", Path(__file__).resolve().parents[2])) / "127_Exploratory_PanelSize_Coordination_20260909/B_coordination_axis/results"))
OUT = Path(str(Path(os.environ.get("KPU_PROJECT_ROOT", Path(__file__).resolve().parents[2])) / "128_Exploratory_Integration_20260909/tables"))

# (display cohort, tsv cohort, axis, label, comparison)  -- selection is explicit, never a filter
ROWS = [
    ("Colonomics", "Colonomics", "delta_axis", "BRAF V600E", "BRAF V600E mutant vs wild-type"),
    ("Colonomics", "Colonomics", "delta_axis", "KRAS", "KRAS mutated vs wild-type"),
    ("Colonomics", "Colonomics", "delta_axis", "CMS", "CMS four groups [CMS1,CMS2,CMS3,CMS4]"),
    ("Colonomics", "Colonomics", "delta_axis", "CMS1", "CMS1 vs CMS2-4"),
    ("Colonomics", "Colonomics", "delta_axis", "Site", "right- vs left-sided"),
    ("Colonomics", "Colonomics", "delta_axis", "Sex", "male vs female"),
    ("Colonomics", "Colonomics", "delta_axis", "Age", "age (years)"),
    ("Colonomics", "Colonomics", "delta_axis", "Stromal score", "ESTIMATE-style stromal score"),
    ("MATCH (GSE164811)", "GSE164811", "tumor_axis", "CMS", "CMS3 vs CMS2"),
    ("MATCH (GSE164811)", "GSE164811", "tumor_axis", "Site", "right- vs left-sided"),
    ("MATCH (GSE164811)", "GSE164811", "tumor_axis", "Sex", "male vs female"),
    ("CMCBSN", "CMCBSN", "delta_axis", "CMS", "CMS four groups [CMS1,CMS2,CMS3,CMS4]"),
    ("CMCBSN", "CMCBSN", "delta_axis", "CMS1", "CMS1 vs CMS2-4"),
    ("CMCBSN", "CMCBSN", "delta_axis", "Site", "right- vs left-sided"),
    ("CMCBSN", "CMCBSN", "delta_axis", "MSI", "MSI-H vs MSS/MSI-L"),
    ("CMCBSN", "CMCBSN", "tumor_axis", "CMS", "CMS four groups [CMS1,CMS2,CMS3,CMS4]"),
    ("CMCBSN", "CMCBSN", "tumor_axis", "CMS1", "CMS1 vs CMS2-4"),
    ("CMCBSN", "CMCBSN", "tumor_axis", "Site", "right- vs left-sided"),
    ("CMCBSN", "CMCBSN", "tumor_axis", "MSI", "MSI-H vs MSS/MSI-L"),
    ("TCGA-COADREAD", "TCGA-COADREAD", "tumor_axis", "MSI", "MSI-positive vs MSI-negative"),
    ("TCGA-COADREAD", "TCGA-COADREAD", "tumor_axis", "BRAF", "BRAF abnormal vs normal"),
    ("TCGA-COADREAD", "TCGA-COADREAD", "tumor_axis", "KRAS", "KRAS mutated vs wild-type"),
    ("TCGA-COADREAD", "TCGA-COADREAD", "tumor_axis", "Site", "right- vs left-sided"),
    ("TCGA-COADREAD", "TCGA-COADREAD", "tumor_axis", "Sex", "male vs female"),
    ("TCGA-COADREAD", "TCGA-COADREAD", "tumor_axis", "Age", "age (years)"),
    ("GSE48684", "GSE48684", "tumor_axis", "Site", "right- vs left-sided"),
    ("GSE77718", "GSE77718", "delta_axis", "Site", "right- vs left-sided"),
]

AXIS_NAME = {"delta_axis": "delta PC1", "tumor_axis": "tumour PC1"}
AXIS_UNIT = {"delta_axis": "pairs", "tumor_axis": "tumours"}
FAMILY = {"pre-specified (BH within cohort)": "pre-specified",
          "secondary/exploratory (BH within cohort)": "secondary"}
COMPARISON = {
    "CMS four groups [CMS1,CMS2,CMS3,CMS4]": "CMS (four groups)",
    "CMS1 vs CMS2-4": "CMS1 vs CMS2-4",
    "CMS3 vs CMS2": "CMS3 vs CMS2",
    "right- vs left-sided": "right- vs left-sided tumour",
    "male vs female": "male vs female",
    "age (years)": "age at diagnosis (years)",
    "ESTIMATE-style stromal score": "ESTIMATE-style stromal score",
    "BRAF V600E mutant vs wild-type": "BRAF V600E mutant vs wild-type",
    "BRAF abnormal vs normal": "BRAF abnormal vs normal",
    "KRAS mutated vs wild-type": "KRAS mutated vs wild-type",
    "MSI-H vs MSS/MSI-L": "MSI-H vs MSS/MSI-L",
    "MSI-positive vs MSI-negative": "MSI-positive vs MSI-negative",
}
NOT_TESTED = "not tested (n<5)"
TERM = {"braf": "*BRAF*", "cms1": "CMS1", "right": "right-sided location",
        "age_z": "age", "male": "male sex"}


def sig2(x: float) -> str:
    """Two significant figures; scientific notation below 0.001."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    d = 1 - e
    r = round(x, d)
    if abs(x) < 1e-3:
        return f"{r / 10 ** e:.1f}e{e:+03d}"
    return f"{r:.{max(d, 0)}f}"


def read(name):
    with open(B / name) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build():
    tests = read("B_axis_tests.tsv")
    diag = {(r["cohort"], r["axis"]): r for r in read("B_axis_diagnostics.tsv")}
    idx = {(r["cohort"], r["axis"], r["label"], r["comparison"]): r for r in tests}

    out = []
    for disp, coh, axis, label, comp in ROWS:
        key = (coh, axis, label, comp)
        if key not in idx:
            raise KeyError(f"row not found in B_axis_tests.tsv: {key}")
        r = idx[key]
        d = diag[(coh, axis)]
        axis_cell = (f"{AXIS_NAME[axis]}; n = {int(float(d['n_sample']))} {AXIS_UNIT[axis]}; "
                     f"{int(float(d['n_cpg']))} CpGs; PC1 {float(d['pc1_var_explained']) * 100:.1f}%")
        groups = "; ".join(g.strip().replace("=", " = ") for g in r["group_sizes"].split(";"))

        tested = r["tested"] == "True"
        eff = r["effect"]
        if not tested:
            effect = f"{float(eff):.2f} (descriptive only)" if eff else "not estimable"
            p_cell = q_cell = NOT_TESTED
        else:
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            if r["effect_name"] == "rho":
                effect = f"rho = {float(eff):.2f} ({lo:.2f} to {hi:.2f})"
            elif r["effect_name"] == "epsilon^2":
                effect = f"eps^2 = {float(eff):.2f} ({lo:.2f} to {hi:.2f})"
            else:
                effect = f"{float(eff):.2f} ({lo:.2f} to {hi:.2f})"
            p_cell = sig2(float(r["p_value"]))
            q_cell = sig2(float(r["q_value"]))
        out.append({
            "Cohort": disp,
            "Axis": axis_cell,
            "Comparison": COMPARISON[comp],
            "Groups (n)": groups,
            "Effect (95% CI)": effect,
            "P": p_cell,
            "BH q": q_cell,
            "Family": FAMILY[r["family"]],
        })
    return out, diag


def md_cell(s: str) -> str:
    import re
    MINUS = "\u2212"
    s = (s.replace("rho = ", "*\u03c1* = ").replace("eps^2 = ", "*\u03b5*\u00b2 = ")
          .replace("delta PC1", "\u0394 PC1").replace("tumour PC1", "Tumour PC1")
          .replace("not tested (n<5)", "not tested (*n* < 5)")
          .replace("n = ", "*n* = "))
    s = re.sub(r"\b(BRAF|KRAS)\b", r"*\1*", s)

    def _sci(m):
        sign = MINUS if m.group(2) == "-" else ""
        return m.group(1) + " \u00d7 10<sup>" + sign + m.group(3) + "</sup>"

    s = re.sub(r"(\d\.\d)e([+-])0?(\d+)", _sci, s)
    s = re.sub(r"(?<![\w>])-(?=\d)", MINUS, s)
    return s


def main():
    rows, diag = build()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())

    with open(OUT / "Table_S2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    mv = {r["term"]: r for r in read("B_multivariable_colonomics.tsv")
          if r["cohort"] == "Colonomics" and r["axis"] == "delta_axis" and r["model"] == "pre-specified"}
    st = mv["stromal_z"]
    sens = {r["term"]: r for r in read("B_multivariable_colonomics.tsv")
            if r["cohort"] == "Colonomics" and r["axis"] == "delta_axis"
            and r["model"].startswith("sensitivity")}["stromal_z"]
    orho = [abs(float(r["spearman_vs_mean_tumor_beta"])) for r in read("B_axis_diagnostics.tsv")
            if r["spearman_vs_mean_tumor_beta"]]
    var = {c: f"{float(diag[(c, a)]['pc1_var_explained']) * 100:.1f}%"
           for (c, a) in diag if diag[(c, a)]["pc1_var_explained"]}
    dv = lambda c, a: f"{float(diag[(c, a)]['pc1_var_explained']) * 100:.1f}%"

    lines = [
        "# Table S2. Associations of the shared methylation axis with clinical and molecular labels",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(md_cell(r[c]) for c in cols) + " |")

    lines += [
        "",
        "### Footnotes",
        "",
        "**Axis definition and orientation.** The shared methylation axis is the first principal "
        "component (PC1) of the panel CpGs after standardising each CpG to zero mean and unit "
        "variance within the cohort. Two versions are used. The **Δ PC1 (delta) axis** is computed on "
        "the tumour-minus-normal difference in β within each verified tumour–normal pair and is the "
        "pre-specified coordination axis; it exists only in cohorts with at least ten verified pairs. "
        "The **tumour PC1 axis** is computed on tumour specimens alone and is used where paired "
        "normals are unavailable. In every cohort the sign of PC1 is fixed so that a higher score "
        "means more methylation, by requiring a positive Spearman correlation between the axis score "
        f"and the mean β of the panel CpGs in the tumour (Spearman ρ {min(orho):.2f}–{max(orho):.3f} "
        f"across the {len(orho)} computed axes, `B_axis_diagnostics.tsv`). Axis scores are in PC1 units of the standardised CpG matrix, so a "
        "median difference is reported in those axis units and is comparable within, but not across, "
        "cohorts.",
        "",
        "**PC1 variance explained, per cohort and axis** (`B_axis_diagnostics.tsv`): "
        f"Colonomics Δ PC1 {dv('Colonomics','delta_axis')} (tumour PC1 {dv('Colonomics','tumor_axis')}); "
        f"MATCH/GSE164811 tumour PC1 {dv('GSE164811','tumor_axis')}; "
        f"CMCBSN Δ PC1 {dv('CMCBSN','delta_axis')} and tumour PC1 {dv('CMCBSN','tumor_axis')}; "
        f"TCGA-COADREAD tumour PC1 {dv('TCGA-COADREAD','tumor_axis')}; "
        f"GSE48684 tumour PC1 {dv('GSE48684','tumor_axis')}; "
        f"GSE77718 Δ PC1 {dv('GSE77718','delta_axis')}. "
        f"For completeness the remaining cohorts are GSE193535 Δ PC1 {dv('GSE193535','delta_axis')} "
        f"and GSE42752 Δ PC1 {dv('GSE42752','delta_axis')}.",
        "",
        "**Multivariable model (Colonomics, \u0394 PC1 axis).** In the pre-specified linear model "
        "with *BRAF*, CMS1, right-sided location, stromal score (per SD), age (per SD) and male sex "
        "as terms, the stromal score is the only term associated with the axis: "
        + md_cell(f"{float(st['coef']):.2f} axis units per SD (HC3 robust SE "
                  f"{float(st['robust_se_HC3']):.2f}; 95% CI {float(st['ci_lo']):.2f} to "
                  f"{float(st['ci_hi']):.2f}; P = {sig2(float(st['p_value']))}; n = {st['n']}, "
                  f"R2 = {float(st['r_squared']):.2f}).").replace("P =", "*P* =").replace("R2 =", "*R*\u00b2 =")
        + " All other terms were not significant (*P* \u2265 0.11: "
        + ", ".join(f"{TERM[t]} (*P* = {sig2(float(mv[t]['p_value']))})"
                    for t in ("braf", "cms1", "right", "age_z", "male"))
        + "). A sensitivity model dropping the near-collinear *BRAF* and CMS1 terms gives a very "
        + md_cell(f"similar stromal estimate ({float(sens['coef']):.2f} per SD; HC3 SE "
                  f"{float(sens['robust_se_HC3']):.2f}; P = {sig2(float(sens['p_value']))}; "
                  f"n = {sens['n']}).").replace("P =", "*P* ="),
        "",
        "**Label availability.** CIMP status is not available for any cohort in the local data "
        "holdings, so no CIMP comparison could be attempted. The ESTIMATE-style stromal score exists "
        "only for Colonomics. *BRAF* status is present for 3 mutant patients in Colonomics and 3 "
        "abnormal patients in TCGA-COADREAD, so both *BRAF* comparisons are descriptive only. MSI was "
        "found in the CMCBSN clinical table after the analysis plan was written and is reported "
        "post-hoc (3 MSI-H patients on the Δ PC1 axis, 14 on the tumour PC1 axis); TCGA-COADREAD has "
        "11 MSI-positive of 110 patients with a call. Stage could not be tested in Colonomics (all "
        "92 patients stage I–II) and site, stage, sex and age are entirely missing in GSE42752, so "
        "these comparisons are absent from the table. Full label counts, including missingness, are "
        "in `B_label_counts.tsv`.",
        "",
        "**Statistics.** Two-group comparisons use the Wilcoxon rank-sum test with a median "
        "difference and a 2,000-resample percentile bootstrap CI; four-group comparisons use the "
        "Kruskal–Wallis test with ε² and a bootstrap CI; continuous labels use the Spearman "
        "correlation with a bootstrap CI. Benjamini–Hochberg q values are computed within cohort and "
        "family. Any group with fewer than 5 patients is reported descriptively and not tested; such "
        "rows carry \"not tested (*n* < 5)\".",
        "",
        "**Analysis status.** This is a pre-specified exploratory analysis performed after the "
        "primary analyses; it was not part of the confirmatory plan and no result here is used to "
        "support a primary claim. The MSI comparisons in CMCBSN were not in the pre-specified family "
        "and are labelled secondary, as are the CMCBSN site comparisons, which were added from the "
        "CMCBSN RNA sample table after the plan was fixed.",
        "",
        "**Source files.**",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for n in ("B_axis_tests.tsv", "B_axis_diagnostics.tsv", "B_multivariable_colonomics.tsv",
              "B_label_counts.tsv"):
        lines.append(f"| `{n}` | `{hashlib.sha256((B / n).read_bytes()).hexdigest()[:16]}…` |")
    lines.append("")
    lines.append(f"Generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC from "
                 "`127_Exploratory_PanelSize_Coordination_20260909/B_coordination_axis/results/`.")

    (OUT / "Table_S2.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> Table_S2.csv / Table_S2.md")


if __name__ == "__main__":
    main()
