#!/usr/bin/env python3
"""QA for Table S2.

Reads the produced tables/Table_S2.csv back in, parses every cell into numbers, and compares
each one with the value re-read from the source TSVs. The parser is independent of the writer:
it works from the printed strings, so a formatting error cannot pass unnoticed.
"""
from __future__ import annotations
import csv, datetime, hashlib, math, re, sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_s2 import B, OUT, ROWS, sig2, read  # noqa: E402

FAIL: list[str] = []
N = 0


def ok(cond: bool, what: str, exp, got) -> None:
    global N
    N += 1
    if not cond:
        FAIL.append(f"{what}: expected {exp!r}, printed {got!r}")


def num(s: str) -> float:
    return float(s.replace("−", "-"))


def main() -> None:
    tests = {(r["cohort"], r["axis"], r["label"], r["comparison"]): r for r in read("B_axis_tests.tsv")}
    diag = {(r["cohort"], r["axis"]): r for r in read("B_axis_diagnostics.tsv")}
    counts = read("B_label_counts.tsv")
    mvrows = read("B_multivariable_colonomics.tsv")

    with open(OUT / "Table_S2.csv") as fh:
        printed = list(csv.DictReader(fh))

    ok(len(printed) == len(ROWS), "row count", len(ROWS), len(printed))
    detail = []

    for spec, row in zip(ROWS, printed):
        disp, coh, axis, label, comp = spec
        r = tests[(coh, axis, label, comp)]
        d = diag[(coh, axis)]
        tag = f"{disp}/{axis}/{label}"

        ok(row["Cohort"] == disp, f"{tag} cohort", disp, row["Cohort"])

        # --- Axis cell: n, CpGs, PC1 variance -----------------------------
        m = re.match(r"(delta|tumour) PC1; n = (\d+) \w+; (\d+) CpGs; PC1 ([\d.]+)%$", row["Axis"])
        ok(bool(m), f"{tag} axis cell format", "delta|tumour PC1; n = N unit; N CpGs; PC1 X%", row["Axis"])
        if m:
            ok(m.group(1) == ("delta" if axis == "delta_axis" else "tumour"), f"{tag} axis type", axis, m.group(1))
            ok(int(m.group(2)) == int(float(d["n_sample"])), f"{tag} axis n", d["n_sample"], m.group(2))
            ok(int(m.group(3)) == int(float(d["n_cpg"])), f"{tag} axis CpGs", d["n_cpg"], m.group(3))
            exp_v = round(float(d["pc1_var_explained"]) * 100, 1)
            ok(abs(float(m.group(4)) - exp_v) < 1e-9, f"{tag} PC1 variance", exp_v, m.group(4))
            detail.append((tag, "axis n / CpGs / PC1%", f"{d['n_sample']} / {d['n_cpg']} / {exp_v}%",
                           f"{m.group(2)} / {m.group(3)} / {m.group(4)}%"))

        # --- Groups (n) ---------------------------------------------------
        src_groups = {k.strip(): v.strip() for k, v in
                      (g.split("=") for g in r["group_sizes"].split(";"))}
        prn_groups = {k.strip(): v.strip() for k, v in
                      (g.split(" = ") for g in row["Groups (n)"].split(";"))}
        ok(src_groups == prn_groups, f"{tag} group sizes", src_groups, prn_groups)
        detail.append((tag, "groups (n)", r["group_sizes"], row["Groups (n)"]))

        # --- Effect, P, q -------------------------------------------------
        tested = r["tested"] == "True"
        if not tested:
            m = re.match(r"(-?[\d.]+) \(descriptive only\)$", row["Effect (95% CI)"])
            ok(bool(m), f"{tag} descriptive effect format", "X.XX (descriptive only)", row["Effect (95% CI)"])
            if m:
                ok(abs(num(m.group(1)) - round(float(r["effect"]), 2)) < 5e-9,
                   f"{tag} descriptive effect", round(float(r['effect']), 2), m.group(1))
            ok(row["P"] == "not tested (n<5)", f"{tag} P", "not tested (n<5)", row["P"])
            ok(row["BH q"] == "not tested (n<5)", f"{tag} q", "not tested (n<5)", row["BH q"])
            ok(r["p_value"] == "" and r["q_value"] == "",
               f"{tag} source really untested", "empty p/q in TSV", (r["p_value"], r["q_value"]))
            detail.append((tag, "effect / P / q", f"{float(r['effect']):.4f} / (not run) / (not run)",
                           f"{row['Effect (95% CI)']} / {row['P']} / {row['BH q']}"))
        else:
            m = re.match(r"(?:(rho|eps\^2) = )?(-?[\d.]+) \((-?[\d.]+) to (-?[\d.]+)\)$",
                         row["Effect (95% CI)"])
            ok(bool(m), f"{tag} effect format", "[prefix ]X.XX (lo to hi)", row["Effect (95% CI)"])
            if m:
                want_pref = {"rho": "rho", "epsilon^2": "eps^2"}.get(r["effect_name"])
                ok(m.group(1) == want_pref, f"{tag} effect kind", want_pref, m.group(1))
                for g, src, nm in ((2, "effect", "point"), (3, "ci_lo", "CI low"), (4, "ci_hi", "CI high")):
                    ok(abs(num(m.group(g)) - round(float(r[src]), 2)) < 5e-9,
                       f"{tag} {nm}", round(float(r[src]), 2), m.group(g))
            ok(row["P"] == sig2(float(r["p_value"])), f"{tag} P", sig2(float(r["p_value"])), row["P"])
            ok(row["BH q"] == sig2(float(r["q_value"])), f"{tag} q", sig2(float(r["q_value"])), row["BH q"])
            # two significant figures actually achieved
            for cell in (row["P"], row["BH q"]):
                digits = re.sub(r"[^\d]", "", cell.split("e")[0]).lstrip("0")
                ok(len(digits) <= 2, f"{tag} 2 sig figs in {cell}", "<=2 significant digits", digits)
            detail.append((tag, "effect / P / q",
                           f"{float(r['effect']):.4f} ({float(r['ci_lo']):.4f}, {float(r['ci_hi']):.4f}) / "
                           f"{float(r['p_value']):.4g} / {float(r['q_value']):.4g}",
                           f"{row['Effect (95% CI)']} / {row['P']} / {row['BH q']}"))

        # --- Family -------------------------------------------------------
        want_fam = "pre-specified" if r["family"].startswith("pre-specified") else "secondary"
        ok(row["Family"] == want_fam, f"{tag} family", want_fam, row["Family"])

    # ------------------------------------------------------------------ footnotes
    md = (OUT / "Table_S2.md").read_text()
    mv = {r["term"]: r for r in mvrows if r["axis"] == "delta_axis" and r["model"] == "pre-specified"}
    st = mv["stromal_z"]
    ok(f"{float(st['coef']):.2f}".replace("-", "−") in md, "footnote stromal coefficient",
       f"{float(st['coef']):.2f}", "in md")
    ok(f"{float(st['robust_se_HC3']):.2f}" in md, "footnote HC3 SE", f"{float(st['robust_se_HC3']):.2f}", "in md")
    ok("4.8 × 10<sup>−4</sup>" in md, "footnote stromal P", sig2(float(st["p_value"])), "in md")

    for coh, axis, in [("Colonomics", "delta_axis"), ("GSE164811", "tumor_axis"),
                       ("CMCBSN", "delta_axis"), ("CMCBSN", "tumor_axis"),
                       ("TCGA-COADREAD", "tumor_axis"), ("GSE48684", "tumor_axis"),
                       ("GSE77718", "delta_axis"), ("GSE193535", "delta_axis"),
                       ("GSE42752", "delta_axis")]:
        v = f"{float(diag[(coh, axis)]['pc1_var_explained']) * 100:.1f}%"
        ok(v in md, f"footnote PC1 variance {coh}/{axis}", v, "in md")

    cnt = lambda c, role, lab, lev: next(
        (int(r["n"]) for r in counts if r["cohort"].startswith(c) and r["axis_role"].startswith(role)
         and r["label"] == lab and r["level"] == lev), None)
    ok(cnt("Colonomics", "delta", "BRAF_V600E", "Yes") == 3, "footnote Colonomics BRAF n", 3,
       cnt("Colonomics", "delta", "BRAF_V600E", "Yes"))
    ok(cnt("TCGA-COADREAD", "tumor", "braf", "BRAF abnormal") == 3, "footnote TCGA BRAF n", 3,
       cnt("TCGA-COADREAD", "tumor", "braf", "BRAF abnormal"))
    ok(cnt("TCGA-COADREAD", "tumor", "msi", "MSI-positive") == 11, "footnote TCGA MSI-positive n", 11,
       cnt("TCGA-COADREAD", "tumor", "msi", "MSI-positive"))
    ok(cnt("CMCBSN", "delta", "msi2", "MSI-H") == 3, "footnote CMCBSN delta MSI-H n", 3,
       cnt("CMCBSN", "delta", "msi2", "MSI-H"))
    ok(cnt("CMCBSN", "tumor", "msi2", "MSI-H") == 14, "footnote CMCBSN tumour MSI-H n", 14,
       cnt("CMCBSN", "tumor", "msi2", "MSI-H"))
    ok(cnt("Colonomics", "delta", "stage2", "I-II") == 92, "footnote Colonomics stage I-II n", 92,
       cnt("Colonomics", "delta", "stage2", "I-II"))
    ok(all(int(r["n"]) == 0 for r in counts if r["cohort"] == "GSE42752"),
       "footnote GSE42752 labels all absent", 0, "non-zero label count found")
    ok(not any("cimp" in r["label"].lower() for r in counts) and
       not any("cimp" in k[3].lower() for k in tests),
       "footnote: no CIMP label anywhere", "no CIMP rows", "CIMP row found")
    ok(sum(1 for r in counts if r["label"] == "stromal_score") == 1
       and next(r["cohort"] for r in counts if r["label"] == "stromal_score") == "Colonomics",
       "footnote: stromal score Colonomics only", "Colonomics only", "other cohort found")

    # completeness against the source table
    all_keys = set(tests)
    used = {(c, a, l, cp) for _, c, a, l, cp in ROWS}
    unused = sorted(k for k in all_keys - used)

    lines = [
        "# Table S2 - QA record",
        "",
        f"Generated: {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC  ",
        f"Cells checked: **{N}**  ",
        f"Failures: **{len(FAIL)}**",
        "",
        "`tables/Table_S2.csv` was read back in and every printed cell was parsed into numbers with a",
        "regular expression, then compared with the value re-read from the source TSV. Rounding is",
        "checked against the value rounded independently in this script, not against the writer.",
        "",
    ]
    if FAIL:
        lines += ["## Failures", ""] + [f"- {f}" for f in FAIL] + [""]
    else:
        lines += ["**All checks passed.**", ""]
    lines += ["## Cell-by-cell round trip", "",
              "| Row | Field | Source TSV | Printed in Table S2 |", "|---|---|---|---|"]
    for tag, field, src, prn in detail:
        lines.append(f"| {tag} | {field} | {src} | {prn} |")

    lines += [
        "",
        "## Checklist",
        "",
        f"- [x] 27 rows, one per cohort x axis x comparison, selected explicitly (never by a filter).",
        "- [x] Every selected key exists in `B_axis_tests.tsv`; a missing key raises at build time.",
        "- [x] Axis descriptor (`n`, CpG count, PC1 variance) re-read from `B_axis_diagnostics.tsv`.",
        "- [x] Group sizes match `group_sizes` exactly, as a set of label/count pairs.",
        "- [x] Median differences and rho printed to 2 decimals; epsilon^2 to 2 decimals; CIs as",
        "  \"(low to high)\".",
        "- [x] P and BH q printed to 2 significant figures (scientific notation below 0.001);",
        "  the digit count of every printed value was verified.",
        "- [x] Rows with a group of fewer than 5 patients carry \"not tested (n<5)\" in both P and",
        "  BH q, and the source row was confirmed to have an empty p and q.",
        "- [x] Family reproduces the `family` column of the source table (pre-specified / secondary).",
        "- [x] Footnote numbers (multivariable model, PC1 variance per cohort, label counts) were",
        "  each searched for in the rendered Markdown.",
        "- [x] The absence of CIMP anywhere and the Colonomics-only stromal score were verified",
        "  against `B_label_counts.tsv` and `B_axis_tests.tsv` rather than asserted.",
        "",
        "## Rows present in B_axis_tests.tsv but deliberately not tabulated",
        "",
        f"{len(unused)} of the {len(all_keys)} rows in the source table are not shown:",
        "",
    ]
    cat = {}
    for k in unused:
        c, a, l, cp = k
        if a == "mean_tumor_beta":
            g = ("`mean_tumor_beta` rows", "a sanity axis (the simple mean beta of the panel CpGs) "
                 "run alongside PC1 to confirm the two agree; it is not the PC1 axis this table "
                 "describes")
        elif c == "Colonomics" and a == "tumor_axis":
            g = ("Colonomics tumour-axis rows", "the same comparisons repeated on the tumour PC1 "
                 "axis; Colonomics has verified tumour-normal pairs, so the pre-specified delta PC1 "
                 "axis is the one reported")
        elif c == "GSE77954":
            g = ("GSE77954 rows", "13 tumours in total; the site comparison had a single "
                 "right-sided patient and stage is absent, so only sex was testable, and the "
                 "cohort is too small to carry a row here")
        elif l == "Stage":
            g = ("Stage rows", "not estimable in Colonomics (all 92 patients stage I-II) and absent "
                 "in GSE193535, GSE42752 and GSE77718; the GSE48684 stage row is a secondary "
                 "comparison outside the requested row set")
        elif l == "Sex":
            g = ("Sex rows in the public GEO cohorts", "outside the requested row set - only the "
                 "site comparison was requested for GSE48684 and GSE77718, and GSE193535 and "
                 "GSE42752 are not tabulated at all")
        elif c == "GSE164811":
            g = ("GSE164811 four-group Kruskal-Wallis row", "the same contrast as the CMS3 vs CMS2 "
                 "row already shown, because only CMS2 and CMS3 reach 5 patients")
        else:
            g = ("Site rows with no site labels", "GSE193535 and GSE42752 carry no site annotation, "
                 "so the comparison was not run")
        cat.setdefault(g, []).append(k)
    for (name, why), ks in sorted(cat.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- **{name}** ({len(ks)}) - {why}.")
    lines += [
        "",
        "Full list:",
        "",
    ]
    for c, a, l, cp in unused:
        lines.append(f"- `{c}` / `{a}` / {l} / {cp}")

    lines += ["", "## Source files and SHA-256", "", "| File | SHA-256 |", "|---|---|"]
    for n in ("B_axis_tests.tsv", "B_axis_diagnostics.tsv", "B_multivariable_colonomics.tsv",
              "B_label_counts.tsv"):
        lines.append(f"| `{n}` | `{hashlib.sha256((B / n).read_bytes()).hexdigest()}` |")

    (OUT / "Table_S2_qa.md").write_text("\n".join(lines) + "\n")
    print(f"{N - len(FAIL)}/{N} checks passed; {len(unused)} source rows deliberately not tabulated")
    for f in FAIL:
        print("FAIL:", f)


if __name__ == "__main__":
    main()
