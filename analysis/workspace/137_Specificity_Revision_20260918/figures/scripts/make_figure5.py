#!/usr/bin/env python3
"""Figure 5 - how specific to the panel is the coordination?

A  The candidate panel against 5,000 matched reference sets given identical treatment,
   under three conditions on one shared scale: unconditioned, conditioned on tumor
   content, conditioned on the noncandidate common score.
B  What each conditioning axis removes, with the numbers.
C  The candidate correlation inside de novo CIMP strata.

Every value is read from the aggregate tables; nothing is recomputed here.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs

RES = Path('results')
COH = ['ASAN', 'SNUH']
CC = {'ASAN': fs.C['cohort']['ASAN'], 'SNUH': fs.C['cohort']['SNUH']}
MK = {'ASAN': 'o', 'SNUH': 's'}          # shape carries identity as well as hue
GREY = '#C9CFD4'

summary = pd.read_csv(RES / 'C_reference_residual_summary.tsv', sep='\t').set_index('cohort')
perset = pd.read_csv(RES / 'C_reference_residual_per_set.tsv', sep='\t')
purity = pd.read_csv(RES / 'A_purity_reference_per_set.tsv', sep='\t')
cond = pd.concat([pd.read_csv(RES / 'A_conditioning_comparison.tsv', sep='\t'),
                  pd.read_csv(RES / 'A_cimp_index_conditioning.tsv', sep='\t')],
                 ignore_index=True)
strat = pd.read_csv(RES / 'B_cimp_stratified_correlation.tsv', sep='\t')

CONDITIONS = [
    ('unconditioned', 'Unconditioned',
     lambda c: perset.loc[perset.cohort == c, 'raw_balanced_median'].to_numpy(),
     lambda c: summary.loc[c, 'candidate_raw_balanced_median']),
    ('purity', 'Tumor content removed',
     lambda c: purity.loc[purity.cohort == c, 'conditioned_on_purity_hypo'].to_numpy(),
     lambda c: cond[(cond.cohort == c) & (cond.axis == 'purity_hypo')].conditioned_balanced_median.iloc[0]),
    ('common', 'Common score removed',
     lambda c: perset.loc[perset.cohort == c, 'conditioned_leave_own_genes_out'].to_numpy(),
     lambda c: summary.loc[c, 'candidate_conditioned_balanced_median']),
]

XLO, XHI = -0.07, 0.80
fig = fs.figure(180, 190)

# ------------------------------------------------------------------ A ----
fs.panel_label(fig, 6, 9, 'A')
fs.text(fig, 'The candidate panel against 5,000 matched reference sets, same treatment, one scale',
        14, 9.4, fontsize=fs.FS['body'])
axA = fs.ax_mm(fig, 64, 18, 82, 62)
axA.set_xlim(XLO, XHI); axA.set_autoscale_on(False)

rows = [(c, key, lab, getv, getc) for c in COH for key, lab, getv, getc in CONDITIONS]
ys = np.arange(len(rows))[::-1].astype(float)
span = XHI - XLO
for i, ((c, key, lab, getv, getc), y) in enumerate(zip(rows, ys)):
    if i % 3 == 0:
        fs.group_band(axA, y - 2.5, y + .60, XLO, XHI, fs.C['band'])
    v, cand = getv(c), getc(c)
    hist, edges = np.histogram(v, bins=90, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    h = hist / hist.max() * .42
    axA.fill_between(centres, y, y + h, color=GREY, lw=0, zorder=2)
    axA.plot([v.min(), v.max()], [y - .15] * 2, color=fs.C['muted'], lw=.5,
             solid_capstyle='butt', zorder=3)
    lo, hi = np.quantile(v, [.025, .975])
    axA.plot([lo, hi], [y - .15] * 2, color=fs.C['ink2'], lw=1.6,
             solid_capstyle='butt', zorder=4)
    axA.plot(np.median(v), y - .15, '|', ms=4.2, mew=.9, color=fs.C['ink'], zorder=5)
    axA.plot(cand, y + .19, MK[c], ms=4.6, mfc=CC[c], mec='white', mew=.8, zorder=6)
    axA.text(cand, y + .56, f'{cand:.3f}', ha='center', va='bottom',
             fontsize=fs.FS['small'], color=CC[c], zorder=7)
    n_at = int((v >= cand).sum())
    txt = 'none of 5,000 reach it' if n_at == 0 else f'{n_at:,} of 5,000 reach it'
    axA.text(XHI + .02 * span, y + .02, txt, ha='left', va='center',
             fontsize=fs.FS['tiny'], color=fs.C['muted'], clip_on=False)
    axA.text(XLO - .02 * span, y + .02, lab, ha='right', va='center',
             fontsize=fs.FS['small'], color=fs.C['ink2'], clip_on=False)

axA.set_yticks([]); axA.set_ylim(-.75, len(rows) - .30)
axA.set_xticks([0, .2, .4, .6, .8])
axA.set_xlabel('Gene-pair-balanced median Spearman \u03c1', fontsize=fs.FS['axis'])
fs.despine(axA, left=False); fs.light_grid(axA)
axA.axvline(0, color=fs.C['rule'], lw=.5, zorder=1)
for k, c in enumerate(COH):
    top, bot = ys[k * 3] + .65, ys[k * 3 + 2] - .60
    xb = XLO - .46 * span
    axA.plot([xb] * 2, [bot, top], color=CC[c], lw=1.6,
             clip_on=False, solid_capstyle='butt', zorder=7)
    axA.text(xb - .028 * span, (top + bot) / 2, f'{c}  ({int(summary.loc[c, "pairs"])} pairs)',
             rotation=90, ha='center', va='center',
             fontsize=fs.FS['small'], color=CC[c], clip_on=False)

leg = [Line2D([], [], color=GREY, lw=5, label='5,000 reference sets'),
       Line2D([], [], color=fs.C['ink2'], lw=1.6, label='central 95%; thin line, full range'),
       Line2D([], [], marker='o', color='none', mfc=fs.C['muted'], mec='white', mew=.7,
              ms=4.4, label='candidate panel')]
axA.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -0.16),
           ncol=3, fontsize=fs.FS['tiny'], handlelength=1.6, columnspacing=1.8,
           borderpad=0, handletextpad=.5)

# ------------------------------------------------------------------ B ----
fs.panel_label(fig, 6, 96, 'B')
fs.text(fig, 'What each axis removes from the candidate correlation', 14, 96.4,
        fontsize=fs.FS['body'])
axB = fs.ax_mm(fig, 62, 106, 114, 44)
AX = [('purity_hypo', 'Tumor content', 'open-sea methylation loss'),
      ('cimp_index_continuous', 'CIMP index', 'five-gene Weisenberger panel'),
      ('island_gain', 'Global island gain', 'genome-wide CpG islands'),
      ('common_score', 'Noncandidate common score', '151 genes outside the panel'),
      ('common_score_plus_purity_hypo', 'Common score and tumor content', 'both together')]
yb = np.arange(len(AX))[::-1].astype(float)
axB.set_xlim(-0.02, 0.76); axB.set_autoscale_on(False)
for i, ((key, name, sub), y) in enumerate(zip(AX, yb)):
    if i % 2 == 0:
        fs.group_band(axB, y - .48, y + .48, -0.02, 0.76, fs.C['band'])
    for k, c in enumerate(COH):
        r = cond[(cond.cohort == c) & (cond.axis == key)].iloc[0]
        off = .19 if k == 0 else -.19
        axB.plot([r.conditioned_balanced_median, r.raw_balanced_median], [y + off] * 2,
                 color=CC[c], lw=1.0, solid_capstyle='butt', zorder=3)
        axB.plot(r.raw_balanced_median, y + off, MK[c], ms=3.0, mfc='white',
                 mec=CC[c], mew=.8, zorder=4)
        axB.plot(r.conditioned_balanced_median, y + off, MK[c], ms=3.8, mfc=CC[c],
                 mec='white', mew=.5, zorder=5)
        axB.text(r.conditioned_balanced_median - .013, y + off, f'{r.conditioned_balanced_median:.3f}',
                 ha='right', va='center', fontsize=fs.FS['tiny'], color=CC[c])
    axB.text(-0.032, y + .13, name, ha='right', va='center', fontsize=fs.FS['small'],
             color=fs.C['ink'], clip_on=False)
    axB.text(-0.032, y - .21, sub, ha='right', va='center', fontsize=fs.FS['tiny'],
             color=fs.C['muted'], clip_on=False)
axB.set_yticks([]); axB.set_ylim(-.62, len(AX) - .38)
axB.set_xticks([0, .2, .4, .6])
axB.set_xlabel('Gene-pair-balanced median Spearman ρ', fontsize=fs.FS['axis'])
fs.despine(axB, left=False); fs.light_grid(axB)
fs.text(fig, 'open mark, unconditioned      filled mark, conditioned', 63, 103.2,
        fontsize=fs.FS['tiny'], color=fs.C['muted'], ha='left')
for k, c in enumerate(COH):
    axB.plot([], [], MK[c], color=CC[c], ms=3.4, label=c)
axB.legend(loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2,
           fontsize=fs.FS['tiny'], handlelength=1.0, borderpad=0, handletextpad=.4,
           columnspacing=1.2)

# ------------------------------------------------------------------ C ----
fs.panel_label(fig, 6, 160, 'C')
fs.text(fig, 'Candidate correlation inside de novo CIMP strata', 14, 160.4, fontsize=fs.FS['body'])
axC = fs.ax_mm(fig, 62, 167, 114, 13)
sel = strat[strat.stratification == 'tumor_beta_0.30_ge3']
order = [('ASAN', 'high'), ('ASAN', 'low'), ('SNUH', 'high'), ('SNUH', 'low')]
yc = np.arange(len(order))[::-1].astype(float)
axC.set_xlim(-0.02, 0.76); axC.set_autoscale_on(False)
for (c, part), y in zip(order, yc):
    r = sel[(sel.cohort == c) & (sel.stratum == part)].iloc[0]
    allv = cond[(cond.cohort == c) & (cond.axis == 'common_score')].raw_balanced_median.iloc[0]
    axC.plot([0, .76], [y, y], color=fs.C['grid'], lw=.4, zorder=0)
    axC.plot(allv, y, '|', ms=6, mew=.9, color=fs.C['muted'], zorder=3)
    if np.isfinite(r.balanced_median):
        axC.plot(r.balanced_median, y, MK[c], ms=3.8, mfc=CC[c], mec='white', mew=.5, zorder=4)
        axC.text(r.balanced_median - .013, y, f'{r.balanced_median:.3f}', ha='right',
                 va='center', fontsize=fs.FS['tiny'], color=CC[c])
    else:
        axC.text(.02, y, 'not estimated, fewer than 20 patients', va='center', ha='left',
                 fontsize=fs.FS['tiny'], color=fs.C['muted'])
    axC.text(-0.032, y, f'{c}  CIMP-{part}  (n {int(r.patients)})', ha='right', va='center',
             fontsize=fs.FS['small'], color=fs.C['ink'], clip_on=False)
axC.set_yticks([]); axC.set_ylim(-.6, len(order) - .4)
axC.set_xticks([0, .2, .4, .6])
axC.set_xlabel('Gene-pair-balanced median Spearman ρ', fontsize=fs.FS['axis'])
fs.despine(axC, left=False); fs.light_grid(axC)
axC.text(0.755, len(order) - .5, 'vertical tick, all patients in that cohort',
         ha='right', va='center', fontsize=fs.FS['tiny'], color=fs.C['muted'])

fs.save(fig, 'Figure_5',
        sources=[RES / 'C_reference_residual_summary.tsv', RES / 'C_reference_residual_per_set.tsv',
                 RES / 'A_purity_reference_per_set.tsv', RES / 'A_conditioning_comparison.tsv',
                 RES / 'B_cimp_stratified_correlation.tsv'],
        notes='Panel A shares one scale across the three conditions so the collapse is directly comparable.',
        out=Path('output'))
fs.preview('Figure_5', out=Path('output'))
print('done')
