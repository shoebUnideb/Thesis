"""
fig_2_7_literature_comparison.py
──────────────────────────────────
Figure 2.7 — Literature comparison bar chart.

Horizontal bars: one entry per prior work + this-work entries.
Metric: RMSE or equivalent (nm).  Where a paper reports a non-RMSE metric
(percentage accuracy, Å per layer) the bar is shown with a hatched fill and
an annotation explaining the source value.

Sources:
  [11] Liu et al. (2021)  Light: Sci. Appl.   Si+SiN, CNN          ~0.8 nm  (estimated from reported accuracy)
  [12] Arunachalam (2022) JVST A              ZnO ALD, k-NN        ~2.0 nm  (estimated; primary metric: % channels removed)
  [13] Arunachalam (2022) arXiv               TiO₂, RF             ~1.2 nm  (derived: 88.76% within ±1.5 nm → σ≈1.25 nm)
  [15] Barkhordari (2024) Sci. Reports        ZnTiO₃, best ML      ~3.5 nm  (estimated from R² on real experimental data)
  [17] Kwak et al. (2021) Light: Adv. Manuf.  3D-NAND/Si₃N₄, ANN  0.16 nm  (directly stated: 1.6 Å per layer)
  [18] Urban & Barton (2024) JVST A           ITO/Si, ANN          ~2.0 nm  (estimated; primary metric: fit residuals reported)
  This work — RF          (simulation, multi-material)             1.86 nm  (mean ± 5 seeds)
  This work — MLP         (simulation, multi-material)             1.56 nm  (mean ± 5 seeds)

NOTE: Values marked with * are estimated or derived from the reported metric
because the papers do not quote RMSE directly.  Comparison should account
for differences in material, thickness range and experimental vs simulated data.

Output: figures/fig_2_7_literature_comparison.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
# (label, rmse_nm, is_estimated, material_note, color_group)
# color_group: 0 = prior work estimated, 1 = prior work derived,
#              2 = prior work exact, 3 = this work
entries = [
    # label                                          rmse   est?   annotation
    ("Liu et al. 2021 [11]\nSi/SiN, CNN",           0.80,  True,  "~0.8 nm*"),
    ("Arunachalam et al. 2022 [12]\nZnO ALD, k-NN", 2.00,  True,  "~2.0 nm*"),
    ("Arunachalam et al. 2022 [13]\nTiO₂, RF",      1.20,  True,  "~1.2 nm*\n(88.76% within ±1.5 nm)"),
    ("Barkhordari et al. 2024 [15]\nZnTiO₃, SVM/RF",3.50,  True,  "~3.5 nm*\n(real exp. data)"),
    ("Kwak et al. 2021 [17]\n3D-NAND, ANN",         0.16,  False, "0.16 nm\n(1.6 Å per layer)"),
    ("Urban & Barton 2024 [18]\nITO/Si, ANN",        2.00,  True,  "~2.0 nm*"),
    ("This work — RF\n(multi-material, sim.)",        1.86,  False, "1.86 nm\n(mean, 5 seeds)"),
    ("This work — MLP\n(multi-material, sim.)",       1.56,  False, "1.56 nm\n(mean, 5 seeds)"),
]

labels  = [e[0] for e in entries]
values  = [e[1] for e in entries]
is_est  = [e[2] for e in entries]
annots  = [e[3] for e in entries]

n = len(entries)
y_pos = np.arange(n)

# Colours
PRIOR_EXACT = '#5b9bd5'     # solid blue — exact / directly stated
PRIOR_EST   = '#aec6e8'     # pale blue  — estimated / derived
THIS_RF     = '#e07b39'     # orange     — RF
THIS_MLP    = '#2ca02c'     # green      — MLP

colors = []
hatches = []
for i, (lbl, val, est, ann) in enumerate(entries):
    if 'This work' in lbl and 'RF' in lbl:
        colors.append(THIS_RF);  hatches.append('')
    elif 'This work' in lbl and 'MLP' in lbl:
        colors.append(THIS_MLP); hatches.append('')
    elif est:
        colors.append(PRIOR_EST); hatches.append('//')
    else:
        colors.append(PRIOR_EXACT); hatches.append('')

# ─────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

bars = ax.barh(
    y_pos, values,
    color=colors,
    hatch=hatches,
    edgecolor='white',
    linewidth=0.8,
    height=0.62,
    zorder=3,
)

# Value annotations at end of each bar
for i, (val, ann) in enumerate(zip(values, annots)):
    ax.text(
        val + 0.06, i,
        ann,
        va='center', ha='left',
        fontsize=8.5,
        color='#333333',
        linespacing=1.35,
    )

# Vertical reference line at this-work MLP result
ax.axvline(1.56, color=THIS_MLP, lw=1.2, ls='--', alpha=0.6, zorder=2)
ax.axvline(1.86, color=THIS_RF,  lw=1.2, ls='--', alpha=0.6, zorder=2)

# Horizontal separator before "This work" entries
ax.axhline(5.5, color='0.4', lw=0.8, ls='--', zorder=2)
ax.text(4.2, 5.58, 'This work', fontsize=9, color='0.4', va='bottom', ha='right')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel('RMSE or equivalent  (nm)', fontsize=11)
ax.set_xlim(0, 5.2)
ax.set_title(
    'Figure 2.7 — Literature Comparison: Reported Thickness Prediction Accuracy',
    fontsize=12, fontweight='bold', pad=10,
)
ax.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.6, zorder=1)
ax.set_axisbelow(True)
ax.invert_yaxis()   # top entry first

# Legend
legend_handles = [
    mpatches.Patch(facecolor=PRIOR_EXACT, edgecolor='white', label='Prior work — directly reported'),
    mpatches.Patch(facecolor=PRIOR_EST,   edgecolor='white', hatch='//', label='Prior work — estimated/derived*'),
    mpatches.Patch(facecolor=THIS_RF,     edgecolor='white', label='This work — Random Forest'),
    mpatches.Patch(facecolor=THIS_MLP,    edgecolor='white', label='This work — MLP'),
]
ax.legend(
    handles=legend_handles,
    loc='lower right',
    fontsize=9,
    frameon=True,
    framealpha=0.92,
)

# Footnote
fig.text(
    0.01, -0.02,
    '* Estimated or derived from the reported metric (see text). '
    'Direct numerical comparison is limited by differences in material, '
    'thickness range, and experimental vs simulated data.',
    fontsize=7.5, color='0.45', style='italic',
    wrap=True,
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_2_7_literature_comparison.png')
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_PATH}")
