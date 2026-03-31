"""
fig_2_4_noise_types.py
──────────────────────
Figure 2.4 — Effect of five noise types on a representative Ψ(λ) spectrum.
Uses Si, d = 50 nm as the reference sample.

Noise types shown:
  gaussian · relative · uniform · bias · mixed

Clean spectrum drawn as a thick grey reference line.
Each noise type drawn on a separate panel in a 2×3 grid
(bottom-right panel left blank or used for a legend).

Output: figures/fig_2_4_noise_types.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOISE_DIR  = os.path.join(SCRIPT_DIR, '..', '2_simulation', 'noise_datasets')
OUT_DIR    = SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Load combined noise file (all 5 noise columns in one CSV)
# ─────────────────────────────────────────────────────────────────
combined_path = os.path.join(NOISE_DIR, 'dataset_combined_noise.csv')
df = pd.read_csv(combined_path)

# Filter Si, d=50 nm
mask = (df['material'] == 'Si') & (df['thickness_nm'] == 50)
sub  = df[mask].sort_values('wavelength_nm').reset_index(drop=True)

wl    = sub['wavelength_nm'].values
clean = sub['psi'].values

NOISE_COLS = {
    'Gaussian':  'psi_with_gaussian_noise',
    'Relative':  'psi_with_relative_noise',
    'Uniform':   'psi_with_uniform_noise',
    'Bias':      'psi_with_bias',
    'Mixed':     'psi_with_mixed_noise',
}

COLORS = {
    'Gaussian': '#1f77b4',
    'Relative': '#ff7f0e',
    'Uniform':  '#2ca02c',
    'Bias':     '#d62728',
    'Mixed':    '#9467bd',
}

# ─────────────────────────────────────────────────────────────────
# Plot — 2 × 3 grid; 5 panels used, last panel = global legend
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
axes_flat = axes.flatten()

for idx, (label, col) in enumerate(NOISE_COLS.items()):
    ax = axes_flat[idx]
    ax.plot(wl, clean,        color='0.55', lw=1.4, ls='--', label='Clean', zorder=2)
    ax.plot(wl, sub[col].values, color=COLORS[label], lw=1.1, alpha=0.85,
            label=f'{label} noise', zorder=3)
    ax.set_title(f'({chr(97+idx)})  {label} noise', fontsize=11, fontweight='bold')
    ax.set_xlabel('Wavelength  (nm)', fontsize=10)
    ax.set_ylabel('Ψ  (deg)', fontsize=10)
    ax.set_xlim(300, 800)
    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.55)
    ax.tick_params(labelsize=9)

# Last panel → combined legend
ax_leg = axes_flat[5]
ax_leg.axis('off')
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='0.55', lw=2, ls='--', label='Clean'),
] + [
    Line2D([0], [0], color=COLORS[lbl], lw=2, label=f'{lbl} noise')
    for lbl in NOISE_COLS
]
ax_leg.legend(
    handles=legend_handles,
    loc='center',
    fontsize=11,
    frameon=True,
    title='Legend',
    title_fontsize=11,
)

fig.suptitle(
    'Effect of Five Noise Types on Ψ(λ)  —  Si, d = 50 nm',
    fontsize=13, fontweight='bold',
)

out_path = os.path.join(OUT_DIR, 'fig_2_4_noise_types.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
