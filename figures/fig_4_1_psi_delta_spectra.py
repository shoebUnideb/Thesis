"""
fig_4_1_psi_delta_spectra.py
────────────────────────────
Figure 4.1 — Simulated Ψ(λ) and Δ(λ) spectra for all five semiconductor
materials at a fixed film thickness d = 50 nm, θ₀ = 70°, λ = 300–800 nm.

Each column is one material; top row = Ψ, bottom row = Δ.

Output: figures/fig_4_1_psi_delta_spectra.png
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
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', '1_raw_data', 'ellipsometry_dataset.csv')
OUT_DIR    = SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Load & filter
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
THICKNESS = 50  # nm

MATERIALS = ['Si', 'GaAs', 'GaN', 'Ge', 'InP']
COLORS = {
    'Si':   '#1f77b4',
    'GaAs': '#ff7f0e',
    'GaN':  '#2ca02c',
    'Ge':   '#d62728',
    'InP':  '#9467bd',
}

# ─────────────────────────────────────────────────────────────────
# Plot — 2 rows × 5 columns
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 5,
    figsize=(16, 6),
    constrained_layout=True,
    sharey='row',
)

for col_idx, mat in enumerate(MATERIALS):
    sub = (
        df[(df['material'] == mat) & (df['thickness_nm'] == THICKNESS)]
        .sort_values('wavelength_nm')
    )
    wl  = sub['wavelength_nm'].values
    psi = sub['psi_deg'].values
    dlt = sub['delta_deg'].values

    ax_psi = axes[0, col_idx]
    ax_dlt = axes[1, col_idx]

    ax_psi.plot(wl, psi, color=COLORS[mat], lw=1.7)
    ax_dlt.plot(wl, dlt, color=COLORS[mat], lw=1.7)

    ax_psi.set_title(mat, fontsize=12, fontweight='bold', color=COLORS[mat])
    ax_psi.set_xlim(300, 800)
    ax_dlt.set_xlim(300, 800)

    for ax in (ax_psi, ax_dlt):
        ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.55)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('λ  (nm)', fontsize=9)

    if col_idx == 0:
        ax_psi.set_ylabel('Ψ  (deg)', fontsize=10)
        ax_dlt.set_ylabel('Δ  (deg)', fontsize=10)

# Row labels on the right
for row_idx, label in enumerate(['Ψ(λ)', 'Δ(λ)']):
    axes[row_idx, -1].yaxis.set_label_position('right')
    axes[row_idx, -1].set_ylabel(label, fontsize=11, fontweight='bold', rotation=270,
                                  labelpad=14, va='bottom')

fig.suptitle(
    f'Simulated Ψ(λ) and Δ(λ) Spectra — All Five Materials,  d = {THICKNESS} nm',
    fontsize=13, fontweight='bold',
)

out_path = os.path.join(OUT_DIR, 'fig_4_1_psi_delta_spectra.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
