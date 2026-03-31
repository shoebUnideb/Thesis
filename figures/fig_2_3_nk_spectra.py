"""
fig_2_3_nk_spectra.py
─────────────────────
Figure 2.3 — Refractive index n(λ) and extinction coefficient k(λ)
for all five semiconductor materials (Si, GaAs, GaN, Ge, InP)
over λ = 300–800 nm.

Output: figures/fig_2_3_nk_spectra.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NK_DIR     = os.path.join(SCRIPT_DIR, '..', '1_raw_data', 'semiconductor_nk')
OUT_DIR    = os.path.join(SCRIPT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Materials → file map
# ─────────────────────────────────────────────────────────────────
MATERIALS = {
    'Si':   'si_nk_300_800nm_1nm.csv',
    'GaAs': 'gaas_nk_300_800nm_1nm_final.csv',
    'GaN':  'gan_nk_300_800nm_1nm.csv',
    'Ge':   'ge_nk_300_800nm_1nm.csv',
    'InP':  'inp_nk_300_800nm_1nm.csv',
}

COLORS = {
    'Si':   '#1f77b4',
    'GaAs': '#ff7f0e',
    'GaN':  '#2ca02c',
    'Ge':   '#d62728',
    'InP':  '#9467bd',
}

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
data = {}
for mat, fname in MATERIALS.items():
    fpath = os.path.join(NK_DIR, fname)
    df = pd.read_csv(fpath)
    # Normalise column names (wavelength_nm, n, k)
    df.columns = [c.strip().lower() for c in df.columns]
    data[mat] = df

# ─────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

for mat, df in data.items():
    wl = df['wavelength_nm']
    axes[0].plot(wl, df['n'], color=COLORS[mat], linewidth=1.6, label=mat)
    axes[1].plot(wl, df['k'], color=COLORS[mat], linewidth=1.6, label=mat)

# Axes formatting
for ax, ylabel, title in zip(
    axes,
    ['Refractive index  $n$', 'Extinction coefficient  $k$'],
    ['(a)  $n(\\lambda)$', '(b)  $k(\\lambda)$'],
):
    ax.set_xlabel('Wavelength  (nm)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(300, 800)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    ax.tick_params(labelsize=10)

# Single shared legend below both panels
handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=m) for m in MATERIALS]
fig.legend(
    handles=handles,
    loc='lower center',
    ncol=5,
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.5, -0.08),
)

out_path = os.path.join(OUT_DIR, 'fig_2_3_nk_spectra.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
