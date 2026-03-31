"""
fig_5_1_predicted_vs_true.py
────────────────────────────
Figure 5.1 — Predicted vs true thickness scatter plots for
Random Forest and MLP at full feature set (501 wavelengths),
seed = 42, using the same training pipeline as ml_averaged.py.

Layout: 1 × 2 panels (RF left, MLP right)
Each panel:  scatter (colour = material), perfect-prediction diagonal,
             RMSE annotation, axis labels.

Output: figures/fig_5_1_predicted_vs_true.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.ensemble         import RandomForestRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import mean_squared_error

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', '6_ml_averaged', 'ellipsometry_dataset_raw.csv')
OUT_DIR    = SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

COLORS = {
    'Si':   '#1f77b4',
    'GaAs': '#ff7f0e',
    'GaN':  '#2ca02c',
    'Ge':   '#d62728',
    'InP':  '#9467bd',
}

# ─────────────────────────────────────────────────────────────────
# Load & pivot (replicate ml_averaged.py exactly)
# ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

wavelengths = np.arange(300, 801, 1)   # 501 points

psi_wide = df.pivot_table(
    index=['material', 'thickness_nm'], columns='wavelength_nm', values='psi_deg'
)
delta_wide = df.pivot_table(
    index=['material', 'thickness_nm'], columns='wavelength_nm', values='delta_deg'
)
psi_wide.columns   = [f'psi_{int(c)}'   for c in psi_wide.columns]
delta_wide.columns = [f'delta_{int(c)}' for c in delta_wide.columns]

df_wide = pd.concat([psi_wide, delta_wide], axis=1).reset_index()

feature_cols = [f'psi_{w}' for w in wavelengths] + [f'delta_{w}' for w in wavelengths]
X = df_wide[feature_cols].values
y = df_wide['thickness_nm'].values
materials = df_wide['material'].values

# ─────────────────────────────────────────────────────────────────
# Train / test split (same as ml_averaged.py)
# ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, mat_train, mat_test = train_test_split(
    X, y, materials,
    test_size=0.2,
    random_state=SEED,
    stratify=materials,
)

# ─────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────
print("Training Random Forest (this may take ~60 s)...")
rf  = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"  RF RMSE = {rf_rmse:.4f} nm")

# ─────────────────────────────────────────────────────────────────
# MLP (with StandardScaler)
# ─────────────────────────────────────────────────────────────────
print("Training MLP (this may take ~120 s)...")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=2000,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
)
mlp.fit(X_tr_s, y_train)
mlp_pred = mlp.predict(X_te_s)
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
print(f"  MLP RMSE = {mlp_rmse:.4f} nm")

# ─────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

PANEL_INFO = [
    ('(a)  Random Forest', rf_pred,  rf_rmse),
    ('(b)  MLP',           mlp_pred, mlp_rmse),
]

unique_mats = ['Si', 'GaAs', 'GaN', 'Ge', 'InP']

for ax, (title, y_pred, rmse) in zip(axes, PANEL_INFO):
    for mat in unique_mats:
        idx = mat_test == mat
        ax.scatter(
            y_test[idx], y_pred[idx],
            color=COLORS[mat], s=8, alpha=0.55, linewidths=0,
            label=mat, zorder=3,
        )

    # Perfect-prediction diagonal
    lims = [20, 80]
    ax.plot(lims, lims, 'k--', lw=1.2, zorder=4, label='Perfect prediction')

    ax.set_xlim(20, 80)
    ax.set_ylim(20, 80)
    ax.set_xlabel('True thickness  (nm)', fontsize=11)
    ax.set_ylabel('Predicted thickness  (nm)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.55)
    ax.tick_params(labelsize=10)

    # RMSE annotation
    ax.text(
        0.04, 0.96,
        f'RMSE = {rmse:.3f} nm',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85),
    )

# Legend on the right panel (replace per-axes legends)
for ax in axes:
    ax.get_legend_handles_labels()   # clear auto-legend

handles_mat = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[m],
           markersize=7, label=m)
    for m in unique_mats
]
handles_mat.append(Line2D([0], [0], color='k', lw=1.5, ls='--', label='Perfect'))

axes[1].legend(
    handles=handles_mat,
    loc='lower right',
    fontsize=9,
    frameon=True,
)

fig.suptitle(
    'Predicted vs True Thickness — RF and MLP  (seed = 42, 501 wavelengths)',
    fontsize=13, fontweight='bold',
)

out_path = os.path.join(OUT_DIR, 'fig_5_1_predicted_vs_true.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
