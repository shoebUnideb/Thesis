import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────
OUT = "ml_dense_study"
os.makedirs(f"{OUT}/plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PIVOT DATASET
# Long format (152,805 rows) → Wide format (305 rows × 1002 features)
# ─────────────────────────────────────────────────────────────────
print("Loading and pivoting dataset...")
df = pd.read_csv("ellipsometry_dataset.csv")

wavelengths = np.arange(300, 801, 1)   # 501 wavelength points
all_thicknesses = np.arange(20, 81, 1) # 61 thickness values

psi_wide = df.pivot_table(
    index=['material', 'thickness_nm'],
    columns='wavelength_nm',
    values='psi_deg'
)
delta_wide = df.pivot_table(
    index=['material', 'thickness_nm'],
    columns='wavelength_nm',
    values='delta_deg'
)

psi_wide.columns   = [f'psi_{int(c)}'   for c in psi_wide.columns]
delta_wide.columns = [f'delta_{int(c)}' for c in delta_wide.columns]

df_wide = pd.concat([psi_wide, delta_wide], axis=1).reset_index()

psi_cols_full   = [f'psi_{w}'   for w in wavelengths]
delta_cols_full = [f'delta_{w}' for w in wavelengths]
all_feature_cols = psi_cols_full + delta_cols_full

print(f"Wide dataset: {df_wide.shape[0]} samples x {df_wide.shape[1]-2} features")

# Save feature matrix
df_wide.to_csv(f"{OUT}/ml_feature_matrix.csv", index=False)
print(f"Saved: {OUT}/ml_feature_matrix.csv")

# Copy raw data
shutil.copy("ellipsometry_dataset.csv", f"{OUT}/ellipsometry_dataset_raw.csv")
print(f"Saved: {OUT}/ellipsometry_dataset_raw.csv")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — MODEL HELPERS
# ─────────────────────────────────────────────────────────────────
def get_models():
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
    }

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    """Train model (StandardScaler applied only for MLP) and return RMSE, MAE, R2, MSE."""
    if name == 'MLP':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    return round(rmse, 4), round(mae, 4), round(r2, 6), round(mse, 4)

# ─────────────────────────────────────────────────────────────────
# STUDY 1 — WAVELENGTH REDUCTION
# Dense sweep: 501 → 50, step = 5
# Sequence: 501, 495, 490, 485, ..., 55, 50
# ─────────────────────────────────────────────────────────────────
wavelength_cases = [501] + list(range(495, 45, -5))
# → [501, 495, 490, 485, ..., 55, 50]

print(f"\n─── Study 1: Wavelength Reduction ───")
print(f"Cases: {len(wavelength_cases)}  ({wavelength_cases[0]} → {wavelength_cases[-1]}, step=5)")
print(f"Total experiments: {len(wavelength_cases) * 3}")

y_full = df_wide['thickness_nm'].values

# Fixed 80/20 stratified split by material (same split for all wavelength cases)
idx_train, idx_test = train_test_split(
    np.arange(len(df_wide)),
    test_size=0.2,
    random_state=42,
    stratify=df_wide['material'].values
)
y_train_full = y_full[idx_train]
y_test_full  = y_full[idx_test]

results_wl = []
n_total_s1 = len(wavelength_cases)

for i, n_wl in enumerate(wavelength_cases):
    wl_indices  = np.round(np.linspace(0, len(wavelengths) - 1, n_wl)).astype(int)
    selected_wl = wavelengths[wl_indices]
    psi_cols    = [f'psi_{w}'   for w in selected_wl]
    delta_cols  = [f'delta_{w}' for w in selected_wl]

    X = df_wide[psi_cols + delta_cols].values
    X_train = X[idx_train]
    X_test  = X[idx_test]

    row = {'wavelengths': n_wl}
    for name, model in get_models().items():
        rmse, mae, r2, mse = train_and_evaluate(
            name, model, X_train.copy(), X_test.copy(), y_train_full, y_test_full
        )
        results_wl.append({'wavelengths': n_wl, 'model': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        row[name] = rmse

    print(f"  [{i+1:3d}/{n_total_s1}] wl={n_wl:4d}  "
          f"LR={row['Linear Regression']:9.4f}  "
          f"RF={row['Random Forest']:7.4f}  "
          f"MLP={row['MLP']:7.4f}  nm")

df_results_wl = pd.DataFrame(results_wl)
df_results_wl.to_csv(f"{OUT}/results_study1_wavelength.csv", index=False)
print(f"Saved: {OUT}/results_study1_wavelength.csv")

# ─────────────────────────────────────────────────────────────────
# STUDY 2 — DATASET SIZE REDUCTION
# Dense sweep: 61 → 1, step = 1
# ─────────────────────────────────────────────────────────────────
thickness_cases = list(range(61, 0, -1))
# → [61, 60, 59, ..., 2, 1]

print(f"\n─── Study 2: Dataset Size Reduction ───")
print(f"Cases: {len(thickness_cases)}  ({thickness_cases[0]} → {thickness_cases[-1]}, step=1)")
print(f"Total experiments: {len(thickness_cases) * 3}")

results_ds = []
n_total_s2 = len(thickness_cases)

for i, n_thick in enumerate(thickness_cases):
    t_indices      = np.round(np.linspace(0, len(all_thicknesses) - 1, n_thick)).astype(int)
    selected_thick = all_thicknesses[t_indices]
    df_sub         = df_wide[df_wide['thickness_nm'].isin(selected_thick)].copy()

    X_sub   = df_sub[all_feature_cols].values
    y_sub   = df_sub['thickness_nm'].values
    mat_sub = df_sub['material'].values

    # Stratified split; fall back to random when too few samples per class
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42, stratify=mat_sub
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42
        )

    row = {'n_thicknesses': n_thick}
    for name, model in get_models().items():
        rmse, mae, r2, mse = train_and_evaluate(
            name, model, X_tr.copy(), X_te.copy(), y_tr, y_te
        )
        results_ds.append({
            'n_thicknesses': n_thick,
            'total_samples': len(df_sub),
            'model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        row[name] = rmse

    print(f"  [{i+1:3d}/{n_total_s2}] thick={n_thick:3d}  samples={len(df_sub):4d}  "
          f"LR={row['Linear Regression']:9.4f}  "
          f"RF={row['Random Forest']:7.4f}  "
          f"MLP={row['MLP']:7.4f}  nm")

df_results_ds = pd.DataFrame(results_ds)
df_results_ds.to_csv(f"{OUT}/results_study2_dataset_size.csv", index=False)
print(f"Saved: {OUT}/results_study2_dataset_size.csv")

# ─────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

MODEL_NAMES = ['Linear Regression', 'Random Forest', 'MLP']
COLORS      = ['steelblue', 'darkorange', 'seagreen']

def save_plot(df_res, x_col, xlabel, metric, title, fpath,
              models=MODEL_NAMES, log_scale=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in zip(MODEL_NAMES, COLORS):
        if name not in models:
            continue
        sub = df_res[df_res['model'] == name].sort_values(x_col)
        ax.plot(sub[x_col], sub[metric], label=name, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=13)
    if metric == 'R2':
        ylabel = 'R² Score'
    elif metric == 'MSE':
        ylabel = 'MSE (nm²)'
    else:
        ylabel = f'{metric} (nm)'
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    if metric == 'R2':
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close()
    print(f"  Saved: {fpath}")

# ── Study 1 plots ──
save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'RMSE', 'Study 1 — Wavelength Reduction (RMSE)',
          f"{OUT}/plots/study1_rmse_all.png")

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'MAE',  'Study 1 — Wavelength Reduction (MAE)',
          f"{OUT}/plots/study1_mae_all.png")

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'R2',   'Study 1 — Wavelength Reduction (R² Score)',
          f"{OUT}/plots/study1_r2_all.png")

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'RMSE', 'Study 1 — Wavelength Reduction (RMSE, log scale)',
          f"{OUT}/plots/study1_rmse_log.png", log_scale=True)

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'RMSE', 'Study 1 — Wavelength Reduction: RF & MLP (zoomed)',
          f"{OUT}/plots/study1_rmse_zoomed.png",
          models=['Random Forest', 'MLP'])

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'MAE',  'Study 1 — Wavelength Reduction: RF & MLP (zoomed)',
          f"{OUT}/plots/study1_mae_zoomed.png",
          models=['Random Forest', 'MLP'])

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'R2',   'Study 1 — Wavelength Reduction: RF & MLP R² (zoomed)',
          f"{OUT}/plots/study1_r2_zoomed.png",
          models=['Random Forest', 'MLP'])

# ── Study 2 plots ──
save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'RMSE', 'Study 2 — Dataset Size Reduction (RMSE)',
          f"{OUT}/plots/study2_rmse_all.png")

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'MAE',  'Study 2 — Dataset Size Reduction (MAE)',
          f"{OUT}/plots/study2_mae_all.png")

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'R2',   'Study 2 — Dataset Size Reduction (R² Score)',
          f"{OUT}/plots/study2_r2_all.png")

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'RMSE', 'Study 2 — Dataset Size Reduction (RMSE, log scale)',
          f"{OUT}/plots/study2_rmse_log.png", log_scale=True)

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'RMSE', 'Study 2 — Dataset Size Reduction: RF & MLP (zoomed)',
          f"{OUT}/plots/study2_rmse_zoomed.png",
          models=['Random Forest', 'MLP'])

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'MAE',  'Study 2 — Dataset Size Reduction: RF & MLP (zoomed)',
          f"{OUT}/plots/study2_mae_zoomed.png",
          models=['Random Forest', 'MLP'])

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'R2',   'Study 2 — Dataset Size Reduction: RF & MLP R² (zoomed)',
          f"{OUT}/plots/study2_r2_zoomed.png",
          models=['Random Forest', 'MLP'])

# ── MSE plots (all models) ──
save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'MSE', 'Study 1 — Wavelength Reduction (MSE)',
          f"{OUT}/plots/study1_mse_all.png")

save_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
          'MSE', 'Study 1 — Wavelength Reduction: RF & MLP (MSE zoomed)',
          f"{OUT}/plots/study1_mse_zoomed.png",
          models=['Random Forest', 'MLP'])

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'MSE', 'Study 2 — Dataset Size Reduction (MSE)',
          f"{OUT}/plots/study2_mse_all.png")

save_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'MSE', 'Study 2 — Dataset Size Reduction: RF & MLP (MSE zoomed)',
          f"{OUT}/plots/study2_mse_zoomed.png",
          models=['Random Forest', 'MLP'])

# ── Best-model 4-metric panels ──
BEST_MODELS  = ['Random Forest', 'MLP']
BEST_COLORS  = ['darkorange', 'seagreen']
METRIC_INFO  = [
    ('MSE',  'MSE (nm²)',  False),
    ('RMSE', 'RMSE (nm)',  False),
    ('MAE',  'MAE (nm)',   False),
    ('R2',   'R² Score',   True),
]

for study_tag, df_r, x_col, xlabel in [
    ('study1', df_results_wl, 'wavelengths',    'Number of Wavelengths'),
    ('study2', df_results_ds, 'n_thicknesses',  'Thickness Values per Material'),
]:
    study_label = 'Wavelength Reduction' if study_tag == 'study1' else 'Dataset Size Reduction'
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (metric, ylabel, is_r2) in zip(axes.flatten(), METRIC_INFO):
        for name, color in zip(BEST_MODELS, BEST_COLORS):
            sub = df_r[df_r['model'] == name].sort_values(x_col)
            ax.plot(sub[x_col], sub[metric], label=name, color=color, linewidth=2)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{study_label} — {metric}', fontsize=13)
        if is_r2:
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle(f'Best Models — {study_label}: All 4 Regression Metrics (Ψ,Δ → d)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fpath = f"{OUT}/plots/{study_tag}_best_4metrics.png"
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ── 8-panel: both studies × 4 metrics for best models ──
fig, axes = plt.subplots(4, 2, figsize=(18, 24))
for row_idx, (metric, ylabel, is_r2) in enumerate(METRIC_INFO):
    for col_idx, (df_r, x_col, xlabel, study_label) in enumerate([
        (df_results_wl, 'wavelengths',   'Number of Wavelengths',        'Study 1 — Wavelength Reduction'),
        (df_results_ds, 'n_thicknesses', 'Thickness Values per Material', 'Study 2 — Dataset Size Reduction'),
    ]):
        ax = axes[row_idx, col_idx]
        for name, color in zip(BEST_MODELS, BEST_COLORS):
            sub = df_r[df_r['model'] == name].sort_values(x_col)
            ax.plot(sub[x_col], sub[metric], label=name, color=color, linewidth=2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{study_label} — {metric}', fontsize=12)
        if is_r2:
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle('Best Models (RF & MLP) — All Regression Metrics: MSE · RMSE · MAE · R²',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/summary_best_8panel.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/summary_best_8panel.png")

# ── 4-panel summary ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
panels = [
    (df_results_wl, 'wavelengths', 'Number of Wavelengths',
     'RMSE', 'Study 1 — Wavelength Reduction (RMSE)'),
    (df_results_wl, 'wavelengths', 'Number of Wavelengths',
     'MAE',  'Study 1 — Wavelength Reduction (MAE)'),
    (df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
     'RMSE', 'Study 2 — Dataset Size Reduction (RMSE)'),
    (df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
     'MAE',  'Study 2 — Dataset Size Reduction (MAE)'),
]

# ── 6-panel summary with R² ──
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
panels6 = [
    (df_results_wl, 'wavelengths', 'Number of Wavelengths',
     'RMSE', 'Study 1 — Wavelength Reduction (RMSE)'),
    (df_results_wl, 'wavelengths', 'Number of Wavelengths',
     'MAE',  'Study 1 — Wavelength Reduction (MAE)'),
    (df_results_wl, 'wavelengths', 'Number of Wavelengths',
     'R2',   'Study 1 — Wavelength Reduction (R²)'),
    (df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
     'RMSE', 'Study 2 — Dataset Size Reduction (RMSE)'),
    (df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
     'MAE',  'Study 2 — Dataset Size Reduction (MAE)'),
    (df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
     'R2',   'Study 2 — Dataset Size Reduction (R²)'),
]
for ax, (df_r, x_col, xlabel, metric, title) in zip(axes.flatten(), panels6):
    for name, color in zip(MODEL_NAMES, COLORS):
        sub = df_r[df_r['model'] == name].sort_values(x_col)
        ax.plot(sub[x_col], sub[metric], label=name, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ylabel = 'R² Score' if metric == 'R2' else f'{metric} (nm)'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    if metric == 'R2':
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle('ML Dense Study — RMSE, MAE, R² (Ψ,Δ → d)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/plots/summary_6panel_with_r2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/summary_6panel_with_r2.png")
for ax, (df_r, x_col, xlabel, metric, title) in zip(axes.flatten(), panels):
    for name, color in zip(MODEL_NAMES, COLORS):
        sub = df_r[df_r['model'] == name].sort_values(x_col)
        ax.plot(sub[x_col], sub[metric], label=name, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(f"{metric} (nm)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle("ML Dense Study — Ellipsometric Thickness Prediction (Ψ,Δ → d)",
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/summary_4panel.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/summary_4panel.png")

# ── 4-panel log-scale summary ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (df_r, x_col, xlabel, metric, title) in zip(axes.flatten(), panels):
    for name, color in zip(MODEL_NAMES, COLORS):
        sub = df_r[df_r['model'] == name].sort_values(x_col)
        ax.plot(sub[x_col], sub[metric], label=name, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(f"{metric} (nm, log)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle("ML Dense Study — Log Scale (Ψ,Δ → d)",
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/summary_4panel_log.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/summary_4panel_log.png")

# ─────────────────────────────────────────────────────────────────
# COPY CODE INTO OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────
shutil.copy("ml_dense.py", f"{OUT}/ml_dense.py")
print(f"\nSaved: {OUT}/ml_dense.py  (code copy)")

# ─────────────────────────────────────────────────────────────────
# WRITE README.txt
# ─────────────────────────────────────────────────────────────────
readme = f"""
=======================================================================
ML DENSE STUDY — ELLIPSOMETRIC THICKNESS PREDICTION
thesis: "Analysis of Optical Measurements by ML and Analytical Models"
=======================================================================

OBJECTIVE
---------
Predict thin-film thickness d from ellipsometric spectra (Psi, Delta)
using three regression models. Two dense parametric studies were run.

Problem type : Regression
Input  (X)   : [psi_300, ..., psi_800, delta_300, ..., delta_800]  (1002 features)
Output (y)   : thickness_nm  (20-80 nm, scalar)

PHYSICAL SETUP
--------------
System        : Air / semiconductor thin film / SiO2 substrate
Materials     : Si, GaAs, GaN, Ge, InP  (5 materials)
Thickness     : 20 - 80 nm, 1 nm step   (61 values)
Wavelength    : 300 - 800 nm, 1 nm step (501 points)
Angle         : 70 degrees (fixed)
Simulation    : Fresnel 3-layer thin-film interference model

DATASET
-------
Source file   : ellipsometry_dataset_raw.csv
Format        : long format (material, thickness_nm, wavelength_nm, psi_deg, delta_deg)
Rows          : 152,805
Pivoted to    : ml_feature_matrix.csv
               wide format (305 samples x 1004 columns = material + thickness + 1002 features)

TRAIN / TEST SPLIT
------------------
Ratio         : 80% train / 20% test
Stratified by : material (ensures all 5 materials in both sets)
Random seed   : 42

MODELS
------
1. Linear Regression   - baseline, no scaling
2. Random Forest       - 100 trees, n_jobs=-1, random_state=42
3. MLP (Neural Net)    - layers: 256->128->64, ReLU, early stopping
                         StandardScaler applied (fit on train only)

STUDY 1 — WAVELENGTH RESOLUTION REDUCTION
------------------------------------------
Logic         : Evenly subsample N wavelengths from the full 300-800 nm range
Sequence      : {wavelength_cases[0]}, {wavelength_cases[1]}, {wavelength_cases[2]}, ... {wavelength_cases[-1]}
Cases         : {len(wavelength_cases)} wavelength levels
Step          : 5 (except first step: 501 -> 495 = 6)
Experiments   : {len(wavelength_cases)} x 3 models = {len(wavelength_cases)*3}
Fixed         : Full 305 samples, same train/test split every case
Output        : results_study1_wavelength.csv

STUDY 2 — DATASET SIZE REDUCTION
----------------------------------
Logic         : Evenly subsample N thickness values per material from 20-80 nm
Sequence      : {thickness_cases[0]}, {thickness_cases[1]}, {thickness_cases[2]}, ... {thickness_cases[-1]}
Cases         : {len(thickness_cases)} thickness levels
Step          : 1
Experiments   : {len(thickness_cases)} x 3 models = {len(thickness_cases)*3}
Fixed         : Full 501 wavelengths, fresh split per case
Output        : results_study2_dataset_size.csv

TOTAL EXPERIMENTS: {len(wavelength_cases)*3 + len(thickness_cases)*3}

METRICS
-------
RMSE : Root Mean Squared Error (nm)
MAE  : Mean Absolute Error (nm)

FILES IN THIS FOLDER
--------------------
ml_feature_matrix.csv          - Wide-format feature matrix (305 x 1004)
ellipsometry_dataset_raw.csv   - Original raw dataset (152,805 rows)
results_study1_wavelength.csv  - Study 1 results (wavelength x model x RMSE/MAE)
results_study2_dataset_size.csv- Study 2 results (n_thick x model x RMSE/MAE)
ml_dense.py                    - Full Python script that generated all outputs
plots/
  study1_rmse_all.png          - Study 1 RMSE all 3 models (full scale)
  study1_mae_all.png           - Study 1 MAE all 3 models
  study1_rmse_log.png          - Study 1 RMSE log scale
  study1_rmse_zoomed.png       - Study 1 RMSE RF + MLP only (zoomed)
  study1_mae_zoomed.png        - Study 1 MAE RF + MLP only (zoomed)
  study2_rmse_all.png          - Study 2 RMSE all 3 models (full scale)
  study2_mae_all.png           - Study 2 MAE all 3 models
  study2_rmse_log.png          - Study 2 RMSE log scale
  study2_rmse_zoomed.png       - Study 2 RMSE RF + MLP only (zoomed)
  study2_mae_zoomed.png        - Study 2 MAE RF + MLP only (zoomed)
  summary_4panel.png           - 4-panel combined summary
  summary_4panel_log.png       - 4-panel combined summary (log scale)

GENERATED: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
=======================================================================
"""

with open(f"{OUT}/README.txt", "w") as f:
    f.write(readme)
print(f"Saved: {OUT}/README.txt")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE — Final best results")
print("="*60)

print("\nStudy 1 — Best RMSE (full 501 wavelengths):")
s1_best = df_results_wl[df_results_wl['wavelengths'] == 501][['model','RMSE','MAE']]
print(s1_best.to_string(index=False))

print("\nStudy 2 — Best RMSE (full 61 thicknesses):")
s2_best = df_results_ds[df_results_ds['n_thicknesses'] == 61][['model','RMSE','MAE']]
print(s2_best.to_string(index=False))

print(f"\nAll outputs saved to: {OUT}/")
