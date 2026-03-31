"""
ml_averaged.py — Averaged ML Study (5 random seeds)
─────────────────────────────────────────────────────────────────────────────
Each (experiment × model) is evaluated 5 times under different random seeds.
Seeds control BOTH the train/test split AND the MLP/RF initialization.
Results are reported as mean ± std across seeds.
Plots show smooth mean curves with ±1σ shaded bands → publication quality.

Studies:
  Study 1 — Wavelength Reduction  : 501 → 50, step=5   (91 cases × 3 models × 5 seeds = 1365 fits)
  Study 2 — Dataset Size Reduction : 61 → 1,  step=1   (61 cases × 3 models × 5 seeds =  915 fits)
  Total: 2280 model fits
"""

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
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
SEEDS = [42, 0, 1, 7, 13]          # 5 random seeds for averaging
OUT   = "ml_averaged_study"
os.makedirs(f"{OUT}/plots", exist_ok=True)

MODEL_NAMES = ['Linear Regression', 'Random Forest', 'MLP']
COLORS      = ['steelblue', 'darkorange', 'seagreen']
METRICS     = ['MSE', 'RMSE', 'MAE', 'R2']
METRIC_YLABELS = {'MSE': 'MSE (nm²)', 'RMSE': 'RMSE (nm)', 'MAE': 'MAE (nm)', 'R2': 'R² Score'}

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PIVOT DATASET
# ─────────────────────────────────────────────────────────────────
print("Loading and pivoting dataset...")
df = pd.read_csv("ellipsometry_dataset.csv")

wavelengths      = np.arange(300, 801, 1)   # 501 wavelength points
all_thicknesses  = np.arange(20, 81, 1)     # 61 thickness values

psi_wide = df.pivot_table(
    index=['material', 'thickness_nm'], columns='wavelength_nm', values='psi_deg'
)
delta_wide = df.pivot_table(
    index=['material', 'thickness_nm'], columns='wavelength_nm', values='delta_deg'
)
psi_wide.columns   = [f'psi_{int(c)}'   for c in psi_wide.columns]
delta_wide.columns = [f'delta_{int(c)}' for c in delta_wide.columns]

df_wide = pd.concat([psi_wide, delta_wide], axis=1).reset_index()

psi_cols_full    = [f'psi_{w}'   for w in wavelengths]
delta_cols_full  = [f'delta_{w}' for w in wavelengths]
all_feature_cols = psi_cols_full + delta_cols_full
y_full           = df_wide['thickness_nm'].values

print(f"Wide dataset: {df_wide.shape[0]} samples × {df_wide.shape[1]-2} features")

# Save outputs
df_wide.to_csv(f"{OUT}/ml_feature_matrix.csv", index=False)
shutil.copy("ellipsometry_dataset.csv", f"{OUT}/ellipsometry_dataset_raw.csv")
print(f"Saved: {OUT}/ml_feature_matrix.csv")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — MODEL + EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────
def get_models(seed):
    """Return fresh model instances with the given random seed."""
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            max_iter=2000,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1
        ),
    }

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    """Fit model and return (RMSE, MAE, R², MSE)."""
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
    return rmse, mae, r2, mse

# ─────────────────────────────────────────────────────────────────
# STUDY 1 — WAVELENGTH REDUCTION  (averaged over SEEDS)
# Dense sweep: 501 → 50, step=5  →  91 cases
# ─────────────────────────────────────────────────────────────────
wavelength_cases = [501] + list(range(495, 45, -5))

print(f"\n─── Study 1: Wavelength Reduction (averaged, {len(SEEDS)} seeds) ───")
print(f"Cases: {len(wavelength_cases)}  ({wavelength_cases[0]} → {wavelength_cases[-1]})")
print(f"Total fits: {len(wavelength_cases)} × 3 × {len(SEEDS)} = "
      f"{len(wavelength_cases)*3*len(SEEDS)}")

results_wl = []
n_total_s1 = len(wavelength_cases)

for i, n_wl in enumerate(wavelength_cases):
    wl_indices  = np.round(np.linspace(0, len(wavelengths) - 1, n_wl)).astype(int)
    selected_wl = wavelengths[wl_indices]
    psi_cols    = [f'psi_{w}'   for w in selected_wl]
    delta_cols  = [f'delta_{w}' for w in selected_wl]
    X = df_wide[psi_cols + delta_cols].values

    # Accumulate metrics per model across seeds
    accum = {name: {m: [] for m in METRICS} for name in MODEL_NAMES}

    for seed in SEEDS:
        idx_tr, idx_te = train_test_split(
            np.arange(len(df_wide)), test_size=0.2,
            random_state=seed, stratify=df_wide['material'].values
        )
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y_full[idx_tr], y_full[idx_te]

        for name, model in get_models(seed).items():
            rmse, mae, r2, mse = train_and_evaluate(
                name, model, X_tr.copy(), X_te.copy(), y_tr, y_te
            )
            accum[name]['RMSE'].append(rmse)
            accum[name]['MAE'].append(mae)
            accum[name]['R2'].append(r2)
            accum[name]['MSE'].append(mse)

    # Collapse to mean ± std
    for name in MODEL_NAMES:
        row = {'wavelengths': n_wl, 'model': name}
        for m in METRICS:
            vals = accum[name][m]
            row[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            row[f'{m}_std']  = round(float(np.std(vals)),  4)
        results_wl.append(row)

    if (i+1) % 10 == 0 or i == 0 or i == n_total_s1 - 1:
        rf_rmse  = np.mean(accum['Random Forest']['RMSE'])
        mlp_rmse = np.mean(accum['MLP']['RMSE'])
        print(f"  [{i+1:3d}/{n_total_s1}] wl={n_wl:4d}  "
              f"RF={rf_rmse:.4f}  MLP={mlp_rmse:.4f}  nm (mean)")

df_results_wl = pd.DataFrame(results_wl)
df_results_wl.to_csv(f"{OUT}/results_study1_wavelength_avg.csv", index=False)
print(f"Saved: {OUT}/results_study1_wavelength_avg.csv")

# ─────────────────────────────────────────────────────────────────
# STUDY 2 — DATASET SIZE REDUCTION  (averaged over SEEDS)
# Dense sweep: 61 → 1, step=1  →  61 cases
# ─────────────────────────────────────────────────────────────────
thickness_cases = list(range(61, 0, -1))

print(f"\n─── Study 2: Dataset Size Reduction (averaged, {len(SEEDS)} seeds) ───")
print(f"Cases: {len(thickness_cases)}  ({thickness_cases[0]} → {thickness_cases[-1]})")
print(f"Total fits: {len(thickness_cases)} × 3 × {len(SEEDS)} = "
      f"{len(thickness_cases)*3*len(SEEDS)}")

results_ds = []
n_total_s2 = len(thickness_cases)

for i, n_thick in enumerate(thickness_cases):
    t_indices      = np.round(np.linspace(0, len(all_thicknesses) - 1, n_thick)).astype(int)
    selected_thick = all_thicknesses[t_indices]
    df_sub         = df_wide[df_wide['thickness_nm'].isin(selected_thick)].copy()
    X_sub          = df_sub[all_feature_cols].values
    y_sub          = df_sub['thickness_nm'].values
    mat_sub        = df_sub['material'].values

    accum = {name: {m: [] for m in METRICS} for name in MODEL_NAMES}

    for seed in SEEDS:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=seed, stratify=mat_sub
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=seed
            )

        for name, model in get_models(seed).items():
            rmse, mae, r2, mse = train_and_evaluate(
                name, model, X_tr.copy(), X_te.copy(), y_tr, y_te
            )
            accum[name]['RMSE'].append(rmse)
            accum[name]['MAE'].append(mae)
            accum[name]['R2'].append(r2)
            accum[name]['MSE'].append(mse)

    for name in MODEL_NAMES:
        row = {'n_thicknesses': n_thick, 'total_samples': len(df_sub), 'model': name}
        for m in METRICS:
            vals = accum[name][m]
            row[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            row[f'{m}_std']  = round(float(np.std(vals)),  4)
        results_ds.append(row)

    if (i+1) % 10 == 0 or i == 0 or i == n_total_s2 - 1:
        rf_rmse  = np.mean(accum['Random Forest']['RMSE'])
        mlp_rmse = np.mean(accum['MLP']['RMSE'])
        print(f"  [{i+1:3d}/{n_total_s2}] thick={n_thick:3d}  "
              f"RF={rf_rmse:.4f}  MLP={mlp_rmse:.4f}  nm (mean)")

df_results_ds = pd.DataFrame(results_ds)
df_results_ds.to_csv(f"{OUT}/results_study2_dataset_size_avg.csv", index=False)
print(f"Saved: {OUT}/results_study2_dataset_size_avg.csv")

# ─────────────────────────────────────────────────────────────────
# PLOTTING  (mean line + ±1σ shaded band)
# ─────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

def save_avg_plot(df_res, x_col, xlabel, metric, title, fpath,
                  models=MODEL_NAMES, log_scale=False):
    """Plot mean ± 1σ shaded band for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in zip(MODEL_NAMES, COLORS):
        if name not in models:
            continue
        sub  = df_res[df_res['model'] == name].sort_values(x_col)
        x    = sub[x_col].values
        mean = sub[f'{metric}_mean'].values
        std  = sub[f'{metric}_std'].values
        ax.plot(x, mean, label=name, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(METRIC_YLABELS[metric], fontsize=13)
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

# ── Study 1 ──
for metric in METRICS:
    save_avg_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
                  metric, f'Study 1 — Wavelength Reduction ({metric}, mean ± 1σ)',
                  f"{OUT}/plots/study1_{metric.lower()}_avg.png")

save_avg_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
              'RMSE', 'Study 1 — Wavelength Reduction (RMSE, log scale, mean ± 1σ)',
              f"{OUT}/plots/study1_rmse_log_avg.png", log_scale=True)

for metric in ['RMSE', 'MAE', 'R2', 'MSE']:
    save_avg_plot(df_results_wl, 'wavelengths', 'Number of Wavelengths',
                  metric, f'Study 1 — Wavelength Reduction: RF & MLP ({metric}, zoomed)',
                  f"{OUT}/plots/study1_{metric.lower()}_zoomed_avg.png",
                  models=['Random Forest', 'MLP'])

# ── Study 2 ──
for metric in METRICS:
    save_avg_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
                  metric, f'Study 2 — Dataset Size Reduction ({metric}, mean ± 1σ)',
                  f"{OUT}/plots/study2_{metric.lower()}_avg.png")

save_avg_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
              'RMSE', 'Study 2 — Dataset Size Reduction (RMSE, log scale, mean ± 1σ)',
              f"{OUT}/plots/study2_rmse_log_avg.png", log_scale=True)

for metric in ['RMSE', 'MAE', 'R2', 'MSE']:
    save_avg_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
                  metric, f'Study 2 — Dataset Size Reduction: RF & MLP ({metric}, zoomed)',
                  f"{OUT}/plots/study2_{metric.lower()}_zoomed_avg.png",
                  models=['Random Forest', 'MLP'])

# ── Best-model 4-metric panels (one per study) ──
BEST_MODELS = ['Random Forest', 'MLP']
BEST_COLORS = ['darkorange', 'seagreen']
METRIC_INFO = [
    ('MSE',  False),
    ('RMSE', False),
    ('MAE',  False),
    ('R2',   True),
]

for study_tag, df_r, x_col, xlabel in [
    ('study1', df_results_wl, 'wavelengths',    'Number of Wavelengths'),
    ('study2', df_results_ds, 'n_thicknesses',  'Thickness Values per Material'),
]:
    label = 'Wavelength Reduction' if study_tag == 'study1' else 'Dataset Size Reduction'
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (metric, is_r2) in zip(axes.flatten(), METRIC_INFO):
        for name, color in zip(BEST_MODELS, BEST_COLORS):
            sub  = df_r[df_r['model'] == name].sort_values(x_col)
            x    = sub[x_col].values
            mean = sub[f'{metric}_mean'].values
            std  = sub[f'{metric}_std'].values
            ax.plot(x, mean, label=name, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(METRIC_YLABELS[metric], fontsize=12)
        ax.set_title(f'{label} — {metric}  (mean ± 1σ)', fontsize=13)
        if is_r2:
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle(f'Best Models (RF & MLP) — {label}: All 4 Regression Metrics  '
                 f'[averaged over {len(SEEDS)} seeds]',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fpath = f"{OUT}/plots/{study_tag}_best_4metrics_avg.png"
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ── 8-panel summary: both studies × 4 metrics  (best models, shaded) ──
fig, axes = plt.subplots(4, 2, figsize=(18, 24))
for row_idx, (metric, is_r2) in enumerate(METRIC_INFO):
    for col_idx, (df_r, x_col, xlabel, slabel) in enumerate([
        (df_results_wl, 'wavelengths',   'Number of Wavelengths',        'Study 1 — Wavelength Reduction'),
        (df_results_ds, 'n_thicknesses', 'Thickness Values per Material', 'Study 2 — Dataset Size Reduction'),
    ]):
        ax = axes[row_idx, col_idx]
        for name, color in zip(BEST_MODELS, BEST_COLORS):
            sub  = df_r[df_r['model'] == name].sort_values(x_col)
            x    = sub[x_col].values
            mean = sub[f'{metric}_mean'].values
            std  = sub[f'{metric}_std'].values
            ax.plot(x, mean, label=name, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(METRIC_YLABELS[metric], fontsize=11)
        ax.set_title(f'{slabel} — {metric}', fontsize=12)
        if is_r2:
            ax.set_ylim(-0.1, 1.05)
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(
    f'Best Models (RF & MLP) — All Regression Metrics  '
    f'[mean ± 1σ, {len(SEEDS)} seeds]\n'
    'MSE · RMSE · MAE · R²   |   Wavelength Reduction & Dataset Size Reduction',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f"{OUT}/plots/summary_best_8panel_avg.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/summary_best_8panel_avg.png")

# ── 6-panel: RMSE + MAE + R² for both studies (all 3 models, shaded) ──
fig, axes = plt.subplots(2, 3, figsize=(22, 13))
panels6 = [
    (df_results_wl, 'wavelengths',    'Number of Wavelengths',        'RMSE', 'Study 1 — Wavelength Reduction (RMSE)'),
    (df_results_wl, 'wavelengths',    'Number of Wavelengths',        'MAE',  'Study 1 — Wavelength Reduction (MAE)'),
    (df_results_wl, 'wavelengths',    'Number of Wavelengths',        'R2',   'Study 1 — Wavelength Reduction (R²)'),
    (df_results_ds, 'n_thicknesses',  'Thickness Values per Material','RMSE', 'Study 2 — Dataset Size Reduction (RMSE)'),
    (df_results_ds, 'n_thicknesses',  'Thickness Values per Material','MAE',  'Study 2 — Dataset Size Reduction (MAE)'),
    (df_results_ds, 'n_thicknesses',  'Thickness Values per Material','R2',   'Study 2 — Dataset Size Reduction (R²)'),
]
for ax, (df_r, x_col, xlabel, metric, title) in zip(axes.flatten(), panels6):
    for name, color in zip(MODEL_NAMES, COLORS):
        sub  = df_r[df_r['model'] == name].sort_values(x_col)
        x    = sub[x_col].values
        mean = sub[f'{metric}_mean'].values
        std  = sub[f'{metric}_std'].values
        ax.plot(x, mean, label=name, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(METRIC_YLABELS[metric], fontsize=11)
    ax.set_title(title, fontsize=12)
    if metric == 'R2':
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(
    f'ML Averaged Study — RMSE · MAE · R²   (Ψ,Δ → d)  '
    f'[mean ± 1σ, {len(SEEDS)} seeds]',
    fontsize=15, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f"{OUT}/plots/summary_6panel_avg.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/summary_6panel_avg.png")

# ── All-3-models 4-metric panel for Study 1 ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (metric, is_r2) in zip(axes.flatten(), METRIC_INFO):
    for name, color in zip(MODEL_NAMES, COLORS):
        sub  = df_results_wl[df_results_wl['model'] == name].sort_values('wavelengths')
        x    = sub['wavelengths'].values
        mean = sub[f'{metric}_mean'].values
        std  = sub[f'{metric}_std'].values
        ax.plot(x, mean, label=name, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel('Number of Wavelengths', fontsize=12)
    ax.set_ylabel(METRIC_YLABELS[metric], fontsize=12)
    ax.set_title(f'Study 1 — Wavelength Reduction: {metric}  (mean ± 1σ)', fontsize=13)
    if is_r2:
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(f'Study 1 — All Models — 4 Regression Metrics  '
             f'[mean ± 1σ, {len(SEEDS)} seeds]', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/study1_all3_4metrics_avg.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/study1_all3_4metrics_avg.png")

# ── All-3-models 4-metric panel for Study 2 ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (metric, is_r2) in zip(axes.flatten(), METRIC_INFO):
    for name, color in zip(MODEL_NAMES, COLORS):
        sub  = df_results_ds[df_results_ds['model'] == name].sort_values('n_thicknesses')
        x    = sub['n_thicknesses'].values
        mean = sub[f'{metric}_mean'].values
        std  = sub[f'{metric}_std'].values
        ax.plot(x, mean, label=name, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel('Thickness Values per Material', fontsize=12)
    ax.set_ylabel(METRIC_YLABELS[metric], fontsize=12)
    ax.set_title(f'Study 2 — Dataset Size Reduction: {metric}  (mean ± 1σ)', fontsize=13)
    if is_r2:
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle(f'Study 2 — All Models — 4 Regression Metrics  '
             f'[mean ± 1σ, {len(SEEDS)} seeds]', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/plots/study2_all3_4metrics_avg.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT}/plots/study2_all3_4metrics_avg.png")

# ─────────────────────────────────────────────────────────────────
# COPY CODE INTO OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────
shutil.copy("ml_averaged.py", f"{OUT}/ml_averaged.py")
print(f"\nSaved: {OUT}/ml_averaged.py  (code copy)")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("DONE — Final best results  (mean ± std across 5 seeds)")
print("="*65)

for study_label, df_r, x_col, x_val in [
    ("Study 1 — Best (501 wavelengths)", df_results_wl,  'wavelengths',   501),
    ("Study 2 — Best (61 thicknesses)",  df_results_ds, 'n_thicknesses', 61),
]:
    print(f"\n{study_label}:")
    best = df_r[df_r[x_col] == x_val][['model','MSE_mean','RMSE_mean','MAE_mean','R2_mean',
                                         'RMSE_std','R2_std']].set_index('model')
    print(best.to_string())

print(f"\nAll outputs saved to: {OUT}/")
print(f"Seeds used: {SEEDS}")
print(f"Total model fits: {len(wavelength_cases)*3*len(SEEDS) + len(thickness_cases)*3*len(SEEDS)}")
