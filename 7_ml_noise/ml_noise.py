"""
ml_noise.py — Noise-Robustness ML Study (5 random seeds)
─────────────────────────────────────────────────────────────────────────────
Repeats the full averaged ML study (ml_averaged.py) on a single noisy dataset.
Run this script once per noise type by changing NOISE_TYPE below.

Noise types available:
    'gaussian'  → dataset_gaussian_noise.csv
    'relative'  → dataset_relative_noise.csv
    'uniform'   → dataset_uniform_noise.csv
    'bias'      → dataset_bias_noise.csv
    'mixed'     → dataset_mixed_noise.csv

Studies (identical to ml_averaged.py for direct comparison):
  Study 1 — Wavelength Reduction  : 501 → 50, step=5   (91 cases × 3 models × 5 seeds = 1365 fits)
  Study 2 — Dataset Size Reduction : 61 → 1,  step=1   (61 cases × 3 models × 5 seeds =  915 fits)
  Total: 2280 model fits per noise type

Output folder: results/<noise_type>/
  results_study1_wavelength_avg.csv
  results_study2_dataset_size_avg.csv
  plots/
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
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  ← change NOISE_TYPE here for each run
# ─────────────────────────────────────────────────────────────────
NOISE_TYPE = 'mixed'   # options: gaussian | relative | uniform | bias | mixed

NOISE_FILES = {
    'gaussian': '../2_simulation/noise_datasets/dataset_gaussian_noise.csv',
    'relative': '../2_simulation/noise_datasets/dataset_relative_noise.csv',
    'uniform':  '../2_simulation/noise_datasets/dataset_uniform_noise.csv',
    'bias':     '../2_simulation/noise_datasets/dataset_bias_noise.csv',
    'mixed':    '../2_simulation/noise_datasets/dataset_mixed_noise.csv',
}

SEEDS       = [42, 0, 1, 7, 13]
MODEL_NAMES = ['Linear Regression', 'Random Forest', 'MLP']
COLORS      = ['steelblue', 'darkorange', 'seagreen']
METRICS     = ['MSE', 'RMSE', 'MAE', 'R2']
METRIC_YLABELS = {
    'MSE':  'MSE (nm²)',
    'RMSE': 'RMSE (nm)',
    'MAE':  'MAE (nm)',
    'R2':   'R² Score'
}

OUT = f"results/{NOISE_TYPE}"
os.makedirs(f"{OUT}/plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PIVOT NOISY DATASET
# ─────────────────────────────────────────────────────────────────
noise_file = NOISE_FILES[NOISE_TYPE]
print(f"Noise type : {NOISE_TYPE}")
print(f"Input file : {noise_file}")
print("Loading and pivoting dataset...")

df = pd.read_csv(noise_file)

wavelengths     = np.arange(300, 801, 1)   # 501 wavelength points
all_thicknesses = np.arange(20, 81, 1)     # 61 thickness values

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
df_wide.to_csv(f"{OUT}/ml_feature_matrix.csv", index=False)
print(f"Saved: {OUT}/ml_feature_matrix.csv")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — MODEL + EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────
def get_models(seed):
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
    if name == 'MLP':
        scaler  = StandardScaler()
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
# STUDY 1 — WAVELENGTH REDUCTION (91 cases, averaged over 5 seeds)
# ─────────────────────────────────────────────────────────────────
wavelength_cases = [501] + list(range(495, 45, -5))

print(f"\n─── Study 1: Wavelength Reduction ({NOISE_TYPE} noise, {len(SEEDS)} seeds) ───")
print(f"Cases: {len(wavelength_cases)}  ({wavelength_cases[0]} → {wavelength_cases[-1]})")
print(f"Total fits: {len(wavelength_cases)} × 3 × {len(SEEDS)} = "
      f"{len(wavelength_cases)*3*len(SEEDS)}")

results_wl  = []
n_total_s1  = len(wavelength_cases)

for i, n_wl in enumerate(wavelength_cases):
    wl_indices  = np.round(np.linspace(0, len(wavelengths) - 1, n_wl)).astype(int)
    selected_wl = wavelengths[wl_indices]
    psi_cols    = [f'psi_{w}'   for w in selected_wl]
    delta_cols  = [f'delta_{w}' for w in selected_wl]
    X = df_wide[psi_cols + delta_cols].values

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

    for name in MODEL_NAMES:
        row = {'wavelengths': n_wl, 'model': name}
        for m in METRICS:
            vals = accum[name][m]
            row[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            row[f'{m}_std']  = round(float(np.std(vals)),  4)
        results_wl.append(row)

    if (i + 1) % 10 == 0 or i == 0 or i == n_total_s1 - 1:
        rf_rmse  = np.mean(accum['Random Forest']['RMSE'])
        mlp_rmse = np.mean(accum['MLP']['RMSE'])
        print(f"  [{i+1:3d}/{n_total_s1}] wl={n_wl:4d}  "
              f"RF={rf_rmse:.4f}  MLP={mlp_rmse:.4f}  nm (mean)")

df_results_wl = pd.DataFrame(results_wl)
df_results_wl.to_csv(f"{OUT}/results_study1_wavelength_avg.csv", index=False)
print(f"Saved: {OUT}/results_study1_wavelength_avg.csv")

# ─────────────────────────────────────────────────────────────────
# STUDY 2 — DATASET SIZE REDUCTION (61 cases, averaged over 5 seeds)
# ─────────────────────────────────────────────────────────────────
thickness_cases = list(range(61, 0, -1))

print(f"\n─── Study 2: Dataset Size Reduction ({NOISE_TYPE} noise, {len(SEEDS)} seeds) ───")
print(f"Cases: {len(thickness_cases)}  ({thickness_cases[0]} → {thickness_cases[-1]})")
print(f"Total fits: {len(thickness_cases)} × 3 × {len(SEEDS)} = "
      f"{len(thickness_cases)*3*len(SEEDS)}")

results_ds  = []
n_total_s2  = len(thickness_cases)

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

    if (i + 1) % 10 == 0 or i == 0 or i == n_total_s2 - 1:
        rf_rmse  = np.mean(accum['Random Forest']['RMSE'])
        mlp_rmse = np.mean(accum['MLP']['RMSE'])
        print(f"  [{i+1:3d}/{n_total_s2}] thick={n_thick:3d}  "
              f"RF={rf_rmse:.4f}  MLP={mlp_rmse:.4f}  nm (mean)")

df_results_ds = pd.DataFrame(results_ds)
df_results_ds.to_csv(f"{OUT}/results_study2_dataset_size_avg.csv", index=False)
print(f"Saved: {OUT}/results_study2_dataset_size_avg.csv")

# ─────────────────────────────────────────────────────────────────
# PLOTTING — mean line + ±1σ shaded band  (same style as ml_averaged.py)
# ─────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

def save_avg_plot(df_res, x_col, xlabel, metric, title, fpath,
                  models=MODEL_NAMES, log_scale=False):
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
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

noise_label = NOISE_TYPE.capitalize() + ' noise'

# ── Study 1 plots ──
for metric in METRICS:
    # All 3 models
    save_avg_plot(
        df_results_wl, 'wavelengths', 'Number of wavelengths', metric,
        f'Study 1 — Wavelength Reduction [{noise_label}] — {metric} (mean ± 1σ)',
        f'{OUT}/plots/s1_{metric.lower()}_all3.png'
    )
    # RF + MLP only
    save_avg_plot(
        df_results_wl, 'wavelengths', 'Number of wavelengths', metric,
        f'Study 1 — Wavelength Reduction [{noise_label}] — {metric} RF & MLP',
        f'{OUT}/plots/s1_{metric.lower()}_rfmlp.png',
        models=['Random Forest', 'MLP']
    )
    # Log scale
    save_avg_plot(
        df_results_wl, 'wavelengths', 'Number of wavelengths', metric,
        f'Study 1 — Wavelength Reduction [{noise_label}] — {metric} (log scale)',
        f'{OUT}/plots/s1_{metric.lower()}_log.png',
        log_scale=True
    )

# ── Study 2 plots ──
for metric in METRICS:
    save_avg_plot(
        df_results_ds, 'n_thicknesses', 'Number of thickness values', metric,
        f'Study 2 — Dataset Size Reduction [{noise_label}] — {metric} (mean ± 1σ)',
        f'{OUT}/plots/s2_{metric.lower()}_all3.png'
    )
    save_avg_plot(
        df_results_ds, 'n_thicknesses', 'Number of thickness values', metric,
        f'Study 2 — Dataset Size Reduction [{noise_label}] — {metric} RF & MLP',
        f'{OUT}/plots/s2_{metric.lower()}_rfmlp.png',
        models=['Random Forest', 'MLP']
    )
    save_avg_plot(
        df_results_ds, 'n_thicknesses', 'Number of thickness values', metric,
        f'Study 2 — Dataset Size Reduction [{noise_label}] — {metric} (log scale)',
        f'{OUT}/plots/s2_{metric.lower()}_log.png',
        log_scale=True
    )

# ── 8-panel summary (RF + MLP, both studies, all 4 metrics) ──
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle(f'Summary — RF & MLP — {noise_label} (mean ± 1σ, 5 seeds)', fontsize=15)

for col, metric in enumerate(METRICS):
    for row, (df_res, x_col, xlabel) in enumerate([
        (df_results_wl, 'wavelengths',    'Wavelengths'),
        (df_results_ds, 'n_thicknesses',  'Thickness values'),
    ]):
        ax = axes[row][col]
        for name, color in zip(['Random Forest', 'MLP'], ['darkorange', 'seagreen']):
            sub  = df_res[df_res['model'] == name].sort_values(x_col)
            x    = sub[x_col].values
            mean = sub[f'{metric}_mean'].values
            std  = sub[f'{metric}_std'].values
            ax.plot(x, mean, label=name, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(METRIC_YLABELS[metric], fontsize=10)
        study = 'Study 1 (Wavelength)' if row == 0 else 'Study 2 (Dataset Size)'
        ax.set_title(f'{study} — {metric}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUT}/plots/summary_best_8panel.png', dpi=150)
plt.close(fig)

print(f"\nAll plots saved to {OUT}/plots/")
print(f"\n{'='*60}")
print(f"COMPLETED: {NOISE_TYPE} noise study")
print(f"  Study 1 results → {OUT}/results_study1_wavelength_avg.csv")
print(f"  Study 2 results → {OUT}/results_study2_dataset_size_avg.csv")
print(f"  Plots           → {OUT}/plots/  ({len(os.listdir(f'{OUT}/plots'))} files)")
print(f"{'='*60}")
print(f"\nNext step: change NOISE_TYPE at the top of this script")
print(f"  Options remaining: gaussian | relative | uniform | bias | mixed")
