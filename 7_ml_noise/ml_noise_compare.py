"""
ml_noise_compare.py
Cross-noise comparison: clean baseline vs 5 noise types
Loads Study 1 + Study 2 results for all conditions and produces
summary tables + comparison plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Output folder ──────────────────────────────────────────────────────────────
OUT_DIR = 'results/comparison'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Conditions (label → Study-1 CSV, Study-2 CSV) ─────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    'Clean':    (
        os.path.join(BASE, '../6_ml_averaged/results_study1_wavelength_avg.csv'),
        os.path.join(BASE, '../6_ml_averaged/results_study2_dataset_size_avg.csv'),
    ),
    'Gaussian': (
        os.path.join(BASE, 'results/gaussian/results_study1_wavelength_avg.csv'),
        os.path.join(BASE, 'results/gaussian/results_study2_dataset_size_avg.csv'),
    ),
    'Relative': (
        os.path.join(BASE, 'results/relative/results_study1_wavelength_avg.csv'),
        os.path.join(BASE, 'results/relative/results_study2_dataset_size_avg.csv'),
    ),
    'Uniform':  (
        os.path.join(BASE, 'results/uniform/results_study1_wavelength_avg.csv'),
        os.path.join(BASE, 'results/uniform/results_study2_dataset_size_avg.csv'),
    ),
    'Bias':     (
        os.path.join(BASE, 'results/bias/results_study1_wavelength_avg.csv'),
        os.path.join(BASE, 'results/bias/results_study2_dataset_size_avg.csv'),
    ),
    'Mixed':    (
        os.path.join(BASE, 'results/mixed/results_study1_wavelength_avg.csv'),
        os.path.join(BASE, 'results/mixed/results_study2_dataset_size_avg.csv'),
    ),
}

MODELS   = ['Random Forest', 'MLP']
COLORS   = plt.cm.tab10.colors          # up to 10 distinct colours
COND_CLR = {cond: COLORS[i] for i, cond in enumerate(CONDITIONS)}
LINESTYLES = {'Clean': '-', 'Gaussian': '--', 'Relative': '-.', 'Uniform': ':',
              'Bias': (0,(5,1)), 'Mixed': (0,(3,1,1,1))}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load all data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading results for all conditions...")
study1 = {}   # cond → DataFrame
study2 = {}

for cond, (s1_path, s2_path) in CONDITIONS.items():
    study1[cond] = pd.read_csv(s1_path)
    study2[cond] = pd.read_csv(s2_path)
    print(f"  {cond}: Study1 {len(study1[cond])} rows, Study2 {len(study2[cond])} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Summary table — full-resolution (501 wavelengths) metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding summary table at 501 wavelengths...")
rows = []
for cond, df in study1.items():
    df_full = df[df['wavelengths'] == 501]
    for model in ['Linear Regression', 'Random Forest', 'MLP']:
        r = df_full[df_full['model'] == model]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            'condition': cond,
            'model': model,
            'RMSE_mean': round(r['RMSE_mean'], 4),
            'RMSE_std':  round(r['RMSE_std'],  4),
            'MAE_mean':  round(r['MAE_mean'],  4),
            'MAE_std':   round(r['MAE_std'],   4),
            'R2_mean':   round(r['R2_mean'],   4),
            'R2_std':    round(r['R2_std'],    4),
            'MSE_mean':  round(r['MSE_mean'],  4),
            'MSE_std':   round(r['MSE_std'],   4),
        })

summary_df = pd.DataFrame(rows)
summary_csv = os.path.join(OUT_DIR, 'summary_501wl.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"  Saved: {summary_csv}")
print(summary_df[summary_df['model'] != 'Linear Regression'].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Helper: Study-1 filter to RF/MLP only
# ═══════════════════════════════════════════════════════════════════════════════
def s1_model(cond, model):
    df = study1[cond]
    return df[df['model'] == model].sort_values('wavelengths', ascending=False)

def s2_model(cond, model):
    df = study2[cond]
    return df[df['model'] == model].sort_values('n_thicknesses', ascending=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plot A — RMSE bar chart at 501 wavelengths (RF + MLP, side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlot A: RMSE bar chart at 501 wavelengths...")

fig, ax = plt.subplots(figsize=(10, 5))
cond_list = list(CONDITIONS.keys())
x = np.arange(len(cond_list))
width = 0.35

for idx, model in enumerate(MODELS):
    means = []
    stds  = []
    for cond in cond_list:
        row = summary_df[(summary_df['condition'] == cond) & (summary_df['model'] == model)]
        means.append(float(row['RMSE_mean']))
        stds.append(float(row['RMSE_std']))
    offset = (idx - 0.5) * width
    bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                  label=model, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(cond_list)
ax.set_xlabel('Data Condition')
ax.set_ylabel('RMSE (nm)')
ax.set_title('RMSE at Full Wavelength Range (501 wavelengths)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'A_rmse_bar_501wl.png'), dpi=150)
plt.close(fig)
print("  Saved: A_rmse_bar_501wl.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plot B — Study 1: RMSE vs wavelengths — one subplot per model
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot B: Study 1 RMSE vs wavelengths (RF + MLP panels)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, model in zip(axes, MODELS):
    for cond in cond_list:
        d = s1_model(cond, model)
        ax.plot(d['wavelengths'], d['RMSE_mean'],
                color=COND_CLR[cond], linestyle=LINESTYLES[cond], linewidth=1.8,
                label=cond)
        ax.fill_between(d['wavelengths'],
                        d['RMSE_mean'] - d['RMSE_std'],
                        d['RMSE_mean'] + d['RMSE_std'],
                        color=COND_CLR[cond], alpha=0.12)
    ax.set_xlabel('Number of Wavelengths')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title(f'Study 1 — {model}\nRMSE vs Wavelength Range')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'B_study1_rmse_wavelengths.png'), dpi=150)
plt.close(fig)
print("  Saved: B_study1_rmse_wavelengths.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Plot C — Study 1: R² vs wavelengths
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot C: Study 1 R² vs wavelengths...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, model in zip(axes, MODELS):
    for cond in cond_list:
        d = s1_model(cond, model)
        ax.plot(d['wavelengths'], d['R2_mean'],
                color=COND_CLR[cond], linestyle=LINESTYLES[cond], linewidth=1.8,
                label=cond)
        ax.fill_between(d['wavelengths'],
                        d['R2_mean'] - d['R2_std'],
                        d['R2_mean'] + d['R2_std'],
                        color=COND_CLR[cond], alpha=0.12)
    ax.set_xlabel('Number of Wavelengths')
    ax.set_ylabel('R²')
    ax.set_title(f'Study 1 — {model}\nR² vs Wavelength Range')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'C_study1_r2_wavelengths.png'), dpi=150)
plt.close(fig)
print("  Saved: C_study1_r2_wavelengths.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Plot D — Study 2: RMSE vs dataset size
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot D: Study 2 RMSE vs dataset size...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, model in zip(axes, MODELS):
    for cond in cond_list:
        d = s2_model(cond, model)
        ax.plot(d['n_thicknesses'], d['RMSE_mean'],
                color=COND_CLR[cond], linestyle=LINESTYLES[cond], linewidth=1.8,
                label=cond)
        ax.fill_between(d['n_thicknesses'],
                        d['RMSE_mean'] - d['RMSE_std'],
                        d['RMSE_mean'] + d['RMSE_std'],
                        color=COND_CLR[cond], alpha=0.12)
    ax.set_xlabel('Number of Thickness Values')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title(f'Study 2 — {model}\nRMSE vs Dataset Size')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'D_study2_rmse_dataset_size.png'), dpi=150)
plt.close(fig)
print("  Saved: D_study2_rmse_dataset_size.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Plot E — Degradation heatmap: (RMSE_noisy - RMSE_clean) / RMSE_clean × 100
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot E: RMSE degradation heatmap at 501 wavelengths...")

noise_conds = ['Gaussian', 'Relative', 'Uniform', 'Bias', 'Mixed']
heat_data = np.zeros((len(MODELS), len(noise_conds)))

for j, cond in enumerate(noise_conds):
    for i, model in enumerate(MODELS):
        clean_rmse = float(summary_df[(summary_df['condition'] == 'Clean') &
                                      (summary_df['model'] == model)]['RMSE_mean'])
        noisy_rmse = float(summary_df[(summary_df['condition'] == cond) &
                                      (summary_df['model'] == model)]['RMSE_mean'])
        heat_data[i, j] = (noisy_rmse - clean_rmse) / clean_rmse * 100

fig, ax = plt.subplots(figsize=(8, 3.5))
im = ax.imshow(heat_data, cmap='RdYlGn_r', aspect='auto', vmin=0)
ax.set_xticks(range(len(noise_conds)))
ax.set_xticklabels(noise_conds)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS)
ax.set_title('RMSE Degradation (%) vs Clean Baseline at 501 Wavelengths')
plt.colorbar(im, ax=ax, label='% increase over clean RMSE')
for i in range(len(MODELS)):
    for j in range(len(noise_conds)):
        ax.text(j, i, f'{heat_data[i,j]:.1f}%', ha='center', va='center',
                fontsize=10, color='black')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'E_degradation_heatmap.png'), dpi=150)
plt.close(fig)
print("  Saved: E_degradation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Plot F — Study 1 RMSE difference from clean (noise penalty curve)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot F: RMSE noise penalty vs wavelengths...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, model in zip(axes, MODELS):
    clean_d = s1_model('Clean', model).set_index('wavelengths')
    for cond in noise_conds:
        noisy_d = s1_model(cond, model).set_index('wavelengths')
        delta = noisy_d['RMSE_mean'] - clean_d['RMSE_mean']
        ax.plot(delta.index, delta.values,
                color=COND_CLR[cond], linestyle=LINESTYLES[cond],
                linewidth=1.8, label=cond)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Number of Wavelengths')
    ax.set_ylabel('ΔRMSE = noisy − clean (nm)')
    ax.set_title(f'Study 1 — {model}\nNoise Penalty vs Wavelength Range')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'F_study1_noise_penalty.png'), dpi=150)
plt.close(fig)
print("  Saved: F_study1_noise_penalty.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Print final summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("COMPARISON COMPLETE")
print(f"  Output folder : {OUT_DIR}")
print("  Files written :")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    {f}")
print("=" * 65)
