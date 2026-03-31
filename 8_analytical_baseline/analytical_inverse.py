"""
analytical_inverse.py
─────────────────────────────────────────────────────────────────────────────
Physics-based inverse solver for ellipsometry thickness retrieval.

Method: Lookup-table grid search
  1. Precompute the Fresnel thin-film forward model (same equations as
     2_simulation/main.py) over a fine grid of thickness values for every
     material × wavelength combination → lookup table.
  2. For every test sample (known material, measured Ψ(λ) and Δ(λ)):
       SSR(d) = Σ_λ [ (Ψ_model(d,λ) − Ψ_meas(λ))² + (Δ_model(d,λ) − Δ_meas(λ))² ]
     d_pred = argmin SSR over the thickness grid.
  3. Thickness RMSE is compared against the ML baselines.

Datasets evaluated:
  • Clean (1_raw_data/ellipsometry_dataset.csv)
  • Gaussian / Relative / Uniform / Bias / Mixed noise

Outputs (results/):
  • summary.csv                      — RMSE/MAE/R² at 501 wl for all 6 conditions
  • study1_wavelength_analytical.csv — RMSE vs n_wavelengths (analytical, clean only)
  • plots/A_*  bar chart: analytical vs RF vs MLP at 501 wl
  • plots/B_*  Study 1 curves: analytical + RF + MLP overlaid
  • plots/C_*  noise-type bar chart: analytical vs MLP
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

NK_DIR  = os.path.join(BASE, '../1_raw_data/semiconductor_nk')
SIO2_F  = os.path.join(BASE, '../1_raw_data/substrate/sio2_sellmeier_300_800nm_1nm.csv')
CLEAN_F = os.path.join(BASE, '../1_raw_data/ellipsometry_dataset.csv')

NOISE_FILES = {
    'Gaussian': os.path.join(BASE, '../2_simulation/noise_datasets/dataset_gaussian_noise.csv'),
    'Relative': os.path.join(BASE, '../2_simulation/noise_datasets/dataset_relative_noise.csv'),
    'Uniform':  os.path.join(BASE, '../2_simulation/noise_datasets/dataset_uniform_noise.csv'),
    'Bias':     os.path.join(BASE, '../2_simulation/noise_datasets/dataset_bias_noise.csv'),
    'Mixed':    os.path.join(BASE, '../2_simulation/noise_datasets/dataset_mixed_noise.csv'),
}

ML_CLEAN_S1  = os.path.join(BASE, '../6_ml_averaged/results_study1_wavelength_avg.csv')
ML_NOISE_DIR = os.path.join(BASE, '../7_ml_noise/results')

OUT = os.path.join(BASE, 'results')
os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)

# ── Physics constants ──────────────────────────────────────────────────────────
THETA0    = np.radians(70.0)      # angle of incidence (same as simulation)
N_AMBIENT = 1.0 + 0j              # air
WAVELENGTHS = np.arange(300, 801, 1)   # 501 wavelength points  (nm)

# Thickness grid for grid search
D_MIN, D_MAX, D_STEP = 1.0, 150.0, 0.1
D_GRID = np.arange(D_MIN, D_MAX + D_STEP / 2, D_STEP)   # 1491 points

NK_FILES = {
    'Si':   'si_nk_300_800nm_1nm.csv',
    'GaAs': 'gaas_nk_300_800nm_1nm_final.csv',
    'GaN':  'gan_nk_300_800nm_1nm.csv',
    'Ge':   'ge_nk_300_800nm_1nm.csv',
    'InP':  'inp_nk_300_800nm_1nm.csv',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load n/k data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading n/k data...")

sio2_df = pd.read_csv(SIO2_F)
n_sio2  = sio2_df['n'].values.astype(complex)    # substrate (lossless)

nk_data = {}   # material → complex refractive index array [501]
for mat, fname in NK_FILES.items():
    df = pd.read_csv(os.path.join(NK_DIR, fname))
    nk_data[mat] = df['n'].values - 1j * df['k'].values

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Fresnel forward model  (vectorised over wavelengths AND thicknesses)
# ═══════════════════════════════════════════════════════════════════════════════

def _snell(n_i, n_t, theta_i):
    return np.arcsin(n_i * np.sin(theta_i) / n_t)

def _fresnel_rs(n_i, n_t, th_i, th_t):
    return (n_i * np.cos(th_i) - n_t * np.cos(th_t)) / \
           (n_i * np.cos(th_i) + n_t * np.cos(th_t))

def _fresnel_rp(n_i, n_t, th_i, th_t):
    return (n_t * np.cos(th_i) - n_i * np.cos(th_t)) / \
           (n_t * np.cos(th_i) + n_i * np.cos(th_t))

def compute_lookup_table(n1_arr, n2_arr, d_grid, wavelengths, theta0):
    """
    Build psi_table[n_d, n_wl] and delta_table[n_d, n_wl] for a given
    material (n1_arr[n_wl]) on substrate (n2_arr[n_wl]).

    Parameters
    ----------
    n1_arr      : complex [n_wl]  — film refractive index vs wavelength
    n2_arr      : complex [n_wl]  — substrate refractive index vs wavelength
    d_grid      : float   [n_d]   — thickness values (nm)
    wavelengths : float   [n_wl]  — wavelength values (nm)
    theta0      : float           — angle of incidence (radians)
    """
    n_d  = len(d_grid)
    n_wl = len(wavelengths)

    # Broadcast shapes: d_grid [n_d,1]  /  wavelength-dependent arrays [1, n_wl]
    n1 = n1_arr[np.newaxis, :]          # [1, n_wl]
    n2 = n2_arr[np.newaxis, :]          # [1, n_wl]
    n0 = N_AMBIENT
    wl = wavelengths[np.newaxis, :]     # [1, n_wl]
    d  = d_grid[:, np.newaxis]          # [n_d, 1]

    theta1 = _snell(n0, n1, theta0)
    theta2 = _snell(n1, n2, theta1)

    r01_s = _fresnel_rs(n0, n1, theta0, theta1)
    r12_s = _fresnel_rs(n1, n2, theta1, theta2)
    r01_p = _fresnel_rp(n0, n1, theta0, theta1)
    r12_p = _fresnel_rp(n1, n2, theta1, theta2)

    beta     = (2 * np.pi / wl) * n1 * d * np.cos(theta1)   # [n_d, n_wl]
    exp_term = np.exp(-2j * beta)

    rs = (r01_s + r12_s * exp_term) / (1 + r01_s * r12_s * exp_term)
    rp = (r01_p + r12_p * exp_term) / (1 + r01_p * r12_p * exp_term)

    rho   = rp / rs
    psi   = np.degrees(np.arctan(np.abs(rho)))     # [n_d, n_wl]
    delta = np.degrees(np.angle(rho))              # [n_d, n_wl]

    return psi.real, delta.real

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Precompute lookup tables for all materials
# ═══════════════════════════════════════════════════════════════════════════════
print("Precomputing lookup tables (all materials × thickness grid)...")

tables = {}   # material → (psi_table[n_d, n_wl], delta_table[n_d, n_wl])
for mat, n1_arr in nk_data.items():
    psi_t, delta_t = compute_lookup_table(
        n1_arr, n_sio2, D_GRID, WAVELENGTHS, THETA0
    )
    tables[mat] = (psi_t, delta_t)
    print(f"  {mat}: psi_table shape {psi_t.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Inverse solver: given (material, wl_indices, psi_meas, delta_meas)
#    return predicted thickness
# ═══════════════════════════════════════════════════════════════════════════════

def inverse_solve(material, wl_indices, psi_meas, delta_meas):
    """
    Grid-search inverse solver.
    wl_indices : 1D int array — indices into WAVELENGTHS (0-based)
    psi_meas   : float [n_sel_wl]
    delta_meas : float [n_sel_wl]
    Returns    : predicted thickness (nm)
    """
    psi_t, delta_t = tables[material]
    pt = psi_t[:, wl_indices]        # [n_d, n_sel_wl]
    dt = delta_t[:, wl_indices]

    ssr = np.sum((pt - psi_meas) ** 2 + (dt - delta_meas) ** 2, axis=1)
    return D_GRID[np.argmin(ssr)]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Helper: load & pivot dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_wide(path):
    df = pd.read_csv(path)
    psi_w = df.pivot_table(
        index=['material', 'thickness_nm'], columns='wavelength_nm', values='psi_deg'
    )
    delta_w = df.pivot_table(
        index=['material', 'thickness_nm'], columns='wavelength_nm', values='delta_deg'
    )
    psi_w.columns   = [f'psi_{int(c)}'   for c in psi_w.columns]
    delta_w.columns = [f'delta_{int(c)}' for c in delta_w.columns]
    wide = pd.concat([psi_w, delta_w], axis=1).reset_index()
    return wide

psi_cols_all   = [f'psi_{w}'   for w in WAVELENGTHS]
delta_cols_all = [f'delta_{w}' for w in WAVELENGTHS]

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Evaluate solver on one dataset at a given set of wavelength indices
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(wide_df, wl_indices):
    """
    Run inverse solver on every row of wide_df using the given wavelength set.
    Returns RMSE, MAE, R².
    """
    psi_sel   = [f'psi_{WAVELENGTHS[i]}'   for i in wl_indices]
    delta_sel = [f'delta_{WAVELENGTHS[i]}' for i in wl_indices]

    preds = []
    for _, row in wide_df.iterrows():
        mat    = row['material']
        psi_m  = row[psi_sel].values.astype(float)
        delta_m = row[delta_sel].values.astype(float)
        d_pred = inverse_solve(mat, wl_indices, psi_m, delta_m)
        preds.append(d_pred)

    preds  = np.array(preds)
    truths = wide_df['thickness_nm'].values.astype(float)

    residuals = preds - truths
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae  = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((truths - truths.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot

    return float(rmse), float(mae), float(r2), preds, truths

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Full-resolution (501 wl) evaluation on all 6 conditions
# ═══════════════════════════════════════════════════════════════════════════════
ALL_WL_IDX = np.arange(len(WAVELENGTHS), dtype=int)   # all 501

all_conditions = {'Clean': CLEAN_F, **NOISE_FILES}

print("\n─── Full-resolution evaluation (501 wavelengths) ───")
summary_rows = []

for cond, path in all_conditions.items():
    wide = load_wide(path)
    rmse, mae, r2, preds, truths = evaluate_dataset(wide, ALL_WL_IDX)
    mse  = rmse ** 2
    summary_rows.append({'condition': cond, 'model': 'Analytical',
                          'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
                          'R2':   round(r2,   4), 'MSE': round(mse, 4)})
    print(f"  {cond:10s}  RMSE={rmse:.4f} nm  MAE={mae:.4f}  R²={r2:.4f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
print(f"\nSaved: results/summary.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Study 1 — wavelength sweep on CLEAN data (deterministic, no averaging needed)
# ═══════════════════════════════════════════════════════════════════════════════
wavelength_cases = [501] + list(range(495, 45, -5))   # same 91 cases as ML
print(f"\n─── Study 1: Wavelength Reduction (analytical, clean data) ───")
print(f"  {len(wavelength_cases)} cases: {wavelength_cases[0]} → {wavelength_cases[-1]}")

clean_wide = load_wide(CLEAN_F)
study1_rows = []

for i, n_wl in enumerate(wavelength_cases):
    wl_indices = np.round(np.linspace(0, len(WAVELENGTHS) - 1, n_wl)).astype(int)
    rmse, mae, r2, _, _ = evaluate_dataset(clean_wide, wl_indices)
    mse = rmse ** 2
    study1_rows.append({'wavelengths': n_wl, 'model': 'Analytical',
                         'RMSE_mean': round(rmse, 4), 'RMSE_std': 0.0,
                         'MAE_mean':  round(mae,  4), 'MAE_std':  0.0,
                         'R2_mean':   round(r2,   4), 'R2_std':   0.0,
                         'MSE_mean':  round(mse,  4), 'MSE_std':  0.0})
    if (i + 1) % 10 == 0 or i == 0 or i == len(wavelength_cases) - 1:
        print(f"  [{i+1:3d}/{len(wavelength_cases)}] wl={n_wl:4d}  RMSE={rmse:.4f} nm")

study1_df = pd.DataFrame(study1_rows)
study1_df.to_csv(os.path.join(OUT, 'study1_wavelength_analytical.csv'), index=False)
print(f"Saved: results/study1_wavelength_analytical.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Load ML baselines for comparison plots
# ═══════════════════════════════════════════════════════════════════════════════
def load_ml_summary_row(csv_path, condition, model):
    """Pull RMSE_mean/std at 501 wavelengths from an ML study1 CSV."""
    df = pd.read_csv(csv_path)
    row = df[(df['wavelengths'] == 501) & (df['model'] == model)]
    if row.empty:
        return None
    return {
        'condition': condition,
        'model': model,
        'RMSE': row.iloc[0]['RMSE_mean'],
        'RMSE_std': row.iloc[0]['RMSE_std'],
    }

ml_cond_paths = {
    'Clean':    ML_CLEAN_S1,
    'Gaussian': os.path.join(ML_NOISE_DIR, 'gaussian/results_study1_wavelength_avg.csv'),
    'Relative': os.path.join(ML_NOISE_DIR, 'relative/results_study1_wavelength_avg.csv'),
    'Uniform':  os.path.join(ML_NOISE_DIR, 'uniform/results_study1_wavelength_avg.csv'),
    'Bias':     os.path.join(ML_NOISE_DIR, 'bias/results_study1_wavelength_avg.csv'),
    'Mixed':    os.path.join(ML_NOISE_DIR, 'mixed/results_study1_wavelength_avg.csv'),
}

ml_rf_rows  = []
ml_mlp_rows = []
for cond, path in ml_cond_paths.items():
    rf  = load_ml_summary_row(path, cond, 'Random Forest')
    mlp = load_ml_summary_row(path, cond, 'MLP')
    if rf  is not None: ml_rf_rows.append(rf)
    if mlp is not None: ml_mlp_rows.append(mlp)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Plot A — RMSE bar chart: Analytical vs RF vs MLP (all conditions)
# ═══════════════════════════════════════════════════════════════════════════════
cond_list  = list(all_conditions.keys())
x          = np.arange(len(cond_list))
width      = 0.25
colors     = {'Analytical': '#e41a1c', 'Random Forest': '#377eb8', 'MLP': '#4daf4a'}

fig, ax = plt.subplots(figsize=(12, 5))
# Analytical
an_rmse = [summary_df[summary_df['condition'] == c]['RMSE'].values[0] for c in cond_list]
ax.bar(x - width, an_rmse, width, label='Analytical (Grid Search)',
       color=colors['Analytical'], alpha=0.85)

# RF
rf_rmse = [next((r['RMSE'] for r in ml_rf_rows if r['condition'] == c), np.nan)
           for c in cond_list]
rf_std  = [next((r['RMSE_std'] for r in ml_rf_rows if r['condition'] == c), 0)
           for c in cond_list]
ax.bar(x, rf_rmse, width, yerr=rf_std, capsize=3,
       label='Random Forest', color=colors['Random Forest'], alpha=0.85)

# MLP
mlp_rmse = [next((r['RMSE'] for r in ml_mlp_rows if r['condition'] == c), np.nan)
            for c in cond_list]
mlp_std  = [next((r['RMSE_std'] for r in ml_mlp_rows if r['condition'] == c), 0)
            for c in cond_list]
ax.bar(x + width, mlp_rmse, width, yerr=mlp_std, capsize=3,
       label='MLP', color=colors['MLP'], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(cond_list)
ax.set_xlabel('Data Condition')
ax.set_ylabel('RMSE (nm)')
ax.set_title('Thickness RMSE: Analytical vs ML at Full Wavelength Range (501 wavelengths)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'plots/A_rmse_bar_analytical_vs_ml.png'), dpi=150)
plt.close(fig)
print("\nSaved: plots/A_rmse_bar_analytical_vs_ml.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Plot B — Study 1: RMSE vs wavelengths — analytical + RF + MLP (clean only)
# ═══════════════════════════════════════════════════════════════════════════════
ml_s1_clean = pd.read_csv(ML_CLEAN_S1)
fig, ax = plt.subplots(figsize=(9, 5))

# Analytical (no std)
ax.plot(study1_df['wavelengths'], study1_df['RMSE_mean'],
        color=colors['Analytical'], linewidth=2.0, label='Analytical (Grid Search)')

for model, clr in [('Random Forest', colors['Random Forest']), ('MLP', colors['MLP'])]:
    d = ml_s1_clean[ml_s1_clean['model'] == model].sort_values('wavelengths', ascending=False)
    ax.plot(d['wavelengths'], d['RMSE_mean'], color=clr, linewidth=1.8, label=model)
    ax.fill_between(d['wavelengths'],
                    d['RMSE_mean'] - d['RMSE_std'],
                    d['RMSE_mean'] + d['RMSE_std'],
                    color=clr, alpha=0.15)

ax.set_xlabel('Number of Wavelengths')
ax.set_ylabel('RMSE (nm)')
ax.set_title('Study 1 (Clean Data) — Analytical vs ML\nRMSE vs Wavelength Range')
ax.legend()
ax.grid(linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'plots/B_study1_analytical_vs_ml.png'), dpi=150)
plt.close(fig)
print("Saved: plots/B_study1_analytical_vs_ml.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. Plot C — Noise robustness: analytical RMSE vs noise type (bar)
# ═══════════════════════════════════════════════════════════════════════════════
noise_conds  = ['Gaussian', 'Relative', 'Uniform', 'Bias', 'Mixed']
an_noise_rmse  = [summary_df[summary_df['condition'] == c]['RMSE'].values[0]
                  for c in noise_conds]
mlp_noise_rmse = [next((r['RMSE'] for r in ml_mlp_rows if r['condition'] == c), np.nan)
                  for c in noise_conds]
mlp_noise_std  = [next((r['RMSE_std'] for r in ml_mlp_rows if r['condition'] == c), 0)
                  for c in noise_conds]

x2 = np.arange(len(noise_conds))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x2 - 0.2, an_noise_rmse,  0.4, label='Analytical', color=colors['Analytical'], alpha=0.85)
ax.bar(x2 + 0.2, mlp_noise_rmse, 0.4, yerr=mlp_noise_std, capsize=4,
       label='MLP', color=colors['MLP'], alpha=0.85)

# Clean baseline lines
clean_an_rmse  = float(summary_df[summary_df['condition'] == 'Clean']['RMSE'])
clean_mlp_rmse = next(r['RMSE'] for r in ml_mlp_rows if r['condition'] == 'Clean')
ax.axhline(clean_an_rmse,  color=colors['Analytical'], linestyle='--', linewidth=1.2,
           label=f'Analytical clean = {clean_an_rmse:.3f} nm')
ax.axhline(clean_mlp_rmse, color=colors['MLP'],        linestyle='--', linewidth=1.2,
           label=f'MLP clean = {clean_mlp_rmse:.3f} nm')

ax.set_xticks(x2)
ax.set_xticklabels(noise_conds)
ax.set_xlabel('Noise Type')
ax.set_ylabel('RMSE (nm)')
ax.set_title('Noise Robustness: Analytical vs MLP')
ax.legend(fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'plots/C_noise_robustness_analytical_vs_mlp.png'), dpi=150)
plt.close(fig)
print("Saved: plots/C_noise_robustness_analytical_vs_mlp.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 13. Final summary print
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ANALYTICAL BASELINE COMPLETE")
print(f"  Output folder : {OUT}")
print("\n  Full-resolution RMSE (501 wavelengths):")
print(f"  {'Condition':<12} {'Analytical':>12} {'RF':>10} {'MLP':>10}")
print("  " + "-" * 46)
for cond in cond_list:
    an  = summary_df[summary_df['condition'] == cond]['RMSE'].values[0]
    rf  = next((r['RMSE'] for r in ml_rf_rows  if r['condition'] == cond), float('nan'))
    mlp = next((r['RMSE'] for r in ml_mlp_rows if r['condition'] == cond), float('nan'))
    print(f"  {cond:<12} {an:>12.4f} {rf:>10.4f} {mlp:>10.4f}")
print("=" * 65)
