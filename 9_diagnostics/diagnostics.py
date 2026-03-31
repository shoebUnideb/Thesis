"""
diagnostics.py
─────────────────────────────────────────────────────────────────────────────
Two diagnostic analyses on the clean dataset (full 501 wavelengths, seed 42):

  A. Residual / error distribution
       Histogram of (predicted − true) thickness for:
         • Linear Regression
         • Random Forest
         • MLP
         • Analytical (grid-search, same test split)
       Shows whether errors are zero-centred (random) or systematically biased.

  B. Random Forest feature importance
       Total Gini importance per wavelength, shown as:
         • Combined (Ψ + Δ summed) bar chart over 300–800 nm
         • Separate Ψ and Δ panels
       Reveals which spectral regions drive the RF decisions physically.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
CLEAN_F    = os.path.join(BASE, '../1_raw_data/ellipsometry_dataset.csv')
NK_DIR     = os.path.join(BASE, '../1_raw_data/semiconductor_nk')
SIO2_F     = os.path.join(BASE, '../1_raw_data/substrate/sio2_sellmeier_300_800nm_1nm.csv')
OUT        = os.path.join(BASE, 'plots')

SEED       = 42
WAVELENGTHS = np.arange(300, 801, 1)   # 501 points
THETA0     = np.radians(70.0)

NK_FILES = {
    'Si':   'si_nk_300_800nm_1nm.csv',
    'GaAs': 'gaas_nk_300_800nm_1nm_final.csv',
    'GaN':  'gan_nk_300_800nm_1nm.csv',
    'Ge':   'ge_nk_300_800nm_1nm.csv',
    'InP':  'inp_nk_300_800nm_1nm.csv',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & pivot dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading dataset...")
df = pd.read_csv(CLEAN_F)

psi_w   = df.pivot_table(index=['material','thickness_nm'],
                          columns='wavelength_nm', values='psi_deg')
delta_w = df.pivot_table(index=['material','thickness_nm'],
                          columns='wavelength_nm', values='delta_deg')
psi_w.columns   = [f'psi_{int(c)}'   for c in psi_w.columns]
delta_w.columns = [f'delta_{int(c)}' for c in delta_w.columns]
wide = pd.concat([psi_w, delta_w], axis=1).reset_index()

psi_cols   = [f'psi_{w}'   for w in WAVELENGTHS]
delta_cols = [f'delta_{w}' for w in WAVELENGTHS]
feat_cols  = psi_cols + delta_cols

X = wide[feat_cols].values
y = wide['thickness_nm'].values

idx_tr, idx_te = train_test_split(
    np.arange(len(wide)), test_size=0.2,
    random_state=SEED, stratify=wide['material'].values
)
X_tr, X_te = X[idx_tr], X[idx_te]
y_tr, y_te = y[idx_tr], y[idx_te]
mat_te     = wide['material'].values[idx_te]

print(f"  Train: {len(idx_tr)} samples | Test: {len(idx_te)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Train ML models
# ═══════════════════════════════════════════════════════════════════════════════
print("Training models (seed 42)...")

lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred_lr = lr.predict(X_te)

rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
mlp = MLPRegressor(hidden_layer_sizes=(256,128,64), activation='relu',
                    max_iter=2000, random_state=SEED,
                    early_stopping=True, validation_fraction=0.1)
mlp.fit(X_tr_sc, y_tr)
y_pred_mlp = mlp.predict(X_te_sc)

for name, preds in [('LR', y_pred_lr), ('RF', y_pred_rf), ('MLP', y_pred_mlp)]:
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    print(f"  {name}: RMSE = {rmse:.4f} nm")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Analytical solver (same Fresnel forward model) — test set only
# ═══════════════════════════════════════════════════════════════════════════════
print("Running analytical inverse solver on test set...")

# Load n/k
nk_data = {}
for mat, fname in NK_FILES.items():
    d = pd.read_csv(os.path.join(NK_DIR, fname))
    nk_data[mat] = d['n'].values - 1j * d['k'].values
sio2_df = pd.read_csv(SIO2_F)
n_sio2  = sio2_df['n'].values.astype(complex)

D_GRID = np.arange(1.0, 150.1, 0.1)
ALL_IDX = np.arange(len(WAVELENGTHS), dtype=int)

def _snell(ni, nt, ti):    return np.arcsin(ni * np.sin(ti) / nt)
def _frs(ni, nt, ti, tt):  return (ni*np.cos(ti)-nt*np.cos(tt))/(ni*np.cos(ti)+nt*np.cos(tt))
def _frp(ni, nt, ti, tt):  return (nt*np.cos(ti)-ni*np.cos(tt))/(nt*np.cos(ti)+ni*np.cos(tt))

def build_table(n1_arr, n2_arr):
    n1 = n1_arr[np.newaxis,:]
    n2 = n2_arr[np.newaxis,:]
    wl = WAVELENGTHS[np.newaxis,:]
    d  = D_GRID[:,np.newaxis]
    t1 = _snell(1.0+0j, n1, THETA0)
    t2 = _snell(n1, n2, t1)
    r01s = _frs(1.0+0j, n1, THETA0, t1); r12s = _frs(n1, n2, t1, t2)
    r01p = _frp(1.0+0j, n1, THETA0, t1); r12p = _frp(n1, n2, t1, t2)
    beta = (2*np.pi/wl)*n1*d*np.cos(t1)
    e    = np.exp(-2j*beta)
    rs   = (r01s + r12s*e)/(1 + r01s*r12s*e)
    rp   = (r01p + r12p*e)/(1 + r01p*r12p*e)
    rho  = rp/rs
    return np.degrees(np.arctan(np.abs(rho))).real, np.degrees(np.angle(rho)).real

tables = {mat: build_table(nk_data[mat], n_sio2) for mat in NK_FILES}

def invert(mat, wl_idx, psi_m, delta_m):
    pt, dt = tables[mat]
    ssr = np.sum((pt[:,wl_idx]-psi_m)**2 + (dt[:,wl_idx]-delta_m)**2, axis=1)
    return D_GRID[np.argmin(ssr)]

y_pred_an = np.array([
    invert(mat_te[i], ALL_IDX,
           X_te[i, :len(WAVELENGTHS)],
           X_te[i, len(WAVELENGTHS):])
    for i in range(len(idx_te))
])
rmse_an = np.sqrt(mean_squared_error(y_te, y_pred_an))
print(f"  Analytical: RMSE = {rmse_an:.4f} nm")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Residuals
# ═══════════════════════════════════════════════════════════════════════════════
res = {
    'Linear Regression': y_pred_lr  - y_te,
    'Random Forest':     y_pred_rf  - y_te,
    'MLP':               y_pred_mlp - y_te,
    'Analytical':        y_pred_an  - y_te,
}

colors = {
    'Linear Regression': '#e41a1c',
    'Random Forest':     '#377eb8',
    'MLP':               '#4daf4a',
    'Analytical':        '#984ea3',
}

# ── Plot A1: overlaid histograms (RF / MLP / Analytical — exclude LR, too wide) ──
print("\nPlot A1: residual histograms (RF / MLP / Analytical)...")
fig, ax = plt.subplots(figsize=(9, 5))
bin_edges = np.linspace(-15, 15, 80)
for name in ['Random Forest', 'MLP', 'Analytical']:
    r = res[name]
    ax.hist(r, bins=bin_edges, density=True, alpha=0.45,
            color=colors[name], label=name)
    mu, sd = r.mean(), r.std()
    xs = np.linspace(bin_edges[0], bin_edges[-1], 300)
    ax.plot(xs, norm.pdf(xs, mu, sd), color=colors[name], linewidth=1.8, linestyle='--')
ax.axvline(0, color='black', linewidth=1.0, linestyle='-')
ax.set_xlabel('Residual  (predicted − true)  (nm)')
ax.set_ylabel('Probability density')
ax.set_title('Error Distribution — Clean Data (seed 42, test set)\nRF vs MLP vs Analytical')
ax.legend()
ax.grid(linestyle='--', alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'A1_residuals_rf_mlp_analytical.png'), dpi=150)
plt.close(fig)
print("  Saved: A1_residuals_rf_mlp_analytical.png")

# ── Plot A2: 4-panel (all 4 models, individual) ──────────────────────────────
print("Plot A2: 4-panel residuals (all models)...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
bin_edges2 = np.linspace(-30, 30, 100)
for ax, (name, r) in zip(axes.flat, res.items()):
    mu, sd = r.mean(), r.std()
    rmse   = np.sqrt(np.mean(r**2))
    ax.hist(r, bins=bin_edges2, density=True, alpha=0.7, color=colors[name])
    xs = np.linspace(bin_edges2[0], bin_edges2[-1], 400)
    ax.plot(xs, norm.pdf(xs, mu, sd), color='black', linewidth=1.5,
            label=f'μ={mu:.3f}\nσ={sd:.3f}\nRMSE={rmse:.3f} nm')
    ax.axvline(0, color='red', linewidth=1.0, linestyle='--')
    ax.set_title(name)
    ax.set_xlabel('Residual (nm)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.35)
fig.suptitle('Error Distributions — Clean Data (seed 42, test set)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'A2_residuals_4panel.png'), dpi=150)
plt.close(fig)
print("  Saved: A2_residuals_4panel.png")

# ── Plot A3: residuals per material (MLP) ────────────────────────────────────
print("Plot A3: residuals per material (MLP)...")
materials = sorted(np.unique(mat_te))
fig, ax = plt.subplots(figsize=(9, 5))
mat_colors = plt.cm.tab10.colors
for i, mat in enumerate(materials):
    mask = mat_te == mat
    r    = res['MLP'][mask]
    ax.hist(r, bins=40, density=True, alpha=0.55,
            color=mat_colors[i], label=f'{mat}  (μ={r.mean():.2f}, σ={r.std():.2f} nm)')
ax.axvline(0, color='black', linewidth=1.0)
ax.set_xlabel('Residual  (predicted − true)  (nm)')
ax.set_ylabel('Probability density')
ax.set_title('MLP Error Distribution by Material (clean data)')
ax.legend(fontsize=8)
ax.grid(linestyle='--', alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'A3_residuals_mlp_per_material.png'), dpi=150)
plt.close(fig)
print("  Saved: A3_residuals_mlp_per_material.png")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n  Residual summary (test set):")
print(f"  {'Model':<20} {'Mean':>8} {'Std':>8} {'RMSE':>8}  interpretation")
print("  " + "-"*65)
for name, r in res.items():
    mu, sd, rmse_r = r.mean(), r.std(), np.sqrt(np.mean(r**2))
    bias = "systematic bias" if abs(mu) > 0.5 else "zero-centred (random)"
    print(f"  {name:<20} {mu:>8.4f} {sd:>8.4f} {rmse_r:>8.4f}  {bias}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RF Feature importance
# ═══════════════════════════════════════════════════════════════════════════════
print("\nExtracting RF feature importances...")

importances = rf.feature_importances_           # [1002]
imp_psi   = importances[:len(WAVELENGTHS)]      # first 501 → Ψ
imp_delta = importances[len(WAVELENGTHS):]      # last  501 → Δ
imp_total = imp_psi + imp_delta                 # combined per wavelength

# ── Plot B1: combined importance (Ψ + Δ) ─────────────────────────────────────
print("Plot B1: RF feature importance (combined Ψ+Δ)...")
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(WAVELENGTHS, imp_total, width=1.0, color='#377eb8', alpha=0.8)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Feature Importance (Gini, Ψ + Δ)')
ax.set_title('Random Forest Feature Importance — Which Wavelengths Matter Most?')
ax.set_xlim(295, 805)
ax.grid(axis='y', linestyle='--', alpha=0.4)
# Annotate top-5 wavelengths
top5_idx = np.argsort(imp_total)[-5:][::-1]
for idx in top5_idx:
    ax.annotate(f'{WAVELENGTHS[idx]} nm',
                xy=(WAVELENGTHS[idx], imp_total[idx]),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=7, color='darkred')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'B1_rf_importance_combined.png'), dpi=150)
plt.close(fig)
print("  Saved: B1_rf_importance_combined.png")

# ── Plot B2: Ψ and Δ separately (two-panel) ───────────────────────────────────
print("Plot B2: RF feature importance (Ψ and Δ separately)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.bar(WAVELENGTHS, imp_psi,   width=1.0, color='steelblue',  alpha=0.8)
ax1.set_ylabel('Importance (Gini)')
ax1.set_title('RF Feature Importance — Ψ (Psi)')
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_xlim(295, 805)

ax2.bar(WAVELENGTHS, imp_delta, width=1.0, color='darkorange', alpha=0.8)
ax2.set_ylabel('Importance (Gini)')
ax2.set_title('RF Feature Importance — Δ (Delta)')
ax2.set_xlabel('Wavelength (nm)')
ax2.grid(axis='y', linestyle='--', alpha=0.4)

fig.suptitle('Random Forest Feature Importance by Ellipsometric Angle', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'B2_rf_importance_psi_delta.png'), dpi=150)
plt.close(fig)
print("  Saved: B2_rf_importance_psi_delta.png")

# ── Plot B3: cumulative importance ────────────────────────────────────────────
print("Plot B3: cumulative RF importance...")
sorted_imp = np.sort(imp_total)[::-1]
cumulative  = np.cumsum(sorted_imp)
n_feats     = np.arange(1, len(sorted_imp)+1)

# How many wavelengths cover 50%, 80%, 95%?
for threshold in [0.50, 0.80, 0.95]:
    n = int(np.searchsorted(cumulative, threshold)) + 1
    print(f"  Top {n:3d} wavelengths cover {threshold*100:.0f}% of cumulative importance")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n_feats, cumulative, color='#377eb8', linewidth=1.8)
for threshold, ls in [(0.50,'--'),(0.80,'-.'),(0.95,':')]:
    n = int(np.searchsorted(cumulative, threshold)) + 1
    ax.axhline(threshold, color='gray', linestyle=ls, linewidth=0.9)
    ax.axvline(n,         color='gray', linestyle=ls, linewidth=0.9)
    ax.text(n+5, threshold-0.03, f'{threshold*100:.0f}%  (top {n})', fontsize=8)
ax.set_xlabel('Number of wavelengths (ranked by importance)')
ax.set_ylabel('Cumulative importance')
ax.set_title('Cumulative RF Feature Importance')
ax.set_xlim(0, 501)
ax.set_ylim(0, 1.05)
ax.grid(linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'B3_rf_importance_cumulative.png'), dpi=150)
plt.close(fig)
print("  Saved: B3_rf_importance_cumulative.png")

# ── Save importance CSV ───────────────────────────────────────────────────────
imp_df = pd.DataFrame({
    'wavelength_nm': WAVELENGTHS,
    'importance_psi':   imp_psi,
    'importance_delta': imp_delta,
    'importance_total': imp_total,
})
imp_df.to_csv(os.path.join(BASE, 'rf_feature_importance.csv'), index=False)
print("  Saved: rf_feature_importance.csv")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("DIAGNOSTICS COMPLETE")
print(f"  Output: {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"    {f}")
print("="*65)
