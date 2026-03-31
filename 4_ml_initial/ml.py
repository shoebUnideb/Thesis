import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PIVOT DATASET
# Long format (152,805 rows) → Wide format (305 rows × 1002 features)
# One row = one (material, thickness) pair
# Features = [psi_300, ..., psi_800, delta_300, ..., delta_800]
# ─────────────────────────────────────────────────────────────────
print("Loading and pivoting dataset...")
df = pd.read_csv("ellipsometry_dataset.csv")

wavelengths = np.arange(300, 801, 1)  # 501 points

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

# Save full wide-format feature matrix
# Columns: material, thickness_nm, psi_300..psi_800, delta_300..delta_800
df_wide.to_csv("ml_feature_matrix.csv", index=False)
print(f"Wide dataset: {df_wide.shape[0]} samples × {df_wide.shape[1]-2} features")
print(f"Saved: ml_feature_matrix.csv")
# ─────────────────────────────────────────────────────────────────
# STEP 2 — MODEL FACTORY
# ─────────────────────────────────────────────────────────────────
def get_models():
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1
                             ),
        'MLP':               MLPRegressor(
                                hidden_layer_sizes=(256, 128, 64),
                                activation='relu',
                                max_iter=2000,
                                random_state=42,
                                early_stopping=True,
                                validation_fraction=0.1
                             ),
    }

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    """Train model (with scaling for MLP) and return RMSE, MAE."""
    if name == 'MLP':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    return rmse, mae

# ─────────────────────────────────────────────────────────────────
# STUDY 1 — WAVELENGTH RESOLUTION REDUCTION
# Fixed: all 305 samples, full thickness range
# Variable: number of wavelength points sampled evenly from 300–800 nm
# ─────────────────────────────────────────────────────────────────
print("\n─── Study 1: Wavelength Reduction ───")

wavelength_cases = [501, 250, 125, 75, 50]
results_wl = []

y_full = df_wide['thickness_nm'].values

# Stratified 80/20 split — indices fixed for all wavelength cases
X_dummy = df_wide[psi_cols_full].values   # placeholder for split indices
idx_train, idx_test = train_test_split(
    np.arange(len(df_wide)),
    test_size=0.2,
    random_state=42,
    stratify=df_wide['material'].values
)
y_train_full = y_full[idx_train]
y_test_full  = y_full[idx_test]

for n_wl in wavelength_cases:
    print(f"\n  Wavelengths: {n_wl}")

    # Evenly spaced indices across 501 wavelength positions
    wl_indices   = np.round(np.linspace(0, len(wavelengths) - 1, n_wl)).astype(int)
    selected_wl  = wavelengths[wl_indices]

    psi_cols   = [f'psi_{w}'   for w in selected_wl]
    delta_cols = [f'delta_{w}' for w in selected_wl]

    X = df_wide[psi_cols + delta_cols].values
    X_train, X_test = X[idx_train], X[idx_test]

    for name, model in get_models().items():
        rmse, mae = train_and_evaluate(
            name, model,
            X_train.copy(), X_test.copy(),
            y_train_full, y_test_full
        )
        results_wl.append({
            'wavelengths': n_wl,
            'model': name,
            'RMSE': round(rmse, 4),
            'MAE':  round(mae, 4)
        })
        print(f"    {name:20s}  RMSE={rmse:.4f} nm   MAE={mae:.4f} nm")

df_results_wl = pd.DataFrame(results_wl)
df_results_wl.to_csv("ml_results_wavelength.csv", index=False)
print("\nSaved: ml_results_wavelength.csv")

# ─────────────────────────────────────────────────────────────────
# STUDY 2 — DATASET SIZE REDUCTION
# Fixed: full 501 wavelengths
# Variable: number of thickness values per material (subsampled evenly)
# ─────────────────────────────────────────────────────────────────
print("\n─── Study 2: Dataset Size Reduction ───")

thickness_cases   = [61, 50, 40, 25, 10]
all_thicknesses   = np.arange(20, 81, 1)       # 61 values
all_feature_cols  = psi_cols_full + delta_cols_full
results_ds = []

for n_thick in thickness_cases:
    print(f"\n  Thicknesses per material: {n_thick}")

    # Evenly spaced subset of the 61 thickness values
    t_indices      = np.round(np.linspace(0, len(all_thicknesses) - 1, n_thick)).astype(int)
    selected_thick = all_thicknesses[t_indices]

    df_sub = df_wide[df_wide['thickness_nm'].isin(selected_thick)].copy()

    X_sub   = df_sub[all_feature_cols].values
    y_sub   = df_sub['thickness_nm'].values
    mat_sub = df_sub['material'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub,
        test_size=0.2,
        random_state=42,
        stratify=mat_sub
    )

    for name, model in get_models().items():
        rmse, mae = train_and_evaluate(
            name, model,
            X_tr.copy(), X_te.copy(),
            y_tr, y_te
        )
        results_ds.append({
            'n_thicknesses': n_thick,
            'total_samples': len(df_sub),
            'model': name,
            'RMSE': round(rmse, 4),
            'MAE':  round(mae, 4)
        })
        print(f"    {name:20s}  RMSE={rmse:.4f} nm   MAE={mae:.4f} nm")

df_results_ds = pd.DataFrame(results_ds)
df_results_ds.to_csv("ml_results_dataset_size.csv", index=False)
print("\nSaved: ml_results_dataset_size.csv")

# ─────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

MODEL_NAMES = ['Linear Regression', 'Random Forest', 'MLP']
COLORS      = ['steelblue', 'darkorange', 'seagreen']
MARKERS     = ['o', 's', '^']

def make_plot(df_res, x_col, x_label, metric, title, ax):
    for name, color, marker in zip(MODEL_NAMES, COLORS, MARKERS):
        sub = df_res[df_res['model'] == name].sort_values(x_col)
        ax.plot(sub[x_col], sub[metric],
                label=name, color=color, marker=marker,
                linewidth=2, markersize=7)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(f"{metric} (nm)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

# ── RMSE plots ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
make_plot(df_results_wl, 'wavelengths',   'Number of Wavelengths',
          'RMSE', 'Study 1: Wavelength Reduction vs RMSE', axes[0])
make_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'RMSE', 'Study 2: Dataset Size vs RMSE',         axes[1])
plt.tight_layout()
plt.savefig("images/ml_study_rmse.png", dpi=300)
plt.close()
print("Saved: images/ml_study_rmse.png")

# ── MAE plots ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
make_plot(df_results_wl, 'wavelengths',   'Number of Wavelengths',
          'MAE', 'Study 1: Wavelength Reduction vs MAE', axes[0])
make_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'MAE', 'Study 2: Dataset Size vs MAE',         axes[1])
plt.tight_layout()
plt.savefig("images/ml_study_mae.png", dpi=300)
plt.close()
print("Saved: images/ml_study_mae.png")

# ── Combined 4-panel summary ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
make_plot(df_results_wl, 'wavelengths',   'Number of Wavelengths',
          'RMSE', 'Study 1 — Wavelength Reduction (RMSE)', axes[0, 0])
make_plot(df_results_wl, 'wavelengths',   'Number of Wavelengths',
          'MAE',  'Study 1 — Wavelength Reduction (MAE)',  axes[0, 1])
make_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'RMSE', 'Study 2 — Dataset Size Reduction (RMSE)', axes[1, 0])
make_plot(df_results_ds, 'n_thicknesses', 'Thickness Values per Material',
          'MAE',  'Study 2 — Dataset Size Reduction (MAE)',  axes[1, 1])
plt.suptitle("ML Model Comparison — Ellipsometric Thickness Prediction",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("images/ml_summary_4panel.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: images/ml_summary_4panel.png")

# ─────────────────────────────────────────────────────────────────
# PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════════════════════")
print("STUDY 1 — Best results (full 501 wavelengths)")
print("═══════════════════════════════════════════════════════════")
best_wl = df_results_wl[df_results_wl['wavelengths'] == 501][['model', 'RMSE', 'MAE']]
print(best_wl.to_string(index=False))

print("\n═══════════════════════════════════════════════════════════")
print("STUDY 2 — Best results (full 61 thicknesses)")
print("═══════════════════════════════════════════════════════════")
best_ds = df_results_ds[df_results_ds['n_thicknesses'] == 61][['model', 'RMSE', 'MAE']]
print(best_ds.to_string(index=False))

print("\nDone! All results and plots saved.")
