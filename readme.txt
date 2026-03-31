================================================================================
  THESIS: Analysis of Optical Measurements by Machine Learning and
          Analytical Models for Sensors
================================================================================
  Author  : Mohammad Shoeb
  Date    : March 2026
  Folder  : thesis/
================================================================================


FOLDER STRUCTURE
----------------

  thesis/
  ├── README.txt                        ← this file
  ├── 1_raw_data/
  │   ├── ellipsometry_dataset.csv      ← main simulated dataset (152,805 rows)
  │   ├── semiconductor_nk/             ← optical constants for 5 materials
  │   └── substrate/                    ← SiO₂ Sellmeier raw + processed
  ├── 2_simulation/
  │   ├── main.py                       ← Fresnel forward model
  │   ├── noise.py                      ← noise injection script
  │   ├── combined.py                   ← wide-format noise merger
  │   └── noise_datasets/               ← 6 noisy dataset CSVs
  ├── 3_visualization/
  │   ├── plotting.py                   ← spectrum plotting script
  │   └── plots/                        ← Ψ/Δ plots for 5 materials
  ├── 4_ml_initial/
  │   ├── ml.py                         ← initial 5-case ML sweep
  │   ├── ml_feature_matrix.csv         ← wide-format feature matrix
  │   ├── ml_results_wavelength.csv     ← Study 1 results (initial)
  │   ├── ml_results_dataset_size.csv   ← Study 2 results (initial)
  │   └── plots/                        ← summary plots from initial study
  ├── 5_ml_dense/
  │   ├── ml_dense.py                   ← dense sweep, single seed
  │   ├── ml_feature_matrix.csv
  │   ├── results_study1_wavelength.csv ← 91-case wavelength sweep (single run)
  │   ├── results_study2_dataset_size.csv ← 61-case size sweep (single run)
  │   └── plots/                        ← 24 plots (RMSE, MAE, MSE, R², zoomed, log)
  └── 6_ml_averaged/
      ├── ml_averaged.py                ← averaged sweep, 5 seeds
      ├── ml_feature_matrix.csv
      ├── results_study1_wavelength_avg.csv  ← 91 cases × mean±std
      ├── results_study2_dataset_size_avg.csv ← 61 cases × mean±std
      └── plots/                        ← 23 publication-quality plots with ±σ bands


================================================================================
  WHAT WAS DONE — FROM START TO END
================================================================================


PHYSICAL SETUP AND MOTIVATION

The goal of this work was to investigate whether machine learning can solve the
inverse problem in optical ellipsometry: given a measured spectrum of the two
ellipsometric angles Ψ (Psi) and Δ (Delta) across a range of wavelengths, can a
model accurately predict the thickness of a semiconductor thin film deposited on
a substrate? This is a well-known challenge in optical metrology because the
relationship between film thickness and the measured angles is highly nonlinear
and material-dependent.

The physical system modelled throughout is a standard 3-layer thin-film stack:
air (ambient, n=1) on top, a semiconductor thin film in the middle, and fused
silica SiO₂ as the substrate at the bottom. The angle of incidence was fixed at
70°, which is the standard working angle for spectroscopic ellipsometers. The
wavelength range covered was 300 to 800 nm in steps of 1 nm, giving 501
wavelength points per spectrum. Thickness values for the thin film ranged from
20 to 80 nm in 1 nm steps, giving 61 discrete thickness values per material.


OPTICAL CONSTANTS AND SUBSTRATE MODEL

Five semiconductor materials were selected as thin-film candidates: Silicon (Si),
Gallium Arsenide (GaAs), Indium Phosphide (InP), Germanium (Ge), and Gallium
Nitride (GaN). Their complex refractive indices ñ = n − ik were sourced from the
RefractiveIndex.INFO database (literature sources: Green 2008 for Si; Aspnes &
Studna 1983 for GaAs, InP, Ge; Kawashima 1997 for GaN). Each material file was
preprocessed to a uniform 1 nm wavelength grid from 300 to 800 nm using linear
interpolation, producing 501 (n, k) pairs per material. These files are stored in
1_raw_data/semiconductor_nk/.

The SiO₂ substrate was treated as non-absorbing (k = 0). Its refractive index was
computed analytically using the Sellmeier equation from Malitson (1965):
n²(λ) = 1 + Σ Bᵢλ² / (λ² − Cᵢ), where the Sellmeier coefficients were taken
from the raw file Malitson.csv (wavelengths in μm). After unit conversion and
sampling to the same 1 nm grid, the result was saved as
sio2_sellmeier_300_800nm_1nm.csv, found in 1_raw_data/substrate/.


ELLIPSOMETRY FORWARD MODEL (main.py)

The simulation of Ψ and Δ was implemented from scratch in Python using the
standard Fresnel thin-film interference formalism. The full pipeline for each
(material, thickness, wavelength) triplet proceeds as follows. Snell's law with
complex refractive indices is used to compute the refracted angle in each layer.
Fresnel reflection coefficients for both s- and p-polarization are computed at
each interface (air/film and film/substrate). The phase thickness β = (2π/λ) ×
ñ₁ × d × cos(θ₁) represents the round-trip phase accumulated inside the film.
The total reflection amplitude for each polarization is computed using the Airy
thin-film formula: r = (r₀₁ + r₁₂ e^(−2iβ)) / (1 + r₀₁ r₁₂ e^(−2iβ)). The
ellipsometric ratio ρ = r_p / r_s = tan(Ψ) e^(iΔ) then gives:
  Ψ = arctan(|r_p / r_s|)  in degrees
  Δ = arg(r_p / r_s)        in degrees

This was computed for all combinations of 5 materials × 61 thicknesses × 501
wavelengths, producing a dataset of 5 × 61 × 501 = 152,805 rows. Each row stores
five columns: material, thickness_nm, wavelength_nm, psi_deg, delta_deg. The
resulting file is 1_raw_data/ellipsometry_dataset.csv.


NOISE MODELLING (noise.py, combined.py)

To study the effect of measurement noise on the simulation data, five noise types
were implemented and applied to both Ψ and Δ in noise.py. After every noise
addition to Δ, phase wrapping was applied — (Δ + 180) % 360 − 180 — to keep
the wrapped phase within the physical range [−180°, +180°] and avoid unphysical
values.

The five noise models, all parametrized at levels consistent with real instrument
noise, are:

  Gaussian noise: additive white noise with standard deviation σ = 0.5°, applied
  independently to Ψ and Δ. This models random thermal and electronic noise in the
  detector.

  Relative (percentage) noise: each value is multiplied by (1 + N(0, 0.01²)),
  i.e. 1% relative noise. This models amplitude-dependent noise arising from
  calibration uncertainties.

  Uniform noise: additive noise drawn uniformly from [−1°, +1°]. This models
  bounded systematic-like fluctuations.

  Systematic bias: a constant offset of +0.5° added to both Ψ and Δ. This models
  a fixed instrument calibration error.

  Mixed noise: Gaussian (σ = 0.5°) plus systematic bias (+0.5°) applied
  simultaneously. This is considered the most realistic noise scenario for a real
  ellipsometer.

Each noise type was saved as a separate CSV in 2_simulation/noise_datasets/. A
sixth file, dataset_combined_noise.csv, was generated by combined.py. It merges
all five noisy versions into a single wide-format file with 152,805 rows and
15 columns, making it directly usable as a multi-noise ML dataset.


VISUALIZATION (plotting.py)

Spectral plots of Ψ and Δ versus wavelength were generated for each of the five
materials using representative thickness values of 20, 40, 60, and 80 nm. Each
figure uses two panels (Ψ on top, Δ on bottom), saved at 300 DPI for publication
quality. The five PNG files in 3_visualization/plots/ illustrate the widely
different spectral signatures across materials and demonstrate the complex
nonlinear dependence of both angles on wavelength and thickness — the key
challenge that motivates the machine learning approach.


MACHINE LEARNING PROBLEM FORMULATION

The ML task is a supervised regression problem. Each training sample corresponds
to a unique (material, thickness) pair. The input feature vector is formed by
concatenating the full Ψ spectrum and the full Δ spectrum:
  X = [Ψ(300), Ψ(301), ..., Ψ(800), Δ(300), Δ(301), ..., Δ(800)]

At full spectral resolution this gives 501 + 501 = 1002 features per sample. The
target output y is the scalar film thickness in nm. The long-format dataset
(152,805 rows) was pivoted into a wide-format feature matrix of 305 rows × 1002
feature columns (305 = 5 materials × 61 thicknesses). The feature matrix is
stored as ml_feature_matrix.csv in each ML study folder. The dataset was split
80% training / 20% testing with stratification by material to ensure all five
materials appear in both train and test sets.

Three regression models of increasing complexity were selected:
  - Linear Regression: no hyperparameters, no scaling; serves as a linearity
    baseline that is expected to fail on the strongly nonlinear optical response.
  - Random Forest: 100 trees, stratified random split, random_state=42, n_jobs=−1.
    Handles nonlinearity and feature interactions without scaling.
  - MLP (Multilayer Perceptron): architecture 256 → 128 → 64, ReLU activation,
    up to 2000 epochs with early stopping (10% validation fraction).
    StandardScaler applied to features (fit on train set only, applied to test).

Metrics reported for every experiment: RMSE (nm), MAE (nm), MSE (nm²), R²
(coefficient of determination). RMSE and MAE are in nanometres, directly
interpretable as the average thickness prediction error.


INITIAL ML STUDY — 5-CASE SWEEP (4_ml_initial/ml.py)

The first ML script (ml.py) ran a coarse parametric sweep at 5 discrete levels
for each of two studies. Both studies used a fixed 80/20 split with random_state=42.

Study 1 tested wavelength resolution reduction: the full 501-wavelength spectrum
was subsampled to 501, 250, 125, 75, and 50 wavelengths by evenly spaced indices.
At full resolution (501 wavelengths), Linear Regression achieved RMSE = 134.17 nm
and MAE = 89.34 nm — far too large to be useful, confirming the nonlinearity of
the problem. Random Forest achieved RMSE = 1.81 nm, MAE = 1.20 nm. MLP achieved
RMSE = 1.28 nm, MAE = 0.91 nm. Reducing to 50 wavelengths, Random Forest
maintained RMSE = 1.38 nm while MLP degraded slightly to RMSE = 2.06 nm, showing
that tree-based models are more robust to spectral subsampling at this scale.

Study 2 tested training dataset size by reducing the number of thickness values
from 61 down to 50, 40, 25, and 10 per material (subsampled evenly). At the
full 61-thickness dataset the results were identical to Study 1 (same samples,
same split). With only 10 thicknesses (50 total training samples), all three
models degraded significantly: LR RMSE = 14.91 nm, RF RMSE = 10.43 nm, MLP
RMSE = 10.81 nm. The MLP and Random Forest converged and both became unreliable,
indicating a minimum data requirement of roughly 25–40 thicknesses for stable
prediction (RF RMSE ≈ 3.7 nm at 25 thicknesses).

Results saved in ml_results_wavelength.csv and ml_results_dataset_size.csv.


DENSE ML STUDY — 91-CASE AND 61-CASE SWEEPS (5_ml_dense/ml_dense.py)

To obtain a complete picture of model performance across the full parameter range,
a dense sweep was run with a single fixed seed (random_state=42 for both split and
models). Study 1 covered 91 wavelength cases from 501 down to 50, stepping by 5
(sequence: 501, 495, 490, ..., 55, 50). Study 2 covered all 61 thickness cases
from 61 down to 1, stepping by 1. This resulted in (91 + 61) × 3 = 456 model
fits in total.

The dense sweep revealed that Linear Regression is highly unstable and shows
extreme RMSE spikes at certain wavelength counts (e.g. RMSE > 15,000 nm at 125
wavelengths), reflecting poorly conditioned feature matrices at those specific
subsampling patterns. Random Forest and MLP both showed stable, near-constant
RMSE in the 1.3–2.0 nm range across the entire 501-to-50 wavelength sweep, with
no clear degradation threshold, meaning even 50 wavelengths retains essentially
all predictive information for nonlinear models. In the dataset size sweep, RF and
MLP began to degrade noticeably below about 10–15 thickness values, where the
total training set drops below ~50 samples.

Full results in results_study1_wavelength.csv and results_study2_dataset_size.csv.
24 plots covering RMSE, MAE, MSE, R², log-scale, and zoomed views are in
5_ml_dense/plots/.


AVERAGED ML STUDY — 5-SEED AVERAGING (6_ml_averaged/ml_averaged.py)

To eliminate the effect of random seed choice on reported metrics, the same dense
sweep was repeated 5 times under different random seeds: 42, 0, 1, 7, 13. Each
seed controlled both the train/test split and the model initialisation (RF and MLP).
Results were aggregated as mean ± standard deviation across the 5 runs, yielding
statistically robust estimates. Total model fits: 91 × 3 × 5 + 61 × 3 × 5 = 2,280.

Study 1 results at full 501 wavelengths (mean ± std across 5 seeds):
  Linear Regression : RMSE = 161.21 ± 47.65 nm,  MAE = 105.25 ± 35.81 nm,  R² = −102.2
  Random Forest     : RMSE =   1.86 ±  0.26 nm,  MAE =   1.17 ±  0.11 nm,  R² = 0.9875
  MLP               : RMSE =   1.56 ±  0.38 nm,  MAE =   1.04 ±  0.18 nm,  R² = 0.9912

The large standard deviations for Linear Regression across seeds confirm that its
performance is not just poor but also highly sensitive to the particular data split,
while RF and MLP are stable (their std is small relative to mean). Both RF and MLP
maintain R² > 0.98 at all wavelength counts down to 50, confirming robust spectral
compression tolerance. For the dataset size study at the smallest dataset (1
thickness per material, 5 total samples), all models expectedly collapse, while at
25 thicknesses (125 total samples) RF achieves mean RMSE ≈ 1.6 nm and MLP ≈ 1.5 nm,
indicating very efficient data usage.

Plots include full metric sweeps, zoomed views, log-scale RMSE, and shaded ±1σ
error bands on all curves. The summary 8-panel figure (summary_best_8panel_avg.png)
shows all four metrics for RF and MLP side-by-side for both studies — these are
the publication-ready figures.

Full averaged results in results_study1_wavelength_avg.csv and
results_study2_dataset_size_avg.csv. 23 plots in 6_ml_averaged/plots/.


================================================================================
  KEY NUMERICAL RESULTS SUMMARY
================================================================================

At full spectral resolution (501 wavelengths, 61 thicknesses, 305 samples):
  Model              RMSE (nm)    MAE (nm)    R²
  Linear Regression  ~161         ~105        ~−102  (fails completely)
  Random Forest        1.86        1.17       0.9875
  MLP                  1.56        1.04       0.9912

Spectral compression tolerance (Random Forest, 501 → 50 wavelengths):
  RMSE remains in the range 1.3–2.0 nm across all 91 cases. No sharp drop-off.
  This means the most diagnostically relevant spectral information for thickness
  retrieval is already present in far fewer than 501 wavelength points.

Dataset size tolerance (MLP, averaged):
  At 25 thicknesses : RMSE ≈ 1.5 nm  (125 total samples, R² > 0.98)
  At 10 thicknesses : RMSE ≈ 3–5 nm  (50 total samples, moderate degradation)
  At  5 thicknesses : RMSE ≈ 6–10 nm (25 total samples, significant degradation)

Best overall model: MLP at full resolution, R² = 0.9912, RMSE = 1.56 ± 0.38 nm.

================================================================================
  FILE NOTES
================================================================================

All CSV files use comma separation. ellipsometry_dataset.csv is in long format
(one row per wavelength measurement). ml_feature_matrix.csv is in wide format
(one row per sample, one column per feature). Result CSVs in the averaged study
contain columns named RMSE_mean, RMSE_std, MAE_mean, MAE_std, MSE_mean, MSE_std,
R2_mean, R2_std for every (wavelength_case, model) combination. All plots are PNG
at 150–300 DPI. Scripts require numpy, pandas, scikit-learn, and matplotlib.
