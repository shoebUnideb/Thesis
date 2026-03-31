import pandas as pd
import numpy as np

# -----------------------------
# LOAD ORIGINAL DATASET
# -----------------------------
df = pd.read_csv("ellipsometry_dataset.csv")

# -----------------------------
# FUNCTION: WRAP DELTA TO [-180, 180]
# -----------------------------
def wrap_phase(delta):
    return (delta + 180) % 360 - 180

# -----------------------------
# PARAMETERS
# -----------------------------
sigma = 0.5
percent = 0.01
range_uniform = 1.0
bias = 0.5

# -----------------------------
# BASE DATAFRAME
# -----------------------------
df_wide = df[["material", "thickness_nm", "wavelength_nm"]].copy()

# -----------------------------
# CLEAN
# -----------------------------
df_wide["psi"] = df["psi_deg"]
df_wide["delta"] = wrap_phase(df["delta_deg"])

# -----------------------------
# GAUSSIAN
# -----------------------------
psi_g = df["psi_deg"] + np.random.normal(0, sigma, len(df))
delta_g = wrap_phase(df["delta_deg"] + np.random.normal(0, sigma, len(df)))

df_wide["psi_with_gaussian_noise"] = psi_g
df_wide["delta_with_gaussian_noise"] = delta_g

# -----------------------------
# RELATIVE
# -----------------------------
psi_r = df["psi_deg"] * (1 + np.random.normal(0, percent, len(df)))
delta_r = wrap_phase(df["delta_deg"] * (1 + np.random.normal(0, percent, len(df))))

df_wide["psi_with_relative_noise"] = psi_r
df_wide["delta_with_relative_noise"] = delta_r

# -----------------------------
# UNIFORM
# -----------------------------
psi_u = df["psi_deg"] + np.random.uniform(-range_uniform, range_uniform, len(df))
delta_u = wrap_phase(df["delta_deg"] + np.random.uniform(-range_uniform, range_uniform, len(df)))

df_wide["psi_with_uniform_noise"] = psi_u
df_wide["delta_with_uniform_noise"] = delta_u

# -----------------------------
# BIAS
# -----------------------------
psi_b = df["psi_deg"] + bias
delta_b = wrap_phase(df["delta_deg"] + bias)

df_wide["psi_with_bias"] = psi_b
df_wide["delta_with_bias"] = delta_b

# -----------------------------
# MIXED (BEST)
# -----------------------------
psi_m = df["psi_deg"] + np.random.normal(0, sigma, len(df)) + bias
delta_m = wrap_phase(df["delta_deg"] + np.random.normal(0, sigma, len(df)) + bias)

df_wide["psi_with_mixed_noise"] = psi_m
df_wide["delta_with_mixed_noise"] = delta_m

# -----------------------------
# SAVE
# -----------------------------
df_wide.to_csv("dataset_combined_noise.csv", index=False)

print("Wide-format dataset created: dataset_combined_noise.csv")