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
sigma = 0.5        # Gaussian noise (degrees)
percent = 0.01     # Relative noise (1%)
range_uniform = 1.0  # Uniform noise ±1 degree
bias = 0.5         # Systematic bias (degrees)

# -----------------------------
# 1. GAUSSIAN NOISE
# -----------------------------
df_gaussian = df.copy()

df_gaussian["psi_deg"] += np.random.normal(0, sigma, len(df))
df_gaussian["delta_deg"] += np.random.normal(0, sigma, len(df))

df_gaussian["delta_deg"] = wrap_phase(df_gaussian["delta_deg"])

df_gaussian.to_csv("dataset_gaussian_noise.csv", index=False)

# -----------------------------
# 2. RELATIVE (PERCENTAGE) NOISE
# -----------------------------
df_relative = df.copy()

df_relative["psi_deg"] *= (1 + np.random.normal(0, percent, len(df)))
df_relative["delta_deg"] *= (1 + np.random.normal(0, percent, len(df)))

df_relative["delta_deg"] = wrap_phase(df_relative["delta_deg"])

df_relative.to_csv("dataset_relative_noise.csv", index=False)

# -----------------------------
# 3. UNIFORM NOISE
# -----------------------------
df_uniform = df.copy()

df_uniform["psi_deg"] += np.random.uniform(-range_uniform, range_uniform, len(df))
df_uniform["delta_deg"] += np.random.uniform(-range_uniform, range_uniform, len(df))

df_uniform["delta_deg"] = wrap_phase(df_uniform["delta_deg"])

df_uniform.to_csv("dataset_uniform_noise.csv", index=False)

# -----------------------------
# 4. SYSTEMATIC NOISE (BIAS)
# -----------------------------
df_bias = df.copy()

df_bias["psi_deg"] += bias
df_bias["delta_deg"] += bias

df_bias["delta_deg"] = wrap_phase(df_bias["delta_deg"])

df_bias.to_csv("dataset_bias_noise.csv", index=False)

# -----------------------------
# 5. MIXED NOISE (GAUSSIAN + BIAS)
# -----------------------------
df_mixed = df.copy()

df_mixed["psi_deg"] += np.random.normal(0, sigma, len(df)) + bias
df_mixed["delta_deg"] += np.random.normal(0, sigma, len(df)) + bias

df_mixed["delta_deg"] = wrap_phase(df_mixed["delta_deg"])

df_mixed.to_csv("dataset_mixed_noise.csv", index=False)

# -----------------------------
print("All noisy datasets created successfully with phase wrapping!")