import numpy as np
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
theta0_deg = 70
theta0 = np.radians(theta0_deg)

wavelengths = np.arange(300, 801, 1)  # nm
thicknesses = np.arange(20, 81, 1)    # nm

# File paths
files = {
    "Si": "si_nk_300_800nm_1nm.csv",
    "GaAs": "gaas_nk_300_800nm_1nm_final.csv",
    "GaN": "gan_nk_300_800nm_1nm.csv",
    "Ge": "ge_nk_300_800nm_1nm.csv",
    "InP": "inp_nk_300_800nm_1nm.csv"
}

sio2_file = "sio2_sellmeier_300_800nm_1nm.csv"

# -----------------------------
# LOAD SiO2 (substrate)
# -----------------------------
sio2_df = pd.read_csv(sio2_file)
n_sio2 = sio2_df["n"].values
k_sio2 = np.zeros_like(n_sio2)
n2 = n_sio2 - 1j * k_sio2

# Ambient (air)
n0 = 1.0 + 0j

# -----------------------------
# FUNCTIONS
# -----------------------------
def snell(n0, n1, theta0):
    return np.arcsin(n0 * np.sin(theta0) / n1)

def fresnel_rs(n_i, n_j, theta_i, theta_j):
    return (n_i * np.cos(theta_i) - n_j * np.cos(theta_j)) / \
           (n_i * np.cos(theta_i) + n_j * np.cos(theta_j))

def fresnel_rp(n_i, n_j, theta_i, theta_j):
    return (n_j * np.cos(theta_i) - n_i * np.cos(theta_j)) / \
           (n_j * np.cos(theta_i) + n_i * np.cos(theta_j))

def compute_psi_delta(rp, rs):
    rho = rp / rs
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)
    return np.degrees(psi), np.degrees(delta)

# -----------------------------
# MAIN SIMULATION
# -----------------------------
results = []

for material, file in files.items():
    df = pd.read_csv(file)
    n = df["n"].values
    k = df["k"].values

    n1_all = n - 1j * k

    for d in thicknesses:
        for i, wl in enumerate(wavelengths):

            n1 = n1_all[i]
            n2_i = n2[i]

            # Angles
            theta1 = snell(n0, n1, theta0)
            theta2 = snell(n1, n2_i, theta1)

            # Fresnel coefficients
            r01_s = fresnel_rs(n0, n1, theta0, theta1)
            r12_s = fresnel_rs(n1, n2_i, theta1, theta2)

            r01_p = fresnel_rp(n0, n1, theta0, theta1)
            r12_p = fresnel_rp(n1, n2_i, theta1, theta2)

            # Phase thickness
            beta = (2 * np.pi / wl) * n1 * d * np.cos(theta1)

            # Total reflection (thin film interference)
            exp_term = np.exp(-2j * beta)

            rs = (r01_s + r12_s * exp_term) / (1 + r01_s * r12_s * exp_term)
            rp = (r01_p + r12_p * exp_term) / (1 + r01_p * r12_p * exp_term)

            # Psi, Delta
            psi, delta = compute_psi_delta(rp, rs)

            results.append([material, d, wl, psi, delta])

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_df = pd.DataFrame(results, columns=[
    "material", "thickness_nm", "wavelength_nm", "psi_deg", "delta_deg"
])

output_df.to_csv("ellipsometry_dataset.csv", index=False)

print("Simulation complete. File saved as ellipsometry_dataset.csv")