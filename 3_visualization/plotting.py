import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ellipsometry_dataset.csv")

materials = ["Si", "GaAs", "GaN", "Ge", "InP"]
thickness_list = [20, 40, 60, 80]

for material in materials:

    df_mat = df[df["material"] == material]

    fig, axs = plt.subplots(2, 1, sharex=True)

    # -----------------------------
    # PSI (top)
    # -----------------------------
    for d in thickness_list:
        subset = df_mat[df_mat["thickness_nm"] == d]
        axs[0].plot(subset["wavelength_nm"], subset["psi_deg"], label=f"{d} nm")

    axs[0].set_ylabel("Psi (degrees)")
    axs[0].set_title(f"{material}: Ellipsometric Parameters vs Wavelength")
    axs[0].legend()
    axs[0].grid()

    # -----------------------------
    # DELTA (bottom)
    # -----------------------------
    for d in thickness_list:
        subset = df_mat[df_mat["thickness_nm"] == d]
        axs[1].plot(subset["wavelength_nm"], subset["delta_deg"], label=f"{d} nm")

    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Delta (degrees)")
    axs[1].grid()

    # Save each figure
    filename = f"{material}_psi_delta.png"
    plt.savefig(filename, dpi=300)

    plt.close()

print("All material plots generated!")