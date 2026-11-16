import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====================================
# User settings
# ====================================
FIT_TABLE = "gp_run_level_table.csv"
AMP_TABLE = "amplitude_table_veca_rmin10.0_noref_debiased.csv"

OUT_TABLE = "gp_run_level_table_with_chi2.csv"
OUT_PNG = "chi2_red_map_Omega_m_sigma8.png"

M0 = 1.0e13   # pivot mass


# ====================================
# Load tables
# ====================================
df_fit = pd.read_csv(FIT_TABLE)
df_amp = pd.read_csv(AMP_TABLE)

print(f"Loaded fit table: {FIT_TABLE}, shape = {df_fit.shape}")
print(f"Loaded amplitude table: {AMP_TABLE}, shape = {df_amp.shape}")


# ====================================
# Check required columns
# ====================================
fit_required = [
    "run", "Omega_m", "sigma8",
    "logA0", "alpha_fit", "beta_fit"
]

amp_required = [
    "run", "z", "M_mid", "A", "sigma_A"
]

missing_fit = [c for c in fit_required if c not in df_fit.columns]
missing_amp = [c for c in amp_required if c not in df_amp.columns]

if missing_fit:
    raise ValueError(f"Missing required columns in fit table: {missing_fit}")
if missing_amp:
    raise ValueError(f"Missing required columns in amplitude table: {missing_amp}")


# ====================================
# Convert numeric columns
# ====================================
for c in fit_required:
    df_fit[c] = pd.to_numeric(df_fit[c], errors="coerce")

for c in amp_required:
    df_amp[c] = pd.to_numeric(df_amp[c], errors="coerce")


# Drop bad rows
df_fit = df_fit.dropna(subset=fit_required).copy()
df_amp = df_amp.dropna(subset=amp_required).copy()

# Keep only physical rows in amplitude table
df_amp = df_amp[(df_amp["A"] > 0) & (df_amp["sigma_A"] > 0)].copy()

print(f"Cleaned fit table shape = {df_fit.shape}")
print(f"Cleaned amplitude table shape = {df_amp.shape}")


# ====================================
# Compute chi2 per run using stored fit parameters
# Model:
# logA = logA0 + alpha_fit * log(1+z) + beta_fit * log(M/M0)
# ====================================
chi2_results = []

for _, row in df_fit.iterrows():
    run = row["run"]

    g = df_amp[df_amp["run"] == run].copy()

    # Need enough points to define chi2 with 3 fitted parameters
    if len(g) < 4:
        continue

    y = np.log(g["A"].to_numpy())
    x1 = np.log(1.0 + g["z"].to_numpy())
    x2 = np.log(g["M_mid"].to_numpy() / M0)

    sigma_logA = g["sigma_A"].to_numpy() / g["A"].to_numpy()

    # Model prediction from stored fit parameters
    y_model = (
        row["logA0"]
        + row["alpha_fit"] * x1
        + row["beta_fit"] * x2
    )

    # Remove non-finite rows just in case
    mask = (
        np.isfinite(y)
        & np.isfinite(y_model)
        & np.isfinite(sigma_logA)
        & (sigma_logA > 0)
    )

    y = y[mask]
    y_model = y_model[mask]
    sigma_logA = sigma_logA[mask]

    if len(y) < 4:
        continue

    resid = y - y_model
    chi2 = np.sum((resid / sigma_logA) ** 2)
    dof = len(y) - 3   # logA0, alpha, beta

    if dof <= 0:
        continue

    chi2_red = chi2 / dof

    chi2_results.append({
        "run": int(run),
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2_red,
        "N_points": len(y),
    })

df_chi2 = pd.DataFrame(chi2_results)

if len(df_chi2) == 0:
    raise RuntimeError("No valid chi2 results were computed. Please check your tables.")

print(f"Computed chi2 for {len(df_chi2)} runs.")


# ====================================
# Merge chi2 back to fit table
# ====================================
df_plot = df_fit.merge(df_chi2, on="run", how="left")

# Save output table
df_plot = df_plot.sort_values("run").reset_index(drop=True)
df_plot.to_csv(OUT_TABLE, index=False)

print(f"Saved output table to: {OUT_TABLE}")


# ====================================
# Summary
# ====================================
valid = df_plot["chi2_red"].dropna()

print("\nReduced chi2 summary:")
print(valid.describe())


# ====================================
# Plot reduced chi2 in (Omega_m, sigma8) plane
# ====================================
plt.figure(figsize=(7, 5.5))

sc = plt.scatter(
    df_plot["Omega_m"],
    df_plot["sigma8"],
    c=df_plot["chi2_red"],
    s=45,
)

plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\sigma_8$")
plt.title(r"Reduced $\chi^2$ of the scaling fit across cosmology")

cbar = plt.colorbar(sc)
cbar.set_label(r"Reduced $\chi^2$")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved figure to: {OUT_PNG}")

plt.show()
