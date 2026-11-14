"""
Validate GP emulator by reconstructing A(z, M) for a withheld run.

Workflow
--------
1. Train GP emulators for the run-level parameters (logA0, alpha, beta).
2. Leave one simulation run out of the training set.
3. Predict its parameters using the GP emulator.
4. Reconstruct the amplitude A(z, M) from the predicted parameters.
5. Compare reconstructed amplitudes to the true simulation values.

This provides a simple leave-one-run-out validation of the emulator.

Eason Tian
March 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# User settings
RUN_TEST = 243          # run withheld from training
M0 = 1e13               # pivot mass used in amplitude model

GP_TABLE = "gp_run_level_table.csv"
AMP_TABLE = "amplitude_table_veca_rmin10.0_noref_debiased.csv"

# Cosmological parameters used as GP inputs
feat_cols = ["Omega_m", "sigma8", "Omega_b", "little_h", "n_s", "w0", "m_nu"]


def make_gp(n_dim):
    """
    Construct a Gaussian Process model with an RBF kernel.

    Kernel structure:
        Constant * RBF(length scales per dimension) + White noise

    This allows smooth interpolation in cosmological parameter space.
    """
    kernel = (C(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0)))

    return GaussianProcessRegressor(kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42)


def plot_pred_vs_true_logA(test_amp, run_test, out_png):
    """
    Scatter plot comparing predicted vs true logA values.

    Each point corresponds to one (z, M) amplitude measurement.
    The dashed line indicates perfect prediction (y = x).
    """
    plt.figure(figsize=(5.5, 5))

    #sigma_logA = test_amp["sigma_A"] / test_amp["A"]

    plt.errorbar(test_amp["logA_true"],
             test_amp["logA_pred"],
             #xerr=sigma_logA,
             xerr=test_amp["sigma_logA_data"],
             yerr=test_amp["sigma_logA_gp"],
             fmt='o',
             ms=6,
             alpha=0.8,
             capsize=2)

    # Determine range for the reference line
    lo = min(test_amp["logA_true"].min(), test_amp["logA_pred"].min())
    hi = max(test_amp["logA_true"].max(), test_amp["logA_pred"].max())

    # Perfect prediction reference line
    plt.plot([lo, hi], [lo, hi], "--", lw=1.5)

    plt.xlabel(r"True $\log A$")
    plt.ylabel(r"Predicted $\log A$")
    plt.title(f"Withheld run {run_test}: predicted vs true")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:", out_png)


def plot_reconstructed_logA_vs_log1pz(test_amp, logA0_pred, alpha_pred, beta_pred,
        run_test, which_vec, rmin, out_png):
    """
    Plot reconstructed amplitude evolution vs redshift.

    True amplitudes from simulation are shown as scatter points,
    while GP predictions are plotted as smooth curves.

    Each curve corresponds to a different halo mass bin.
    """
    d = test_amp.copy()

    # Derived quantities used in the amplitude model
    d["log1pz"] = np.log(1.0 + d["z"])
    d["logA_true"] = np.log(d["A"])
    d["logM"] = np.log(d["M_mid"] / M0)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Smooth redshift grid for plotting prediction curves
    z_grid = np.linspace(d["z"].min(), d["z"].max(), 300)
    log1pz_grid = np.log(1.0 + z_grid)

    for mass_bin, g in d.groupby("mass_bin"):
        sigma_logA = g["sigma_A"] / g["A"]
        # True measurements with error bars
        ax.errorbar(g["log1pz"],
                g["logA_true"],
                yerr=sigma_logA,
                fmt='o',
                ms=5,
                alpha=0.85,
                capsize=2,
                label=f"True: {mass_bin}")

        # Predicted model curve
        logM_bin = np.median(g["logM"])
        logA_model = (logA0_pred
            + alpha_pred * log1pz_grid
            + beta_pred * logM_bin)

        ax.plot(log1pz_grid,
            logA_model,
            lw=2,
            label=f"Pred: {mass_bin}")

    ax.set_xlabel(r"$\log(1+z)$")
    ax.set_ylabel(r"$\log A$")
    ax.set_title(f"Reconstructed amplitude (vec {which_vec}) "
        f"- run {run_test} | rmin={rmin}")

    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:", out_png)

def plot_logA_vs_logM(test_amp, logA0_pred, alpha_pred, beta_pred,
                     run_test, which_vec, rmin, out_png):
    """
    Plot amplitude dependence on halo mass.

    True amplitudes are shown as scatter points,
    predicted scaling from the model is shown as lines.
    """

    d = test_amp.copy()

    d["log1pz"] = np.log(1.0 + d["z"])
    d["logA_true"] = np.log(d["A"])
    d["logM"] = np.log(d["M_mid"] / M0)

    fig, ax = plt.subplots(figsize=(6,5))

    M_grid = np.linspace(d["logM"].min(), d["logM"].max(), 200)
    # Define a small horizontal offset for each redshift slice
    # This is only for display, to separate overlapping points/lines
    z_groups = sorted(d["z"].unique())
    dx_step = 0.1
    dx_map = {z: i * dx_step for i, z in enumerate(z_groups)}

    for z_val, g in d.groupby("z"):

        sigma_logA = g["sigma_A"] / g["A"]
        # Apply the same offset to all points in this redshift slice
        dx = dx_map[z_val]
        ax.errorbar(g["logM"]+dx,
            g["logA_true"],
            yerr=sigma_logA,
            fmt='o',
            ms=6,
            alpha=0.85,
            capsize=2,
            label=f"True: z={z_val:.2f}")

        log1pz_val = np.log(1.0 + z_val)

        logA_model = (logA0_pred
            + alpha_pred * log1pz_val
            + beta_pred * M_grid)

        ax.plot(M_grid+dx,
                logA_model,
                lw=2,
                label=f"Pred: z={z_val:.2f}")

    ax.set_xlabel(r"$\log(M/M_0)$")
    ax.set_ylabel(r"$\log A$")
    ax.set_title(f"Mass scaling (vec {which_vec}) - run {run_test} | rmin={rmin}")

    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:", out_png)



def plot_residuals(test_amp, run_test, out_png):
    """
    Plot residual diagnostics with error bars.
    """

    d = test_amp.copy()

    d["log1pz"] = np.log(1.0 + d["z"])
    d["logM"] = np.log(d["M_mid"] / M0)

    yerr = d["sigma_logA_tot"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # =========================
    # residual vs redshift
    # =========================
    axes[0].errorbar(d["log1pz"],d["dlogA"],yerr=yerr,fmt='o',ms=5,alpha=0.8,capsize=2)
    axes[0].axhline(0, ls="--", lw=1)
    axes[0].set_xlabel(r"$\log(1+z)$")
    axes[0].set_ylabel(r"$\Delta \log A$")
    axes[0].set_title("Residual vs redshift")

    # =========================
    # residual vs mass
    # =========================
    axes[1].errorbar(d["logM"],d["dlogA"],yerr=yerr,fmt='o',ms=5,alpha=0.8,capsize=2)
    axes[1].axhline(0, ls="--", lw=1)
    axes[1].set_xlabel(r"$\log(M/M_0)$")
    axes[1].set_ylabel(r"$\Delta \log A$")
    axes[1].set_title("Residual vs mass")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:", out_png)    
# Load tables
df_gp = pd.read_csv(GP_TABLE)
df_amp = pd.read_csv(AMP_TABLE)

# Ensure one row per run in GP table
df_gp = df_gp.drop_duplicates(subset=["run"]).copy()

# Keep only physically valid amplitude measurements
df_amp = df_amp[np.isfinite(df_amp["A"]) & (df_amp["A"] > 0)].copy()
df_amp = df_amp[np.isfinite(df_amp["z"]) & np.isfinite(df_amp["M_mid"])].copy()

print("GP runs available:", len(df_gp))
print("Amplitude rows available:", len(df_amp))


# Split train / test
train_gp = df_gp[df_gp["run"] != RUN_TEST].copy()
test_gp = df_gp[df_gp["run"] == RUN_TEST].copy()

test_amp = df_amp[df_amp["run"] == RUN_TEST].copy()

print("Training runs:", len(train_gp))
print("Withheld run:", RUN_TEST)
print("Withheld amplitude rows:", len(test_amp))



# Build training arrays
X_train = train_gp[feat_cols].to_numpy()
X_test = test_gp[feat_cols].to_numpy()

y_logA0_train = train_gp["logA0"].to_numpy()
y_alpha_train = train_gp["alpha_fit"].to_numpy()
y_beta_train = train_gp["beta_fit"].to_numpy()

# Standardise cosmological parameters before GP training
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)



# Train GPs
n_dim = len(feat_cols)

gpr_logA0 = make_gp(n_dim)
gpr_alpha = make_gp(n_dim)
gpr_beta = make_gp(n_dim)

gpr_logA0.fit(X_train_s, y_logA0_train)
gpr_alpha.fit(X_train_s, y_alpha_train)
gpr_beta.fit(X_train_s, y_beta_train)

print("logA0 kernel:", gpr_logA0.kernel_)
print("alpha kernel:", gpr_alpha.kernel_)
print("beta kernel :", gpr_beta.kernel_)



# Predict withheld run parameters
logA0_pred, logA0_std = gpr_logA0.predict(X_test_s, return_std=True)
alpha_pred, alpha_std = gpr_alpha.predict(X_test_s, return_std=True)
beta_pred, beta_std = gpr_beta.predict(X_test_s, return_std=True)

# Convert arrays to scalars
logA0_pred = float(logA0_pred[0])
alpha_pred = float(alpha_pred[0])
beta_pred = float(beta_pred[0])

logA0_std = float(logA0_std[0])
alpha_std = float(alpha_std[0])
beta_std = float(beta_std[0])

# True parameters of the withheld run
logA0_true = float(test_gp["logA0"].iloc[0])
alpha_true = float(test_gp["alpha_fit"].iloc[0])
beta_true = float(test_gp["beta_fit"].iloc[0])

print("\nPredicted parameters for withheld run:")
print(f"  logA0 = {logA0_pred:.6f} +/- {logA0_std:.6f}   (true {logA0_true:.6f})")
print(f"  alpha = {alpha_pred:.6f} +/- {alpha_std:.6f}   (true {alpha_true:.6f})")
print(f"  beta  = {beta_pred:.6f} +/- {beta_std:.6f}   (true {beta_true:.6f})")


# Reconstruct A(z, M)
test_amp = test_amp.copy()

# Derived variables used in the amplitude model
test_amp["log1pz"] = np.log(1.0 + test_amp["z"].to_numpy())
test_amp["logM"] = np.log(test_amp["M_mid"].to_numpy() / M0)
test_amp["logA_true"] = np.log(test_amp["A"].to_numpy())

# Predicted amplitude from GP parameters
test_amp["logA_pred"] = (logA0_pred
    + alpha_pred * test_amp["log1pz"]
    + beta_pred * test_amp["logM"])

test_amp["A_pred"] = np.exp(test_amp["logA_pred"])

# Errors
test_amp["dlogA"] = test_amp["logA_pred"] - test_amp["logA_true"]
test_amp["frac_err_A"] = ((test_amp["A_pred"] - test_amp["A"]) / test_amp["A"])



# Metrics
rmse_logA = np.sqrt(mean_squared_error(test_amp["logA_true"], test_amp["logA_pred"]))

bias_logA = np.mean(test_amp["dlogA"])
med_abs_fracA = np.median(np.abs(test_amp["frac_err_A"]))

print("\nAmplitude reconstruction metrics:")
print(f"  RMSE(logA)      = {rmse_logA:.6f}")
print(f"  Bias(logA)      = {bias_logA:.6f}")
print(f"  Median |deltaA/A|   = {med_abs_fracA:.6f}")


print("\nPer-point comparison:")
print(test_amp[["run", "z", "mass_bin", "M_mid", "A", "A_pred",
     "logA_true", "logA_pred", "dlogA", "frac_err_A"]].to_string(index=False))

# =========================================================
# Total uncertainty in logA: data error + GP model error
# =========================================================

# 1) measurement uncertainty from amplitude extraction
sigma_logA_data = test_amp["sigma_A"] / test_amp["A"]
sigma_logA_data = np.clip(sigma_logA_data, 1e-8, None)

# 2) GP emulator uncertainty propagated from (logA0, alpha, beta)
# logA = logA0 + alpha*log1pz + beta*logM
sigma_logA_gp = np.sqrt(
    logA0_std**2
    + (test_amp["log1pz"] * alpha_std)**2
    + (test_amp["logM"] * beta_std)**2
)

# 3) total uncertainty
sigma_logA_tot = np.sqrt(sigma_logA_data**2 + sigma_logA_gp**2)

# =========================================================
# chi2 / dof in log-space
# =========================================================
chi2 = np.sum((test_amp["dlogA"] / sigma_logA_tot)**2)
dof = len(test_amp) - 3   # parameters: logA0, alpha, beta
chi2_red = chi2 / dof

print("\nUncertainty budget:")
print(f"  median sigma_data(logA) = {np.median(sigma_logA_data):.6f}")
print(f"  median sigma_gp(logA)   = {np.median(sigma_logA_gp):.6f}")
print(f"  median sigma_tot(logA)  = {np.median(sigma_logA_tot):.6f}")

print("\nChi-squared:")
print(f"  chi2            = {chi2:.6f}")
print(f"  dof             = {dof:d}")
print(f"  chi2/dof        = {chi2_red:.6f}")

# optional: contribution from each point
test_amp["sigma_logA_data"] = sigma_logA_data
test_amp["sigma_logA_gp"] = sigma_logA_gp
test_amp["sigma_logA_tot"] = sigma_logA_tot
test_amp["chi2_contrib"] = (test_amp["dlogA"] / test_amp["sigma_logA_tot"])**2

# Plots
plot_pred_vs_true_logA(test_amp,
    run_test=RUN_TEST,
    out_png=f"gp_validate_pred_vs_true_logA_run{RUN_TEST}.png")

plot_reconstructed_logA_vs_log1pz(test_amp,
    logA0_pred=logA0_pred,
    alpha_pred=alpha_pred,
    beta_pred=beta_pred,
    run_test=RUN_TEST,
    which_vec="a",
    rmin=10.0,
    out_png=f"gp_validate_logA_vs_log1pz_run{RUN_TEST}.png")
plot_logA_vs_logM(test_amp,
    logA0_pred,
    alpha_pred,
    beta_pred,
    run_test=RUN_TEST,
    which_vec="a",
    rmin=10.0,
    out_png=f"gp_validate_logA_vs_logM_run{RUN_TEST}.png")

plot_residuals(test_amp,
    run_test=RUN_TEST,
    out_png=f"gp_validate_residuals_run{RUN_TEST}.png")
