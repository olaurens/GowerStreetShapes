import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


# =========================
# User settings
# =========================
TABLE = "amplitude_table_veca_rmin10.0_noref_debiased.csv"   
OUT_PREFIX = "gpr_7d"
N_GRID = 200

# 7D input
feat_cols = ["Omega_m", "sigma8", "Omega_b", "little_h", "n_s", "w0", "m_nu"]


def make_kernel(ndim):
    return (C(1.0, (1e-3, 1e3))
        * RBF(length_scale=[1.0] * ndim, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0)))


def fit_gp(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kernel = make_kernel(X.shape[1])

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,)
    gpr.fit(Xs, y)

    return gpr, scaler


def plot_1d_slice(gpr, scaler, med, feat_cols,
    vary_col, y_label, out_png):
    """
    Conditional 1D slice:
      vary one parameter, fix all others at median
    """
    x_min = df_fit[vary_col].min()
    x_max = df_fit[vary_col].max()
    x_grid = np.linspace(x_min, x_max, N_GRID)

    X_pred = np.tile(med.to_numpy(), (len(x_grid), 1))
    j = feat_cols.index(vary_col)
    X_pred[:, j] = x_grid

    X_pred_s = scaler.transform(X_pred)
    mu, std = gpr.predict(X_pred_s, return_std=True)

    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, mu, lw=2.5, label="GP mean")
    plt.fill_between(
        x_grid,
        mu - 2 * std,
        mu + 2 * std,
        alpha=0.15,
        label=r"$\pm 2\sigma$",
    )

    title = f"{y_label}({vary_col})"
    fixed_str = []
    for c in feat_cols:
        if c != vary_col:
            fixed_str.append(f"{c}={med[c]:.3f}")
    title += " at " + ", ".join(fixed_str)

    plt.title(title)
    plt.xlabel(vary_col)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)


# =========================
# Load table
# =========================
df_fit = pd.read_csv(TABLE)

#Identify unique runs and their w0 values
runs = df_fit[["run", "w0"]].drop_duplicates()

# Separate LCDM runs (w0 = -1) and others
lcdm_runs = runs[np.isclose(runs["w0"], -1.0)]["run"]
other_runs = runs[~np.isclose(runs["w0"], -1.0)]["run"]

# Randomly keep only a fraction of LCDM runs
lcdm_runs_sample = lcdm_runs.sample(frac=0.5, random_state=42)

# Combine sampled LCDM runs with all other runs
keep_runs = pd.concat([other_runs, lcdm_runs_sample])

# Keep only rows belonging to the selected runs
df_fit = df_fit[df_fit["run"].isin(keep_runs)]

# Keep one row per run, since alpha_fit and beta_fit
# are run-level quantities repeated across slice/mass_bin rows
df_fit = df_fit.drop_duplicates(subset=["run"]).copy()

X = df_fit[feat_cols].to_numpy()
alpha = df_fit["alpha_fit"].to_numpy()
beta = df_fit["beta_fit"].to_numpy()

print("Unique runs used for GP:", len(df_fit))
print(df_fit["w0"].value_counts().sort_index())

# =========================
# Fit GP
# =========================
gpr_alpha, scaler_alpha = fit_gp(X, alpha)
gpr_beta, scaler_beta = fit_gp(X, beta)

print("Alpha kernel:", gpr_alpha.kernel_)
print("Beta kernel:", gpr_beta.kernel_)

med = df_fit[feat_cols].median()
print("Median values:")
for c in feat_cols:
    print(f"  {c} = {med[c]}")

# =========================
# Example slices
# =========================
plot_1d_slice(gpr_beta, scaler_beta, med, feat_cols,
    vary_col="sigma8",
    y_label=r"$\beta$",
    out_png=f"{OUT_PREFIX}_beta_vs_sigma8_at_median_others.png")

plot_1d_slice(gpr_alpha, scaler_alpha, med, feat_cols,
    vary_col="Omega_m",
    y_label=r"$\alpha$",
    out_png=f"{OUT_PREFIX}_alpha_vs_Omega_m_at_median_others.png")

plot_1d_slice(gpr_alpha, scaler_alpha, med, feat_cols,
    vary_col="n_s",
    y_label=r"$\alpha$",
    out_png=f"{OUT_PREFIX}_alpha_vs_ns_at_median_others.png")

plot_1d_slice(gpr_beta, scaler_beta, med, feat_cols,
    vary_col="m_nu",
    y_label=r"$\beta$",
    out_png=f"{OUT_PREFIX}_beta_vs_mnu_at_median_others.png")
plot_1d_slice(gpr_alpha, scaler_alpha, med, feat_cols,
    vary_col="w0",
    y_label=r"$\alpha$",
    out_png=f"{OUT_PREFIX}_alpha_vs_w0_at_median_others.png")
plot_1d_slice(gpr_beta, scaler_beta, med, feat_cols,
    vary_col="w0",
    y_label=r"$\beta$",
    out_png=f"{OUT_PREFIX}_beta_vs_w0_at_median_others.png")
plot_1d_slice(gpr_alpha, scaler_alpha, med, feat_cols,
    vary_col="sigma8",
    y_label=r"$\alpha$",
    out_png=f"{OUT_PREFIX}_alpha_vs_sigma8_at_median_others.png")
plot_1d_slice(gpr_beta, scaler_beta, med, feat_cols,
    vary_col="Omega_m",
    y_label=r"$\beta$",
    out_png=f"{OUT_PREFIX}_beta_vs_Omega_m_at_median_others.png")
