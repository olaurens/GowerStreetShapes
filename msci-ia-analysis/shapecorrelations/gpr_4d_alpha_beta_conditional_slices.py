"""
4D Gaussian Process regression for IA parameters.

Models:
    (Omega_m, sigma8, Omega_b, little_h) -> alpha
    (Omega_m, sigma8, Omega_b, little_h) -> beta

Then produces conditional slices:
    beta(sigma8 | Omega_m = median, Omega_b = median, little_h = median)
    alpha(Omega_m | sigma8 = median, Omega_b = median, little_h = median)

Eason Tian
Feb 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load data (use the file WITH cosmo)
df = pd.read_csv("amplitude_table_veca_rmin10.0_noref_debiased.csv")


runs = (df.sort_values("run")
          .drop_duplicates(subset=["run"])
          [["run", "Omega_m", "sigma8", "Omega_b", "little_h", "alpha_fit", "beta_fit"]]
       )

# 4D features
X = runs[["Omega_m", "sigma8", "Omega_b", "little_h"]].values
alpha = runs["alpha_fit"].values
beta  = runs["beta_fit"].values

# Scaling
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Kernel (length_scale dimension -> 4)
kernel = C(1.0, (1e-3, 1e3)) * RBF(
    length_scale=[1.0, 1.0, 1.0, 1.0],
    length_scale_bounds=(1e-2, 1e2)
) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))

# Train GP models
gpr_alpha = GaussianProcessRegressor(kernel=kernel,
                                     normalize_y=True,
                                     n_restarts_optimizer=2)
gpr_alpha.fit(Xs, alpha)

gpr_beta = GaussianProcessRegressor(kernel=kernel,
                                    normalize_y=True,
                                    n_restarts_optimizer=2)
gpr_beta.fit(Xs, beta)

print("Alpha kernel:", gpr_alpha.kernel_)
print("Beta kernel:", gpr_beta.kernel_)

# Compute medians (now 4D)
om_median = np.median(X[:, 0])
s8_median = np.median(X[:, 1])
ob_median = np.median(X[:, 2])
h_median  = np.median(X[:, 3])

print("Median Omega_m =", om_median)
print("Median sigma8  =", s8_median)
print("Median Omega_b =", ob_median)
print("Median little_h =", h_median)

# Slice 1: beta(sigma8 | Omega_m = median, Omega_b = median, little_h = median)
s8_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
X_slice = np.column_stack([
    np.full_like(s8_grid, om_median),
    s8_grid,
    np.full_like(s8_grid, ob_median),
    np.full_like(s8_grid, h_median),
])
X_slice_s = scaler.transform(X_slice)

mu_beta, std_beta = gpr_beta.predict(X_slice_s, return_std=True)
plt.figure(figsize=(6,4))
plt.plot(s8_grid, mu_beta, label="GP mean")
plt.fill_between(s8_grid,
                 mu_beta - 2*std_beta,
                 mu_beta + 2*std_beta,
                 alpha=0.2,
                 label=r"$\pm 2\sigma$")
plt.xlabel(r"$\sigma_8$")
plt.ylabel(r"$\beta$")
plt.title(rf"$\beta(\sigma_8)$ at $\Omega_m={om_median:.3f}$, $\Omega_b={ob_median:.3f}$, $h={h_median:.3f}$")
plt.legend()
plt.tight_layout()
plt.savefig("gpr_4d_beta_vs_sigma8_at_median_Omega_m_Omegab_h_noref.png", dpi=300)
plt.close()

# Slice 2: alpha(Omega_m | sigma8 = median, Omega_b = median, little_h = median)
om_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
X_slice = np.column_stack([
    om_grid,
    np.full_like(om_grid, s8_median),
    np.full_like(om_grid, ob_median),
    np.full_like(om_grid, h_median),
])
X_slice_s = scaler.transform(X_slice)

mu_alpha, std_alpha = gpr_alpha.predict(X_slice_s, return_std=True)

plt.figure(figsize=(6,4))
plt.plot(om_grid, mu_alpha, label="GP mean")
plt.fill_between(om_grid,
                 mu_alpha - 2*std_alpha,
                 mu_alpha + 2*std_alpha,
                 alpha=0.2,
                 label=r"$\pm 2\sigma$")
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\alpha$")
plt.title(rf"$\alpha(\Omega_m)$ at $\sigma_8={s8_median:.3f}$, $\Omega_b={ob_median:.3f}$, $h={h_median:.3f}$")
plt.legend()
plt.tight_layout()
plt.savefig("gpr_4d_alpha_vs_Omega_m_at_median_sigma8_Omegab_h_noref.png", dpi=300)
plt.close()

print("\nSaved:")
print("  gpr_4d_beta_vs_sigma8_at_median_Omega_m_Omegab_h_noref.png")
print("  gpr_4d_alpha_vs_Omega_m_at_median_sigma8_Omegab_h_noref.png")

# Slice 3: alpha(sigma8 | Omega_m = median, Omega_b = median, little_h = median)
s8_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
X_slice = np.column_stack([
    np.full_like(s8_grid, om_median),
    s8_grid,
    np.full_like(s8_grid, ob_median),
    np.full_like(s8_grid, h_median),
])
X_slice_s = scaler.transform(X_slice)

mu_alpha, std_alpha = gpr_alpha.predict(X_slice_s, return_std=True)

plt.figure(figsize=(6,4))
plt.plot(s8_grid, mu_alpha, label="GP mean")
plt.fill_between(s8_grid,
                 mu_alpha - 2*std_alpha,
                 mu_alpha + 2*std_alpha,
                 alpha=0.2,
                 label=r"$\pm 2\sigma$")
plt.xlabel(r"$\sigma_8$")
plt.ylabel(r"$\alpha$")
plt.title(rf"$\alpha(\sigma_8)$ at $\Omega_m={om_median:.3f}$, $\Omega_b={ob_median:.3f}$, $h={h_median:.3f}$")
plt.legend()
plt.tight_layout()
plt.savefig("gpr_4d_alpha_vs_sigma8_at_median_Omega_m_Omegab_h_noref.png", dpi=300)
plt.close()

# Slice 4: beta(Omega_m | sigma8 = median, Omega_b = median, little_h = median)
om_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
X_slice = np.column_stack([
    om_grid,
    np.full_like(om_grid, s8_median),
    np.full_like(om_grid, ob_median),
    np.full_like(om_grid, h_median),
])
X_slice_s = scaler.transform(X_slice)

mu_beta, std_beta = gpr_beta.predict(X_slice_s, return_std=True)

plt.figure(figsize=(6,4))
plt.plot(om_grid, mu_beta, label="GP mean")
plt.fill_between(om_grid,
                 mu_beta - 2*std_beta,
                 mu_beta + 2*std_beta,
                 alpha=0.2,
                 label=r"$\pm 2\sigma$")
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\beta$")
plt.title(rf"$\beta(\Omega_m)$ at $\sigma_8={s8_median:.3f}$, $\Omega_b={ob_median:.3f}$, $h={h_median:.3f}$")
plt.legend()
plt.tight_layout()
plt.savefig("gpr_4d_beta_vs_Omega_m_at_median_sigma8_Omegab_h_noref.png", dpi=300)
plt.close()


def loocv_rmse(Xs, y, kernel):
    """
    Compute Leave-One-Out Cross-Validation (LOOCV) RMSE
    for a Gaussian Process model.
    """
    loo = LeaveOneOut()
    preds = np.zeros_like(y)

    for train, test in loo.split(Xs):
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2
        )
        model.fit(Xs[train], y[train])
        mu = model.predict(Xs[test])
        preds[test[0]] = mu[0]

    return np.sqrt(mean_squared_error(y, preds))


# Compute 4D RMSE
rmse_alpha_4d = loocv_rmse(Xs, alpha, kernel)
rmse_beta_4d  = loocv_rmse(Xs, beta, kernel)

print("\n4D LOOCV RMSE:")
print("alpha:", rmse_alpha_4d)
print("beta :", rmse_beta_4d)
