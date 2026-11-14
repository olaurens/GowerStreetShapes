"""
Gaussian Process Regression (GPR) model for the dependence of IA beta on Omega_m.

This script builds a 1D Gaussian Process emulator for the fitted IA parameter
beta as a function of the cosmological matter density parameter Omega_m.
Eason Tian
Feb 2026
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("amplitude_table_veca_rmin10.0.csv")

# Keep one entry per simulation run (unique Omega_m)
runs = (
    df.sort_values("run")
      .drop_duplicates(subset=["run"])
      [["run", "Omega_m", "beta_fit", "err_beta"]]
)

# Input: Omega_m (shape: N x 1)
X = runs[["Omega_m"]].values

# Target: fitted IA beta parameter
y = runs["beta_fit"].values

# Measurement uncertainty on beta (used as noise variance)
yerr = runs["err_beta"].values




# Standardising Omega_m improves GP hyperparameter optimisation
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ------------------------------------------------------------
# Define Gaussian Process kernel
# ------------------------------------------------------------

# Kernel structure:
#   Constant * RBF  -> smooth cosmology dependence with free amplitude
#   WhiteKernel     -> absorbs any residual intrinsic scatter
kernel = (
    C(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
)

# Fit GPR model
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=yerr**2,          # known heteroscedastic noise
    normalize_y=True,       # subtract mean of beta before fitting
    n_restarts_optimizer=10 # reduce risk of local minima
)

gpr.fit(Xs, y)

print("Learned kernel:", gpr.kernel_)


# Leave-One-Out Cross-Validation 
loo = LeaveOneOut()
pred = np.zeros_like(y)
pred_std = np.zeros_like(y)

for train, test in loo.split(Xs):
    # Refit GP on N-1 points
    m = GaussianProcessRegressor(
        kernel=kernel,
        alpha=yerr[train]**2,
        normalize_y=True,
        n_restarts_optimizer=2
    )
    m.fit(Xs[train], y[train])

    # Predict left-out point
    mu, std = m.predict(Xs[test], return_std=True)
    pred[test[0]] = mu[0]
    pred_std[test[0]] = std[0]

rmse = np.sqrt(mean_squared_error(y, pred))
print("LOOCV RMSE:", rmse)

# Posterior prediction on a dense Omega_m grid
om_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
om_grid_s = scaler.transform(om_grid)

mu, std = gpr.predict(om_grid_s, return_std=True)


# Plot results
plt.figure(figsize=(6, 4))

# Observed beta values with measurement uncertainties
plt.errorbar(
    X[:, 0], y,
    yerr=yerr,
    fmt=".",
    capsize=2,
    label="Simulation runs"
)

# GP posterior mean
plt.plot(om_grid[:, 0], mu, label="GPR mean")

# 2-sigma uncertainty band
plt.fill_between(
    om_grid[:, 0],
    mu - 2 * std,
    mu + 2 * std,
    alpha=0.2,
    label=r"$\pm 2\sigma$"
)

plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\beta$")
plt.legend()
plt.tight_layout()
plt.savefig("gpr_beta_vs_omega_m.png", dpi=300)
plt.show()
