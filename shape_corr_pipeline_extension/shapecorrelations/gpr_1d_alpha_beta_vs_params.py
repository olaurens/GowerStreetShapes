#!/usr/bin/env python3
"""
GPR emulator (1D) for:
  Omega_m -> alpha, beta
  sigma8  -> alpha, beta

- One point per run (drop duplicates on "run")
- Uses measurement errors via alpha=yerr^2
- RBF + WhiteKernel with restarts
- LOOCV RMSE and plots saved to disk
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

CSV = "amplitude_table_veca_rmin10.0_noref_debiased.csv"   
OUT_PREFIX = "gpr_1d"                        # output files prefix
N_GRID = 200


def make_kernel():
# Kernel structure:
#   Constant * RBF  -> smooth cosmology dependence with free amplitude
#   WhiteKernel     -> absorbs any residual intrinsic scatter
    return (C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0)))


def gpr_fit_predict_1d(x_raw, y, yerr, x_label, y_label, tag):
    """
    Fit 1D GPR on (x_raw -> y) with heteroscedastic measurement noise yerr.
    Produces:
      - learned kernel printout
      - LOOCV RMSE
      - plot saved: {OUT_PREFIX}_{tag}.png and .pdf
    """
    x_raw = np.asarray(x_raw).reshape(-1, 1)
    y = np.asarray(y).astype(float)
    yerr = np.asarray(yerr).astype(float)


    # Scale x for numerical stability and interpretable length_scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(x_raw)

    kernel = make_kernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=yerr**2,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gpr.fit(Xs, y)
    print(f"\n[{tag}] Learned kernel: {gpr.kernel_}")

    # LOOCV
    loo = LeaveOneOut()
    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)

    for train, test in loo.split(Xs):
        m = GaussianProcessRegressor(
            kernel=kernel,
            alpha=yerr[train] ** 2,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=0,
        )
        m.fit(Xs[train], y[train])
        mu1, std1 = m.predict(Xs[test], return_std=True)
        pred[test[0]] = mu1[0]
        pred_std[test[0]] = std1[0]

    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"[{tag}] LOOCV RMSE: {rmse:.6f}")

    # Prediction grid (in *raw* x units for plotting)
    x_grid = np.linspace(x_raw.min(), x_raw.max(), N_GRID).reshape(-1, 1)
    x_grid_s = scaler.transform(x_grid)
    mu, std = gpr.predict(x_grid_s, return_std=True)

    # Plot
    plt.figure(figsize=(6, 4))
    #plt.errorbar(x_raw[:, 0], y, yerr=yerr, fmt=".", capsize=2, label="Simulation runs")
    plt.scatter(x_raw[:, 0], y, s=10, alpha=0.3, label="Simulation runs",zorder=1)
    plt.plot(x_grid[:, 0], mu, label="GPR mean",zorder=2)
    plt.fill_between(
        x_grid[:, 0],
        mu - 2 * std,
        mu + 2 * std,
        alpha=0.2,
        label=r"$\pm 2\sigma$",
        zorder=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    out_png = f"{OUT_PREFIX}_{tag}_noref.png"
    out_pdf = f"{OUT_PREFIX}_{tag}_noref.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    return rmse, gpr.kernel_


def main():
    df = pd.read_csv(CSV)

    # One per run 
    runs = (df.sort_values("run")
        .drop_duplicates(subset=["run"])
        .copy())

    # ---- Define the 4 experiments ----
    experiments = [
        # (x_col, y_col, yerr_col, x_label, y_label, tag)
        ("Omega_m", "alpha_fit", "err_alpha", r"$\Omega_m$", r"$\alpha$", "Omega_m__alpha"),
        ("Omega_m", "beta_fit",  "err_beta",  r"$\Omega_m$", r"$\beta$",  "Omega_m__beta"),
        ("sigma8",  "alpha_fit", "err_alpha", r"$\sigma_8$", r"$\alpha$", "sigma8__alpha"),
        ("sigma8",  "beta_fit",  "err_beta",  r"$\sigma_8$", r"$\beta$",  "sigma8__beta"),
    ]
    for x_col, y_col, e_col, xlab, ylab, tag in experiments:
        if x_col not in runs.columns or y_col not in runs.columns or e_col not in runs.columns:
            print(f"Skip {tag}: missing columns (need {x_col}, {y_col}, {e_col})")
            continue

        rmse, ker = gpr_fit_predict_1d(
            x_raw=runs[x_col].values,
            y=runs[y_col].values,
            yerr=runs[e_col].values,
            x_label=xlab,
            y_label=ylab,
            tag=tag,)

if __name__ == "__main__":
    main()

