"""
Plot amplitude response on the (Omega_m, sigma_8) plane.

Example Input File:
  amplitude_table_veca_rmin10.csv
Eason Tian

January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#in_csv = "amplitude_table_veca_rmin10.csv"
#in_csv = "amplitude_table_veca_rmin10.0_rebuilt.csv"
in_csv = "amplitude_table_veca_rmin10.0_noref_debiased.csv"
#out_png = "logA_on_Omegam_sigma8_plane_runmean_noref_debiased.png"
out_png = "logA_on_Omegam_sigma8_plane_runmean_noref.png"
df = pd.read_csv(in_csv)

# Aggregate to one value per run (robust + simple)
# Here we use median A across slices/mass bins for each run.
run_df = (
    df.groupby("run", as_index=False)
      .agg(Omega_m=("Omega_m", "first"),
           sigma8=("sigma8", "first"),
           A_med=("A", "median"),
           N=("A", "size"))
)


plt.figure(figsize=(6, 5))

sc = plt.scatter(
    run_df["Omega_m"],
    run_df["sigma8"],
    c=np.log10(run_df["A_med"]),
    s=90,
    cmap="viridis",
    edgecolor="k",
)

plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\sigma_8$")
cbar = plt.colorbar(sc)
cbar.set_label(r"$\log_{10} A_{\rm median}$")
plt.title(r"Amplitude $A$ on the $(\Omega_m,\sigma_8)$ plane (per-run median)")
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print("Saved:", out_png)
plt.show()

