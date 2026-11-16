"""
This script plot amplitude A depends on
cosmological parameter without first reducing each simulation
run to a single representative value.
Eason Tian
January 2026
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIX_MASSBIN = "7.5e+12-1.0e+13"
FIX_SLICE   = 80


df = pd.read_csv("amplitude_table_veca_rmin10.csv")

df = df[(df["mass_bin"] == FIX_MASSBIN) &
    (df["slice"] == FIX_SLICE) &
    (df["A"] > 0)]

plt.figure(figsize=(6,5))

c = np.log10(df["A"])
vmin, vmax = np.quantile(c, [0.05, 0.95])
print("vmin/vmax:", vmin, vmax)  
plt.title(rf"Relative IA amplitude on the $(\Omega_m,\sigma_8)$ plane "
    rf"(slice = {FIX_SLICE}, {FIX_MASSBIN})")

plt.tight_layout()

# filename with slice + massbin
out_png = f"logA_on_Omegam_sigma8_slice{FIX_SLICE}_{FIX_MASSBIN}.png"
sc = plt.scatter(df["Omega_m"], df["sigma8"],
                 c=c, vmin=vmin, vmax=vmax,
                 s=60, cmap="viridis", edgecolor="k", alpha=0.9)
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\sigma_8$")
cbar = plt.colorbar(sc)
cbar.set_label(r"$\log_{10}(A/A_{\rm ref})$")
plt.savefig(out_png, dpi=200)
print("Saved:", out_png)
plt.show()


