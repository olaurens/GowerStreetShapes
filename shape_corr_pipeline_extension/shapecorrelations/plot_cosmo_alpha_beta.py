"""
Plot cosmology dependence of alpha and beta.

Example Input File:
  amplitude_table_veca_rmin10.0_noref_debiased.csv

Eason Tian

January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Load table


#df = pd.read_csv("amplitude_table_veca_rmin10.csv")
#df = pd.read_csv("amplitude_table_veca_rmin10.0_noref_debiased.csv")
df = pd.read_csv("amplitude_table_veca_rmin10.0_noref.csv")


# Group by cosmology parameters (run-level)
possible_group_cols = ["Omega_m","sigma8","Omega_b","little_h","n_s","w0","m_nu"]
group_cols = [c for c in possible_group_cols if c in df.columns]

cosmo = (df.groupby(group_cols, as_index=False).agg(
          alpha=("alpha_fit","mean"),
          beta=("beta_fit","mean"),
          ealpha=("err_alpha","mean"),
          ebeta=("err_beta","mean"),
          N=("alpha_fit","size")))

print("Number of cosmology points:", len(cosmo))



# plotting function
def plot_map(data, x, y, c=None, cmap="viridis"):

    latex_map = {
        "Omega_m": r"\Omega_m",
        "sigma8": r"\sigma_8",
        "w0": r"w_0",
        "wa": r"w_a",
        "alpha": r"\alpha",
        "beta": r"\beta",
    }

    def tex(var):
        return latex_map.get(var, var)

    plt.figure(figsize=(6,5))

    if c is None:

        plt.scatter(data[x], data[y], s=120, edgecolor="k", alpha=0.85)

        plt.xlabel(rf"${tex(x)}$")
        plt.ylabel(rf"${tex(y)}$")

        plt.title(rf"${tex(y)}\ \mathrm{{vs}}\ {tex(x)}$")

        #outname = f"{y}_vs_{x}_noref_debiased.png"
        outname = f"{y}_vs_{x}_noref.png"
    else:

        sc = plt.scatter(
            data[x],
            data[y],
            c=data[c],
            s=120,
            cmap=cmap,
            edgecolor="k")

        plt.xlabel(rf"${tex(x)}$")
        plt.ylabel(rf"${tex(y)}$")

        cbar = plt.colorbar(sc)
        cbar.set_label(rf"${tex(c)}$")

        plt.title(rf"${tex(c)}({tex(x)}, {tex(y)})$")

        #outname = f"{c}_on_{x}_{y}_noref_debiased.png"
        outname = f"{c}_on_{x}_{y}_noref.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    print("Saved:", outname)

# 2D cosmology maps
plot_map(cosmo,"Omega_m","sigma8","alpha")

plot_map(cosmo,"Omega_m","sigma8","beta",cmap="plasma")

plot_map(cosmo,"w0","sigma8","alpha")

plot_map(cosmo,"w0","Omega_m","alpha")

plot_map(cosmo,"w0","sigma8","beta",cmap="plasma")


# Direct parameter relations
plot_map(cosmo,"w0","beta")

plot_map(cosmo,"sigma8","alpha")

plot_map(cosmo,"Omega_m","beta")
