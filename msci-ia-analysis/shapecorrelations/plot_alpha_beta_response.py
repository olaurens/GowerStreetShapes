import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_csv("amplitude_table_veca_rmin10.csv")
df = pd.read_csv("amplitude_table_veca_rmin10.0_noref_debiased.csv")
# Group by cosmology (run-level)
cosmo = (
    df.groupby(["Omega_m","sigma8"], as_index=False)
      .agg(
          alpha=("alpha_fit","mean"),
          beta=("beta_fit","mean"),
          ealpha=("err_alpha","mean"),
          ebeta=("err_beta","mean"),
          N=("alpha_fit","size")
      )
)

print("Number of cosmology points:", len(cosmo))

def plot_alpha():
  plt.figure(figsize=(6,5))

  sc = plt.scatter(
      cosmo["Omega_m"], cosmo["sigma8"],
      c=cosmo["alpha"],
      s=120, cmap="viridis",
      edgecolor="k")

  plt.xlabel(r"$\Omega_m$")
  plt.ylabel(r"$\sigma_8$")
  cbar = plt.colorbar(sc)
  cbar.set_label(r"$\alpha$")

  plt.title(r"Redshift scaling $\alpha(\Omega_m,\sigma_8)$")
  plt.tight_layout()
  plt.savefig("alpha_on_Omegam_sigma8.png", dpi=200)
  plt.show()
def plot_beta():
  plt.figure(figsize=(6,5))

  sc = plt.scatter(
      cosmo["Omega_m"], cosmo["sigma8"],
      c=cosmo["beta"],
      s=120, cmap="plasma",
      edgecolor="k")

  plt.xlabel(r"$\Omega_m$")
  plt.ylabel(r"$\sigma_8$")
  cbar = plt.colorbar(sc)
  cbar.set_label(r"$\beta$")

  plt.title(r"Mass scaling $\beta(\Omega_m,\sigma_8)$")
  plt.tight_layout()
  plt.savefig("beta_on_Omegam_sigma8.png", dpi=200)
  plt.show()
plot_alpha()
plot_beta()
