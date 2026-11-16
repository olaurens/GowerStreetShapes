#!/usr/bin/env python3
"""
Add a theoretical halo bias column to an existing amplitude table.

- Uses halomod.TracerHaloModel to compute b(M,z) in each mass bin.
- Computes number-weighted mean bias in the bin:
    <b> = \int b(M) (dn/dM) dM / \int (dn/dM) dM
- Assumes mass_bin values are in h^-1 Msun (Msun/h), consistent with many halo catalogues.
"""

import numpy as np
import pandas as pd
from halomod import TracerHaloModel

INFILE  = "amplitude_table_veca_rmin10.0_noref.csv"
#INFILE  = "amplitude_table_veca_rmin10.0.csv"
#INFILE  = "amplitude_table_veca_rmin10.0_with_cosmo.csv"
OUTFILE = "amplitude_table_veca_rmin10.0_noref_with_halobias.csv"
#OUTFILE = "amplitude_table_veca_rmin10.0_with_halobias.csv"

# Integration resolution inside each mass bin (dex)
DLOG10M_INTEGRATE = 0.01

def parse_mass_bin(mass_bin: str):
    """
    mass_bin like '1.0e+13-2.5e+13' -> (low, high) as floats
    """
    low, high = mass_bin.split("-")
    return float(low), float(high)

def bin_averaged_bias_halomod(
    mass_bin: str,
    z: float,
    Omega_m: float,
    Omega_b: float,
    little_h: float,
    sigma8: float,
    n_s: float,
    dlog10m: float = DLOG10M_INTEGRATE,):
    """
    Number-weighted mean halo bias in the mass bin using halomod.

    """
    m1, m2 = parse_mass_bin(mass_bin)  # h^-1 Msun = Msun/h
    logm1, logm2 = np.log10(m1), np.log10(m2)

    hm = TracerHaloModel(
        z=float(z),
        Mmin=float(logm1),
        Mmax=float(logm2),
        dlog10m=float(dlog10m),
        cosmo_params={
            "Om0": float(Omega_m),
            "Ob0": float(Omega_b),
            "H0": 100.0 * float(little_h),
        },
        sigma_8=float(sigma8),
        n=float(n_s),
    )
    # Number-weighted mean halo bias in a mass bin using halomod:

    m = hm.m
    dndm = hm.dndm

    # nu array location differs a bit by version
    nu = hm.nu if hasattr(hm, "nu") else hm.hmf.nu
    # hm.bias is a model object (e.g. Tinker10); evaluate it on nu to get b(M)
    b = hm.bias.bias()
    #  <b> = \int b(M) (dn/dM) dM / \int (dn/dM) dM
    return float(np.trapz(b * dndm, m) / np.trapz(dndm, m))

def main():
    df = pd.read_csv(INFILE)

    # Optional: exclude very high-redshift snapshots if you want
    if "z" in df.columns:
        df = df[df["z"] <= 2.0].copy()

    required = ["mass_bin", "z", "Omega_m", "Omega_b", "little_h", "sigma8", "n_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cache results since many rows repeat the same combos
    cache = {}
    out_bias = []

    for _, r in df.iterrows():
        key = (
            str(r["mass_bin"]),
            float(r["z"]),
            float(r["Omega_m"]),
            float(r["Omega_b"]),
            float(r["little_h"]),
            float(r["sigma8"]),
            float(r["n_s"]),
        )
        if key not in cache:
            cache[key] = bin_averaged_bias_halomod(
                mass_bin=key[0],
                z=key[1],
                Omega_m=key[2],
                Omega_b=key[3],
                little_h=key[4],
                sigma8=key[5],
                n_s=key[6],
                dlog10m=DLOG10M_INTEGRATE,
            )
        out_bias.append(cache[key])

    df["halo_bias"] = out_bias
    df.to_csv(OUTFILE, index=False)

    print(f"Saved: {OUTFILE}")
    print("halo_bias summary:")
    print(df["halo_bias"].describe(percentiles=[0.16, 0.5, 0.84]))
    print(f"Unique cached computations: {len(cache)}")

if __name__ == "__main__":
    main()
