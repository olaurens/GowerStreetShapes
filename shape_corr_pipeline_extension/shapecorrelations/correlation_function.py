"""This script make fof output halo readble by covo to study correlations
Oriane Laurens
June 2025"""

# IMPORTS
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
from astropy import units as u
import math
from pathlib import Path
mp.rcParams['agg.path.chunksize'] = 10000 # to adjust size (suggested by a previous error)

# READING THE COVO OUTPUT FILE
def read_covo_output(file_name):
    """
    Read a covo output file.

    Column convention:
        col 0 : bin index (unused here)
        col 1 : r (distance)
        col 3 : r12_v1a
        col 4 : sigma(r12_v1a)
        col 5 : r12_v1b
        col 6 : sigma(r12_v1b)
    """
    file_path = Path(file_name).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    print(f"About to read {file_path}")
    #ignore'#'
    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    print(f"Read data with shape {data.shape}")
    return data

# GET CORRELATION DATA
def plot_data_correl(data, file_path):
    name = file_path.stem.lower()
    label = "Deepest Potential + Centre Of Mass"
    pos_mode = "DeepestPotential&CentreOfMass"
    
    run = None
    snapshot = None
    #search for the informaiton of run and snapshot using the parent file path
    parent_name = file_path.parent.stem
    parts = parent_name.split("_")
    # look for "runXXX"
    for p in parts:
        if p.startswith("run") and p[3:].isdigit():
             run = p
     # look for 5-digit snapshot like "00040", "00100"
    for p in parts:
        if len(p) == 5 and p.isdigit():
             snapshot = p

    shift = .5
    r = data.iloc[:, 1] #distance
    r12_v1a = data.iloc[:, 3] - shift #correlation vector a shifted
    r12_v1a_std = data.iloc[:, 4] #standard deviation correlation vector a
    r12_v1b= data.iloc[:, 5] - shift#correlation vector b
    r12_v1b_std = data.iloc[:, 6] #standard deviation correlation vector b

    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    ax[0].errorbar(r, r12_v1a, yerr = r12_v1a_std, xerr = None, label=f"{label} - vector a", color='blue')
    ax[0].set_xscale('log')
    #ax[0].set_xlabel('Distance r')
    ax[0].set_xlabel(r'Distance $r\;[h^{-1}\,\mathrm{Mpc}]$')
    ax[0].set_ylabel('Correlation')
    #ax[0].set_title(f"Correlation function of semi-major axis a ({run}, snapshot {snapshot})")
    ax[0].set_title(f"Correlation function of semi-major axis a ({run}, z=0)")
    #ax[0].legend()
    ax[0].grid(True)
    ax[1].errorbar(r, r12_v1b, yerr = r12_v1b_std, xerr = None, label=f"{label} - vector c", color='blue')
    ax[1].set_xscale('log')
    #ax[1].set_xlabel('Distance r')
    ax[1].set_xlabel(r'Distance $r\;[h^{-1}\,\mathrm{Mpc}]$')
    ax[1].set_ylabel('Correlation')
    #ax[1].set_title(f"Correlation function of semi-major axis c ({run}, snapshot {snapshot})")
    ax[1].set_title(f"Correlation function of semi-major axis c ({run}, z=0)")
    #ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    outname = f"corr_{run}_{snapshot}.png"
    outpath = file_path.parent / outname
    plt.savefig(outpath, dpi=300)
    print(f"Saved figure to {outpath}")

# EXECUTION
file_path = Path(sys.argv[1]).expanduser().resolve()
name = f"corr_{file_path.stem}.png"
data = read_covo_output(file_path)
plot_data_correl(data, file_path)
