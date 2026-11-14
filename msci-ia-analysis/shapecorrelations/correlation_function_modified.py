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

    
    #ignore'#'
    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    return data , file_path

# plot CORRELATION for different mass bin in the same time slice
def plot_data_correl_mb(data, file_path,ax0,ax1):
    name = file_path.stem.lower()
    label = "Deepest Potential + Centre Of Mass"
    pos_mode = "DeepestPotential&CentreOfMass"
    
    stem = file_path.parent.name
    mass_part = stem.split("_M", 1)[1]#keep the latter part for label
    
    shift = .5
    r = data.iloc[:, 1] #distance
    r12_v1a = data.iloc[:, 3] - shift #correlation vector a shifted
    r12_v1a_std = data.iloc[:, 4] #standard deviation correlation vector a
    r12_v1b= data.iloc[:, 5] - shift#correlation vector b
    r12_v1b_std = data.iloc[:, 6] #standard deviation correlation vector b
    ax0.errorbar(r, r12_v1a, yerr = r12_v1a_std, xerr = None, label=rf"vector a with halo mass bin-$M={mass_part} [M_\odot\,h^{{-1}}]$")
    ax1.errorbar(r, r12_v1b, yerr = r12_v1b_std, xerr = None, label=rf"vector c with halo mass bin-$M={mass_part} [M_\odot\,h^{{-1}}]$")
    

# plot CORRELATION for different redshit in the same mass bin
def plot_data_correl_rs(data, snapshot, ax0, ax1):
    """
    data: covo output
    snapshot: e.g. '00010', '00040'
    """
    label = "Deepest Potential + Centre Of Mass"
    shift = 0.5

    r = data.iloc[:, 1]
    r12_v1a = data.iloc[:, 3] - shift
    r12_v1a_std = data.iloc[:, 4]
    r12_v1b = data.iloc[:, 5] - shift
    r12_v1b_std = data.iloc[:, 6]

    ax0.errorbar(r, r12_v1a, yerr=r12_v1a_std,label=f"vector a, snapshot {snapshot}" )
    ax1.errorbar(r, r12_v1b, yerr=r12_v1b_std,label=f"vector c, snapshot {snapshot}" )

#run function for different mass bin
def run_multiM(file_paths):
    data, file_path = read_covo_output(file_paths[0])  # using the first file passed in to read off data and file path

    run = None
    snapshot = None
    # search for information of run and snapshot using the parent file path
    parent_name = file_path.parent.stem
    parts = parent_name.split("_")

    fig, ax = plt.subplots(2, 1, figsize=(8, 5))

    # look for "runXXX"
    for p in parts:
        if p.startswith("run") and p[3:].isdigit():
            run = p
    # look for 5-digit snapshot like "00040", "00100"
    for p in parts:
        if len(p) == 5 and p.isdigit():
            snapshot = p

    for fp in file_paths:
        data, this_path = read_covo_output(fp)
        plot_data_correl_mb(data, this_path, ax[0], ax[1]) 
    ax[0].set_xscale('log')
    #ax[0].set_xlabel('Distance r')
    ax[0].set_xlabel(r'Distance $r\;[h^{-1}\,\mathrm{Mpc}]$')
    ax[0].set_ylabel('Correlation')
    ax[0].set_title(rf"Correlation function of semi-major axis a ({run}, snapshot {snapshot})")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xscale('log')
    #ax[1].set_xlabel('Distance r')
    ax[1].set_xlabel(r'Distance $r\;[h^{-1}\,\mathrm{Mpc}]$')
    ax[1].set_ylabel('Correlation')
    ax[1].set_title(rf"Correlation function of semi-major axis c ({run}, snapshot {snapshot})")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()

    name = f"corr_{run}_{snapshot}_multiM.png"
    script_dir = Path(__file__).resolve().parent
    outpath = script_dir / name
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")

#run function for different redshift
def run_multiZ(file_paths):
    data, file_path = read_covo_output(file_paths[0])

    run = None
    mass_bin = None
    # parent name with full mass bin
    parent_name = file_path.parent.name
    parts = parent_name.split("_")

    for p in parts:
        if p.startswith("run") and p[3:].isdigit():
            run = p
        if p.startswith("M"):
            mass_bin = p[1:]

    fig, ax = plt.subplots(2, 1, figsize=(8, 5))

    # multi-snapshot version:
    for fp in file_paths:
        data, this_path = read_covo_output(fp)
        dirname = this_path.parent.name
        parts_this = dirname.split("_")

        snapshot_this = None
        for q in parts_this:
            if len(q) == 5 and q.isdigit():
                snapshot_this = q

        plot_data_correl_rs(data, snapshot_this, ax[0], ax[1])

    ax[0].set_xscale('log')
    #ax[0].set_xlabel('Distance r')
    ax[0].set_xlabel(r'Distance $r\;[h^{-1}\,\mathrm{Mpc}]$')
    ax[0].set_ylabel('Correlation')
    #ax[0].set_title(f"Correlation function of semi-major axis a ({run}, M={mass_bin})")
    ax[0].set_title(rf"Correlation function of major axis a ({run}, $M={mass_bin} [M_\odot\,h^{{-1}}]$)")
    ax[0].grid(True)
    #ax[0].legend()
    ax[0].legend(loc='upper right', fontsize=9, framealpha=0.8)

    ax[1].set_xscale('log')
    #ax[1].set_xlabel('Distance r')
    ax[1].set_xlabel(r'Distance $r\;[h^{-1}\,\mathrm{Mpc}]$')
    ax[1].set_ylabel('Correlation')
    #ax[1].set_title(f"Correlation function of semi-major axis c ({run}, M={mass_bin})")
    ax[1].set_title(rf"Correlation function of major axis c ({run}, $M={mass_bin} [M_\odot\,h^{{-1}}]$)")
    ax[1].grid(True)
    #ax[1].legend()
    ax[1].legend(loc='upper right', fontsize=9, framealpha=0.8)
    plt.tight_layout()

    name = f"corr_{run}_M{mass_bin}_multiZ.png"
    script_dir = Path(__file__).resolve().parent
    outpath = script_dir / name
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")

#EXECUTION
file_paths = [Path(f).expanduser().resolve() for f in sys.argv[1:]]
#run_multiZ(file_paths)
run_multiM(file_paths)
