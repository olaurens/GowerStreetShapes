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
import re
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
    ax0.errorbar(r, r12_v1a, yerr = r12_v1a_std, xerr = None, label=f"vector a with halo mass bin-M={mass_part}")
    ax1.errorbar(r, r12_v1b, yerr = r12_v1b_std, xerr = None, label=f"vector c with halo mass bin-M={mass_part}")
    

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
        plot_data_correl(data, this_path, ax[0], ax[1])

    ax[0].set_xscale('log')
    ax[0].set_xlabel('Distance r')
    ax[0].set_ylabel('Correlation')
    ax[0].set_title(f"Correlation function of semi-major axis a ({run}, snapshot {snapshot})")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xscale('log')
    ax[1].set_xlabel('Distance r')
    ax[1].set_ylabel('Correlation')
    ax[1].set_title(f"Correlation function of semi-major axis c ({run}, snapshot {snapshot})")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()

    name = f"corr_{run}_{snapshot}_multiM.png"
    script_dir = Path(__file__).resolve().parent
    outpath = script_dir / name
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")


def build_slice_to_z(zfile="z_values.txt", which="mid"):
    """
    Build mapping: slice_id (e.g. '00040') -> representative redshift z
    from pkdgrav3 z_values.txt
    """
    colnames = [
        "Step","z_far","z_near","delta_z",
        "cmd_far","cmd_near","delta_cmd",
        "cmd_over_box_far","cmd_over_box_near","delta_cmd_over_box",
        "cmvolume","cmvolume_over_boxvolume",
    ]

    dfz = pd.read_csv(
        zfile,
        sep=",",
        comment="#",
        header=None,
        names=colnames,
    )
    if which == "mid":
        zrep = 0.5 * (dfz["z_far"] + dfz["z_near"])
    elif which == "near":
        zrep = dfz["z_near"]
    elif which == "far":
        zrep = dfz["z_far"]
    
    SLICE_TO_Z = {
        f"{int(step):05d}": float(z)
        for step, z in zip(dfz["Step"], zrep)
    }
    return SLICE_TO_Z

#import redshift value(z)
SLICE_TO_Z = build_slice_to_z("/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/run371/z_values.txt",which="mid")

def parse_run_slice_mass(test_path: Path):
    """
    Expect folder name like:
      results_potential_plus_com_run243_00080_M1.0e+13-2.5e+13
    and test_path ends with /test
    """
    
    folder = test_path.parent.name

    # run number extraction
    mrun = re.search(r"(run\d+)", folder)
    run = mrun.group(1)

    # slice: first 5-digit token
    slice_id = None
    for tok in folder.split("_"):
        if len(tok) == 5 and tok.isdigit():
            slice_id = tok
            break
    # mass: part after "_M"
    if "_M" in folder:
        mass_part = folder.split("_M", 1)[1]
    return run, slice_id, mass_part
   
#run function for a heatmap with different redshift
def make_heatmap(file_paths):
    # read all, parse metadata
    records = []

    for fp in file_paths:
           df, path = read_covo_output(fp)
           run, slice_id, mass_part = parse_run_slice_mass(path)
           r = df.iloc[:, 1].to_numpy()
           counts = df.iloc[:, 2].to_numpy()

        
           shift = 0.5
           a = df.iloc[:, 3].to_numpy() - shift
           c = df.iloc[:, 5].to_numpy() - shift

           MIN_COUNTS = 1000  # exclude small halo number samples

           valid = (counts >= MIN_COUNTS) & np.isfinite(a) & np.isfinite(c)
           records.append({
            "path": path,
            "run": run,
            "slice": slice_id,
            "mass": mass_part,
            "r": r,
            "a": a,# semi-major axis a
            "c": c,# semi-major axis c
            "valid": valid, # bool mask 
            })

    run0 = records[0]["run"]
    mass0 = records[0]["mass"]

    # sort by slice (numeric)
    records.sort(key=lambda x: x["slice"])
    
    r = records[0]["r"]

    # y-axis values: redshift 
    yvals = []
    for rec in records:
        slc = rec["slice"]
        if slc in SLICE_TO_Z:
            yvals.append(SLICE_TO_Z[slc])
    yvals = np.array(yvals)

    # Build matrices: rows = slices, cols = r bins
    A = []
    C = []
    for rec in records:
        # put invalid bins as nan so they show as blank
        a_row = rec["a"].copy()
        c_row = rec["c"].copy()
        a[~valid] = np.nan
        c[~valid] = np.nan
        A.append(a_row)
        C.append(c_row)

    A = np.vstack(A)
    C = np.vstack(C)

    return run0, mass0,r, yvals, A, C, records

#plot heatmap for the same mass bin across redshifts
def plot_heatmap(run, mass, r, y, Z, out_png, title):
    plt.figure(figsize=(9, 5))
    # pcolormesh expects 2D; x should be increasing
    plt.pcolormesh(r, y, Z, shading="auto")
    plt.xscale("log")
    plt.xlabel("Distance r")
    plt.ylabel("redshift (if provided) else slice id")
    plt.title(title)
    plt.colorbar(label="correlation (shifted by -0.5)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_png}")

def plot_three_mass_heatmaps(mass_to_files, which_vec="a", out_png="heatmap_allmass.png",title=None,vmin=None,vmax=None):
    packs = []
    for mass, fps in mass_to_files.items():
        run, mass0, r, y, A, C, records = make_heatmap(fps)
        Z = A if which_vec == "a" else C
        packs.append((mass, r, y, Z, run))
    #if min and max is not given, calculate the range inside each run
    if vmin is None or vmax is None:
        # Collect all heatmap values (Z) across different mass bins / redshifts
        allZ = np.concatenate([p[3].ravel() for p in packs])
        allZ = allZ[np.isfinite(allZ)]
        # Define color scale limits using robust quantiles(avoid extreme values)
        vmin, vmax = np.quantile(allZ, [0.02, 0.98])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True,constrained_layout=True)
    mappable = None

    for ax, (mass, r, y, Z, run) in zip(axes, packs):
        m = ax.pcolormesh(r, y, Z, shading="auto", vmin=vmin, vmax=vmax)
        mappable = m
        ax.set_xscale("log")
        #ax.set_title(f"{run} | {mass} | vec {which_vec}")
        ax.set_title(rf" $M = {mass}\;[M_\odot\,h^{{-1}}]$ ")
        ax.set_xlabel(r"Distance $r\;[h^{-1}\,\mathrm{Mpc}]$")
        #ax.set_xlabel("Distance r")
        ax.grid(False)

    axes[0].set_ylabel("redshift (or slice id)")
    
    fig.suptitle(title)
    cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.95, pad=0.02)
    cbar.set_label("correlation (shifted by -0.5)")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("Saved:", out_png)

def global_vmin_vmax(list_of_mass_to_files, which_vec="a"):
    """
    Compute a shared (vmin, vmax) across multiple runs (each run is a groups dict).
    Uses robust quantiles to avoid outliers dominating the color scale.
    """
    allZ = []
    for mass_to_files in list_of_mass_to_files:
        for mass, fps in mass_to_files.items():
            run, mass0, r, y, A, C, _ = make_heatmap(fps)
            Z = A if which_vec == "a" else C
            allZ.append(Z.ravel())
    allZ = np.concatenate(allZ)
    allZ = allZ[np.isfinite(allZ)]
    return np.quantile(allZ, [0.02, 0.98])

def build_groups(file_paths):
    """
    Group a list of /test files by mass bin.
    Returns: dict {mass_str: [Path, Path, ...]}
    """
    groups = {}
    for fp in file_paths:
        run, slice_id, mass = parse_run_slice_mass(fp)
        groups.setdefault(mass, []).append(fp)
    return groups
#EXECUTION
file_paths = [Path(f).expanduser().resolve() for f in sys.argv[1:]]

paths_by_run = {}
for fp in file_paths:
    run, slice_id, mass = parse_run_slice_mass(fp)
    paths_by_run.setdefault(run, []).append(fp)

#build groups per run (mass to list of slices)
groups_by_run = {}
for r in paths_by_run:
    groups_by_run[r] = build_groups(paths_by_run[r])

vmin_a, vmax_a = global_vmin_vmax([groups_by_run[r]], which_vec="a")
vmin_c, vmax_c = global_vmin_vmax([groups_by_run[r]], which_vec="c")

script_dir = Path(__file__).resolve().parent

#out_a = script_dir / f"heatmap_{run}_{mass}_a.png"
#out_c = script_dir / f"heatmap_{run}_{mass}_c.png"

#title_a = f"Heatmap: vector a | {run} | {mass}"
#title_c = f"Heatmap: vector c | {run} | {mass}"

#plot_heatmap(run, mass, r, y, A, out_a, title_a)
#plot_heatmap(run, mass, r, y, C, out_c, title_c)

#group according to mass bin
#groups = {}
#for fp in file_paths:
        #run, slice_id, mass = parse_run_slice_mass(fp)
        # Collect all slices belonging to the same mass bin
        #if mass not in groups:
            #groups[mass] = []
        #groups[mass].append(fp)

#packs_a = []
#packs_c = []

#script_dir = Path(__file__).resolve().parent
#run_name = run

#vmin_a, vmax_a = global_vmin_vmax([groups243, groups144, groups371], which_vec="a")
#vmin_c, vmax_c = global_vmin_vmax([groups243, groups144, groups371], which_vec="c")


#plot_three_mass_heatmaps(groups,which_vec="a",out_png=script_dir / f"heatmap3_{run_name}_a.png",title=f"Heatmap (3 mass bins) | vector a | {run_name}",)

#plot_three_mass_heatmaps(groups,which_vec="c",out_png=script_dir / f"heatmap3_{run_name}_c.png",

for r in paths_by_run:
    plot_three_mass_heatmaps(groups_by_run[r],which_vec="a",out_png=script_dir / f"heatmap3_{r}_a.png",title=f"Heatmap (3 mass bins) | vector a | {r}",vmin=vmin_a, vmax=vmax_a,)

