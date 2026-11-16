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
        col 2 : counts
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
    return group


def parse_mass_mid(mass_part: str):
    """
    mass_part like '1.0e+13-2.5e+13' -> return geometric mid mass
    """
    low, high = mass_part.split("-")
    mlow = float(low)
    mhigh = float(high)
    # geometric mean is usually better for log-scaling fits
    return np.sqrt(mlow * mhigh)


def build_slice_to_z(z_values_path: Path, which="mid"):
    """
    Reads your z_values.txt which has CSV-like rows:
    Step,z_far,z_near,delta_z,...
    but your file seems to NOT have a header row actually parsed by pandas.
    We'll load with explicit column names and skip comment lines.
    """
    z_values_path = Path(z_values_path).expanduser().resolve()
    if not z_values_path.exists():
        raise FileNotFoundError(f"z_values.txt not found: {z_values_path}")

    # If the file truly has a header line starting with '# Step,...'
    # you can set header=0. But your debug showed pandas treated first data row as header.
    # So: use names=... and header=None, then skip the first line if it begins with '#'.
    names = ["Step", "z_far", "z_near", "delta_z",
             "cmd_far", "cmd_near", "delta_cmd",
             "cmd_box_far", "cmd_box_near", "delta_cmd_box",
             "cmvolume", "cmvolume_boxvolume"]

    # read all lines, drop comment/header lines starting with '#'
    lines = z_values_path.read_text().splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    # now parse with pandas from string
    from io import StringIO
    dfz = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=names)

    # representative z
    if which == "mid":
        # midpoint redshift of the shell: average of far and near redshift
        zrep = 0.5 * (dfz["z_far"] + dfz["z_near"])
    elif which == "near":
        # lower-z edge (nearer to observer)
        zrep = dfz["z_near"]
    elif which == "far":
        # higher-z edge (farther away)
        zrep = dfz["z_far"]
    else:
        raise ValueError("which must be 'mid', 'near', or 'far'")

    # map Step -> "00040" style string
    slice_to_z = {f"{int(step):05d}": float(z) for step, z in zip(dfz["Step"], zrep)}
    return slice_to_z



# -----------------------------
# Amplitude extraction(method1)
# -----------------------------
def extract_amplitude_from_df_refver(df,which_vec="a",rmin=30.0,rmax=None,shift=0.5,min_counts=100,
    ref_df=None,bias=1.0,bias_ref=1.0,debias=True,):
    """
    Returns scalar amplitude A by fitting:
        corr(r) = A * corr_ref(r)
    using a weighted least-squares regression through the origin.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from covo "test" output (whitespace-delimited, comment='#').
        Expected columns by index:
            col1: r
            col2: counts
            col3/5: correlation (depending on which_vec)
    which_vec : "a" or "c"
        "a" uses column 3 (r12_v1a), "c" uses column 5 (r12_v1b).
    rmin, rmax : float
        Fit range in r.
    shift : float
        Baseline shift to subtract (e.g. 0.5).
    min_counts : int
        Minimum pair counts per bin.
    ref_df : pandas.DataFrame
        Reference correlation DataFrame
    bias, bias_ref : float
        Halo bias values for df and ref_df (used only if debias=True).
    debias : bool
        If True, use (corr-shift)/bias and (corr_ref-shift)/bias_ref in the fit.
    """
    print("DEBUG debias=", debias, "bias=", bias, "bias_ref=", bias_ref)
    if ref_df is None:
        raise ValueError("ref_df must be provided")
    r = df.iloc[:, 1].to_numpy()
    counts = df.iloc[:, 2].to_numpy()

    r_ref = ref_df.iloc[:, 1].to_numpy()
    counts_ref = ref_df.iloc[:, 2].to_numpy()

    if which_vec == "a":
        corr = df.iloc[:, 3].to_numpy() - shift
        corr_ref = ref_df.iloc[:, 3].to_numpy() - shift
    else:
        corr = df.iloc[:, 5].to_numpy() - shift
        corr_ref = ref_df.iloc[:, 5].to_numpy() - shift

    # Optional debiasing
    if debias:
        b = float(bias)
        br = float(bias_ref)
        corr = corr / b
        corr_ref = corr_ref 

    valid = (np.isfinite(r) & np.isfinite(corr) & np.isfinite(counts) &
        (counts >= min_counts))
    valid_ref = (np.isfinite(r_ref) & np.isfinite(corr_ref) & np.isfinite(counts_ref) &
        (counts_ref >= min_counts))

    # Check if upper limit is necessary
    if rmax is not None:
        valid &= (r >= rmin) & (r <= rmax)
        valid_ref &= (r_ref >= rmin) & (r_ref <= rmax)
    else:
        valid &= (r >= rmin)
        valid_ref &= (r_ref >= rmin)

    m = valid & valid_ref
    xr = corr_ref[m]                  # x-reference
    xx = corr[m]                      # x-data
    ww = counts[m]     # weights (pair counts)

   
    # Weighted least-squares slope through origin: xx = A * xr
    A = np.sum(ww * xx * xr) / np.sum(ww * xr * xr)
    sigma_A = np.sqrt(1.0 /np.sum(ww * xr * xr) )
    print("DEBUG A=", A, "sigma_A=", sigma_A, "b=", b)
    return A, sigma_A, int(xr.size)


# -----------------------------
# Amplitude extraction(method2)
# -----------------------------
def extract_amplitude_from_df_norefver(df,which_vec="a",rmin=30.0,rmax=None,shift=0.5,min_counts=100,
    ref_df=None,bias=1.0,bias_ref=1.0,debias=True,):
    """
    Estimate the alignment amplitude A from a correlation table.

    The estimator assumes

        corr(r) = shift + A

    so that

        A = <corr(r) - shift>

    where the average is taken over r bins with weights equal to
    the pair counts in each bin.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from covo "test" output (whitespace-delimited, comment='#').
        Expected columns by index:
            col1: r
            col2: counts
            col3/5: correlation (depending on which_vec)
    which_vec : "a" or "c"
        "a" uses column 3 (r12_v1a), "c" uses column 5 (r12_v1b).
    rmin, rmax : float
        Fit range in r.
    shift : float
        Baseline shift to subtract (e.g. 0.5).
    min_counts : int
        Minimum pair counts per bin.
    ref_df : pandas.DataFrame
        Reference correlation DataFrame
    bias, bias_ref : float
        Halo bias values for df and ref_df (used only if debias=True).
    debias : bool
        If True, use (corr-shift)/bias and (corr_ref-shift)/bias_ref in the fit.
    """

    r = df.iloc[:, 1].to_numpy()
    counts = df.iloc[:, 2].to_numpy()

    if which_vec == "a":
        corr = df.iloc[:, 3].to_numpy() - shift
    else:
        corr = df.iloc[:, 5].to_numpy() - shift

    # Optional debiasing
    if debias:
        b = float(bias)
        corr = corr / b

    valid = (np.isfinite(r) & np.isfinite(corr) & np.isfinite(counts) &
        (counts >= min_counts))

    # Check if upper limit is necessary
    if rmax is not None:
        valid &= (r >= rmin) & (r <= rmax)
    else:
        valid &= (r >= rmin)
    m = valid 
    rr = r[m]                
    xx = corr[m]      # x-data
    ww = counts[m]     # weights (pair counts)

    #A = np.sum(ww * xx) / np.sum(ww)   
    # effective number of samples 
    #Neff = (ww.sum() ** 2) / ( (ww**2).sum() )
    
    # weighted variance
    #var = float((ww * (xx - A) ** 2).sum() / ww.sum())
    #sigma_A = float(np.sqrt(var / max(Neff, 1.0)))
    A = np.sum(ww * xx) / np.sum(ww)
    sigma_A = 1.0 / np.sqrt(np.sum(ww))
    if debias:                                                                                                                  sigma_A = sigma_A / b
    return A, sigma_A, int(xx.size)



# -----------------------------
# Fit logA = logA0 + alpha log(1+z) + beta log(M/M0)
# -----------------------------
def fit_alpha_beta(df_amp: pd.DataFrame, M0=1e13):
    """
    Expects columns: A, z, M_mid
    Fits on rows with A>0 (so log defined).
    """
    good = (np.isfinite(df_amp["A"]) & (df_amp["A"] > 0) &
        np.isfinite(df_amp["z"]) &
        np.isfinite(df_amp["M_mid"]))
   
    d = df_amp.loc[good].copy()

    #set upper boundary for the z value
    zmax = np.exp(1.0) - 1.0#so that the x axis only extends to 1
    d = d[d["z"] <= zmax]
     
    y  = np.log(d["A"].to_numpy())
    x1 = np.log(1.0 + d["z"].to_numpy())# alpha term
    x2 = np.log(d["M_mid"].to_numpy() / M0) # beta term
    # Rewrite the model in matrix form as y = X @ coeff, where X = [1, x1, x2]
    X = np.vstack([np.ones_like(y), x1, x2]).T
    #error bar for logA
    sigma_logA = d["sigma_A"].to_numpy() / d["A"].to_numpy()
    w = 1.0 / (sigma_logA**2)

    # Weighted least squares via normal equations
    XtW = X.T * w  # each row of X.T scaled by w
    cov = np.linalg.inv(XtW @ X)
    coeff = cov @ (XtW @ y)
    logA0, alpha, beta = coeff
    err = np.sqrt(np.diag(cov))

    return {"logA0": logA0,"alpha": alpha, "beta": beta,
            "err_logA0": err[0], "err_alpha": err[1], "err_beta": err[2],
            "N_used": int(len(y))}
 

