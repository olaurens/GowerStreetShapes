"""This script processes covo correlation-function outputs and extracts
a scalar amplitude A for each (run, redshift slice, mass bin) by fitting
the correlation function relative to a reference template.

The reference template is taken from the z = 0 slice (slice 00100)
for the same run and mass bin.
For each file the script reads the correlation function ξ(r) and fits
    ξ(r) = A * ξ_ref(r)
where ξ_ref(r) is the reference correlation function at z = 0.

The amplitude A is estimated using a weighted least-squares regression
through the origin

After constructing the amplitude table, the script fits the scaling

    log A = log A0 + α log(1 + z) + β log(M / M0)

where

    M0 = 1e13 M_sun/h

Eason Tian
January 2026"""

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
    m = re.match(r"results_potential_plus_com_run(\d+)_(\d{5})_M(.+)", folder)
    if not m:
        raise ValueError(f"Cannot parse run/slice/mass from {folder}")
    run = int(m.group(1))   # '243'
    slice_id = str(int(m.group(2)))   # "40"   # "00040"
    mass_part = m.group(3)  # "1.0e+13-2.5e+13"

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
def extract_amplitude_from_df(df,which_vec="a",rmin=30.0,rmax=None,shift=0.5,min_counts=100,
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
    
    # propagate noise into weights
    if debias:
        ww = ww * (b ** 2)
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
    #Neff = (np.sum(ww)**2) / np.sum(ww**2)

    # weighted variance
    #var = float((ww * (xx - A) ** 2).sum() / ww.sum())
    #sigma_A = float(np.sqrt(var / max(Neff, 1.0)))
    #return A, sigma_A, int(xx.size)
    A = np.sum(ww * xx) / np.sum(ww)
    sigma_A = 1.0 / np.sqrt(np.sum(ww))

    if debias:
        sigma_A = sigma_A / b
  
    return float(A), float(sigma_A), int(xx.size)



# -----------------------------
# Fit logA = logA0 + alpha log(1+z) + beta log(M/M0)
# -----------------------------
def fit_alpha_beta(df_amp: pd.DataFrame, M0=1e13):
    """
    Expects columns: A, z, M_mid
    Fits on rows with A>0 (so log defined).
    """
    df_amp = df_amp.copy()

    # force numeric columns
    for col in ["z", "A", "M_mid", "sigma_A"]:
        if col in df_amp.columns:
            df_amp[col] = pd.to_numeric(df_amp[col], errors="coerce")

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
    print("\nDEBUG fit input:")
    print(d[["slice", "z", "mass_bin", "M_mid", "A"]].to_string(index=False))
    print("N rows =", len(d))
    print("unique z =", d["z"].nunique())
    print("unique M_mid =", d["M_mid"].nunique())
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

    # ---- Goodness of fit ----
    y_model = X @ coeff
    resid = y - y_model
    chi2 = np.sum((resid / sigma_logA)**2)
    dof = len(y) - len(coeff)   # N_data - N_params = N - 3
    chi2_red = chi2 / dof

    return {"logA0": logA0, "alpha": alpha, "beta": beta,
        "err_logA0": err[0], "err_alpha": err[1], "err_beta": err[2],
        "chi2": chi2, "dof": dof, "chi2_red": chi2_red,
        "N_used": int(len(y))}

def plot_logA_vs_log1pz(df,fit,which_vec,run=None,M0=1e13,mass_col="mass_bin",z_col="z",A_col="A",M_col="M_mid",rmin=None,rmax=None,out_png=None):
    """
    Plot log(A) vs log(1+z) with best-fit model overlay.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns A, z, M_mid, and mass_bin.
    fit : dict
        Output of fit_alpha_beta(), containing
        'logA0', 'alpha', 'beta'.
    which_vec : str
        Vector label, used only for plot title / filename.
    M0 : float
        Pivot mass used in the fit.
    out_png : str or Path or None
        If given, save figure to this path.
    """
     
    # Prepare data for plotting
    d = df.copy()
    d = d[np.isfinite(d[A_col]) & (d[A_col] > 0)]
    d = d[np.isfinite(d[z_col]) & np.isfinite(d[M_col])]
    
    #set upper boundary for the z value
    zmax = np.exp(1.0) - 1.0#so that the x axis only extends to 1
    d = d[d[z_col] <= zmax]

    d["logA"] = np.log(d[A_col])
    d["log1pz"] = np.log(1.0 + d[z_col])
    d["logM"] = np.log(d[M_col]/ M0)
    print("DEBUG unique mass_bin:", d[mass_col].unique())
    print("DEBUG counts by mass_bin:")
    print(d.groupby(mass_col).size())
    print("DEBUG rows:")
    print(d[[mass_col, z_col, A_col, M_col]].head(20).to_string(index=False))
    # Scatter plot by mass bin
    fig, ax = plt.subplots(figsize=(6, 5))
    
    z_grid = np.linspace(d[z_col].min(), d[z_col].max(), 300)
    log1pz_grid = np.log(1.0 + z_grid)
    
    # overall chi2/dof for the fitted model (all points) 
    y_data  = d["logA"].to_numpy()
    y_model = fit["logA0"] + fit["alpha"]*d["log1pz"].to_numpy() + fit["beta"]*d["logM"].to_numpy()
    resid = y_data - y_model
    
    chi2 = np.sum((resid / d["sigma_logA"].to_numpy())**2)
    dof  = len(y_data) - 3
    chi2_red = chi2 / dof
    #ax.text(0.95, 0.95, rf"$\chi^2/\mathrm{{dof}}={chi2_red:.2f}$",
    #    transform=ax.transAxes, va="top")
    
    for mass_bin, g in d.groupby(mass_col):
        #ax.scatter(g["log1pz"], g["logA"],s=35, alpha=0.8, label=mass_bin)
        ax.errorbar(g["log1pz"], g["logA"],yerr=g["sigma_logA"],fmt="o", ms=6, capsize=3, alpha=0.8,label=mass_bin)
        logM_bin = np.median(g["logM"])
        logA_model = fit["logA0"] + fit["alpha"] * log1pz_grid + fit["beta"] * logM_bin
        ax.plot(log1pz_grid, logA_model, lw=2, label=f"{mass_bin} $h^{{-1}}M_\\odot$")
    
    # Best-fit model curve(evaluate at median mass)
    #logM_med = np.median(d["logM"])
    #logA_model = (fit["logA0"]+ fit["alpha"] * log1pz_grid+ fit["beta"] * logM_med)
    #ax.plot(log1pz_grid, logA_model,color="k", lw=2,label="Best-fit model (median mass)")

    # Labels & cosmetics
    ax.set_xlabel(r"$\log(1+z)$")
    ax.set_ylabel(r"$\log A$")
    title = "Amplitude scaling (vec {})".format(which_vec)
    if run is not None:
        title += " - {}".format(run)
    #r_label = rf"$r \in [{rmin:.2g},\, {rmax:.2g}]$(Mpc)"
    r_label = rf"$r \in [{rmin:.2g},\, {rmax:.2g}]\,h^{{-1}}\mathrm{{Mpc}}$"
    title += " | " + r_label
    ax.set_title(title)
    #ax.set_title(f"Amplitude scaling (vec {which_vec})")
    #ax.legend(fontsize=9)
    legend = ax.legend(loc="lower right",fontsize=9)

    legend.set_title(rf"$\chi^2/\mathrm{{dof}} = {chi2_red:.2f}$")
    legend.get_frame().set_alpha(0.3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved figure:", out_png)
    plt.show()
    
def plot_logA_vs_logM(df, fit, which_vec, run=None,M0=1e13,mass_col="mass_bin",z_col="z",
                     A_col="A",M_col="M_mid",rmin=None,rmax=None,out_png=None):

    d = df.copy()
    d = d[np.isfinite(d[A_col]) & (d[A_col] > 0)]
    d = d[np.isfinite(d[z_col]) & np.isfinite(d[M_col])]

    d["logA"] = np.log(d[A_col])
    d["log1pz"] = np.log(1.0 + d[z_col])
    d["logM"] = np.log(d[M_col] / M0)

    fig, ax = plt.subplots(figsize=(6,5))

    M_grid = np.linspace(d[M_col].min(), d[M_col].max(), 300)
    logM_grid = np.log(M_grid / M0)

    # chi2
    y_data = d["logA"].to_numpy()
    y_model = (fit["logA0"]+ fit["alpha"] * d["log1pz"].to_numpy() + fit["beta"] * d["logM"].to_numpy())

    resid = y_data - y_model
    chi2 = np.sum((resid / d["sigma_logA"].to_numpy())**2)
    dof = len(y_data) - 3
    chi2_red = chi2 / dof

    #ax.text(0.05,0.95,rf"$\chi^2/\mathrm{{dof}}={chi2_red:.2f}$",transform=ax.transAxes,va="top")
    z_groups = sorted(d["z"].unique())
    dx_map = {z: i * 0.1 for i, z in enumerate(z_groups)}
    
    # scatter grouped by redshift slice
    for z_val, g in d.groupby(z_col):
        # Offset for this redshift slice
        dx = dx_map[z_val]
        # Plot data points (shifted horizontally for visibility)
        ax.errorbar(g["logM"] + dx,g["logA"],yerr=g["sigma_logA"],fmt='o',ms=6,
        alpha=0.85,capsize=2,label=f"True: z={z_val:.2f}")
        #ax.errorbar(g["logM"],g["logA"],yerr=g["sigma_logA"],fmt="o",
        #ms=6,capsize=3,alpha=0.8,label=f"z={z_val:.2f}")

        log1pz_med = np.median(g["log1pz"])

        logA_model = (fit["logA0"]+ fit["alpha"] * log1pz_med+ fit["beta"] * logM_grid)
        ax.plot(logM_grid + dx,logA_model,lw=2,label=f"Model: z={z_val:.2f}")
        #ax.plot(logM_grid,logA_model,lw=2,label=f"Model: z:{z_val:.2f}")
           
    ax.set_xlabel(r"$\log(M/M_0)$")
    ax.set_ylabel(r"$\log A$")

    title = "Amplitude scaling (vec {})".format(which_vec)

    if run is not None:
        title += " - {}".format(run)

    #r_label = rf"$r \in [{rmin:.2g},\, {rmax:.2g}]$(Mpc)"
    r_label = rf"$r \in [{rmin:.2g},\, {rmax:.2g}]\,h^{{-1}}\mathrm{{Mpc}}$"
    title += " | " + r_label

    ax.set_title(title)

    #ax.legend(fontsize=9)
    legend = ax.legend(loc="lower right",fontsize=9)
    
    legend.set_title(rf"$\chi^2/\mathrm{{dof}} = {chi2_red:.2f}$")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print("Saved figure:", out_png)

    plt.show()

#EXECUTION
which_vec = sys.argv[1]
rmin = float(sys.argv[2])
min_counts = int(sys.argv[3])
test_files = [Path(p).expanduser().resolve() for p in sys.argv[4:]]
rmax=84.13951416
r_label = rf"$r \in [{rmin:.2g},\, {rmax:.2g}]$"
#map each zvalue to it's run
slice_to_z_map = {}

rows = []

ref_map = {}
#prepare ref_ap nad slicez_map
for fp in test_files:
    df, path = read_covo_output(fp)
    run, slice_id, mass_part = parse_run_slice_mass(path)
    #collect z=0 templates (slice00100)
    if slice_id == "00100":
        key = (run, mass_part, which_vec)
        ref_map[key] = df
    if run not in slice_to_z_map:
        runsW_root = Path("/share/testde/ucapwhi/GowerStreetExtendedSims/runsW")
        #zfile_run = runsW_root / run / "z_values.txt"
        zfile_run = runsW_root / f"run{run:03d}" / "z_values.txt"
        slice_to_z_map[run] = build_slice_to_z(zfile_run, which="mid")

#bias_df = pd.read_csv("amplitude_table_veca_rmin10.0_rebuilt.csv")
bias_df = pd.read_csv("amplitude_table_veca_rmin10.0_noref_with_halobias.csv")
bias_lookup = {(int(r["run"]), str(r["slice"]), str(r["mass_bin"])): float(r["halo_bias"]) 
              for _, r in bias_df.iterrows()}
ref_mass = "1.0e+13-2.5e+13"

for fp in test_files:
    df, path = read_covo_output(fp)
    run, slice_id, mass_bin = parse_run_slice_mass(path)  # mass_bin like '7.5e+12-1.0e+13'
    M_mid = parse_mass_mid(mass_bin)
    slice_to_z = slice_to_z_map.get(run)
    z = slice_to_z.get(f"{int(slice_id):05d}")
    # pick ref_df 
    key_ref_df = (run, ref_mass, which_vec)
    ref_df = ref_map.get(key_ref_df)
    #print("DEBUG path =", path)
    #print("DEBUG parsed =", parse_run_slice_mass(path))
    # fetch halo_bias for data and reference
    key_bias = (run, slice_id, mass_bin)
    #print("DEBUG key_bias =", key_bias)
    #print("DEBUG first bias key =", next(iter(bias_lookup)))
    bias = bias_lookup[key_bias]
      

    # compute debiased amplitude
    A, sigma_A, n_used = extract_amplitude_from_df_norefver(df,which_vec=which_vec,rmin=rmin,rmax=None,shift=0.5,
                          min_counts=min_counts,bias=bias,debias=True,)
    rows.append({"run": run,
        "slice": slice_id,
        "z": z,
        "mass_bin": mass_bin,
        "M_mid": M_mid,
        "vec": which_vec,
        "A": A,
        "sigma_A": sigma_A,
        "sigma_logA": sigma_A / A,   
        "n_rbins_used": n_used,
        "path": str(path)})


out = pd.DataFrame(rows).sort_values(["run", "mass_bin", "slice"])
#out_csv = Path(f"amplitude_table_vec{which_vec}.csv")

for run, df_run in out.groupby("run"):
    fit_run = fit_alpha_beta(df_run, M0=1e13)
    #run_png = "logA_vs_log1pz_{}_vec{}.png".format(run, which_vec)i
    #run_png = f"logA_vs_log1pz_{run}_vec{which_vec}_rmin{rmin}_mincnt{min_counts}_noref.png"
    run_png = f"logA_vs_log1pz_{run}_vec{which_vec}_rmin{rmin}_mincnt{min_counts}_noref_debiased.png"
    #d = out[(out["run"]==run) & np.isfinite(out["A"]) & (out["A"]>0) & np.isfinite(out["z"])]
    #print(run, "z_min,z_max used:", d["z"].min(), d["z"].max(), "N=", len(d))
    #plot for z dependenceeach individuial run
    plot_logA_vs_log1pz(df_run, fit_run, which_vec,run=run, M0=1e13,rmin=rmin,
    rmax=rmax, out_png=run_png)
    #plot mass dependence
    #run_png_M = f"logA_vs_logM_{run}_vec{which_vec}_rmin{rmin}_mincnt{min_counts}_noref.png"
    run_png_M = f"logA_vs_logM_{run}_vec{which_vec}_rmin{rmin}_mincnt{min_counts}_noref_debiased.png"

    plot_logA_vs_logM(df_run,fit_run,which_vec,run=run,M0=1e13,
    rmin=rmin,rmax=rmax,out_png=run_png_M)
    print(run, "N_used=", fit_run["N_used"],
          "alpha={:.4f} beta={:.4f}".format(fit_run["alpha"], fit_run["beta"]))

fit = fit_alpha_beta(out, M0=1e13)
print("\nFit logA = logA0 + alpha log(1+z) + beta log(M/M0)")
print("M0 = 1e13")
print("N_used =", fit["N_used"])
#print(f"logA0 = {fit['logA0']:.4f} ± {fit['err_logA0']:.4f}")
#print(f"alpha = {fit['alpha']:.4f} ± {fit['err_alpha']:.4f}")
#print(f"beta  = {fit['beta']:.4f} ± {fit['err_beta']:.4f}")
print("logA0 = {:.4f} +/- {:.4f}".format(fit["logA0"], fit["err_logA0"]))
print("alpha = {:.4f} +/- {:.4f}".format(fit["alpha"], fit["err_alpha"]))
print("beta  = {:.4f} +/- {:.4f}".format(fit["beta"],  fit["err_beta"]))

#plot_logA_vs_log1pz(out,fit,which_vec=which_vec,run=None,rmin=rmin,
#    rmax=rmax,M0=1e13,out_png=f"logA_vs_log1pz_vec{which_vec}_rmin{rmin}_mincnt{min_counts}.png")
