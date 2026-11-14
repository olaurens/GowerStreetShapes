

"""

Construct a consolidated amplitude table for cosmological analysis.

This script loops over a selected subset of simulation runs, reads their
cosmological parameters from `control.par`, and extracts a scalar amplitude
(alpha) from correlation function outputs using a fixed definition
(rmin, reference correlation, vector component, etc.).

"""
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import math
from amp_lib import (
    read_covo_output,
    parse_run_slice_mass,
    parse_mass_mid,
    extract_amplitude_from_df_norefver,
    extract_amplitude_from_df_refver,
    fit_alpha_beta,
    build_slice_to_z)


BASE = Path("/share/testde/ucapwhi/GowerStreetExtendedSims/runsW")

RUNS = [int(sys.argv[1])]
out_path = Path(sys.argv[2])



if len(sys.argv) > 3 and sys.argv[3].endswith(".csv"):
    COSMO_TABLE = sys.argv[3]
    test_paths = sys.argv[4:]
else:
    test_paths = sys.argv[3:]


#RUNS = [92, 119, 189, 221, 240, 255, 257, 311,
    #340, 344, 378, 437, 510, 776, 823, 890,
    #144, 790, 207, 243, 236, 407, 371]


WHICH_VEC = "a"          # correlation vector component
RMIN = 10.0              # minimum separation used in fit
RMAX = None              # maximum separation (None = no upper cut)


REF_SLICE = "00100"      # snapshot / slice used as reference
REF_MASS  = "1.0e+13-2.5e+13"

# Directory containing covo output files
COVO_DIR = Path("results_potential_plus_com")

RUNSW_ROOT = Path("/share/testde/ucapwhi/GowerStreetExtendedSims/runsW")

allowed_slices = {"00040","00060","00080","00100"}
#renof version
#COSMO_TABLE = "amplitude_table_veca_rmin10.0_noref_with_halobias.csv"
COSMO_TABLE = "amplitude_table_veca_rmin10.0_with_halobias.csv"
_cosmo_df = pd.read_csv(COSMO_TABLE)
# Only take one row for each run
_cosmo_runs = _cosmo_df.drop_duplicates("run").set_index("run")

_bias_df = pd.read_csv(COSMO_TABLE)

# Normalize slice id (e.g. "00040" -> "40") to match bias_lookup keys
bias_lookup = {(int(rr["run"]), str(int(rr["slice"])), str(rr["mass_bin"]).strip()): float(rr["halo_bias"])
    for _, rr in _bias_df.iterrows()}

def cosmo_from_table(run_int):
    row = _cosmo_runs.loc[run_int]

    return {"Omega_m": float(row["Omega_m"]),
            "sigma8": float(row["sigma8"]),
            "Omega_b": float(row["Omega_b"]),
            "little_h": float(row["little_h"]),
            "n_s": float(row["n_s"]),
            "w0": float(row["w0"]),
            "wa": float(row["wa"]),
            "m_nu": float(row["m_nu"]),
            "z": row["z"], }

# table that already contains run / slice / z
#_z_df = pd.read_csv("amplitude_table_veca_rmin10.0_noref_with_halobias.csv")
_z_df =pd.read_csv(COSMO_TABLE)
# build lookup dictionary
_z_lookup = {(int(r["run"]), str(int(r["slice"]))): float(r["z"])
    for _, r in _z_df.iterrows()}

def z_from_table(run_int, slice_id):
    """
    Return redshift corresponding to (run, slice)
    """

    slice_key = str(int(slice_id))  # "00040" -> "40"

    key = (int(run_int), slice_key)

    if key not in _z_lookup:
        raise KeyError(f"Missing z for run={run_int}, slice={slice_key}")

    return _z_lookup[key]

def run_control_par(run_name) :# reading control.par file from run
    #file_name = f"/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/{run}/control.par"
    variable = {}
    #print("About to read {}".format(file_name))
    run_root = RUNSW_ROOT / run_name
    matches = list(run_root.rglob("control.par"))
    file_name = matches[0]
    print("About to read", file_name)
    with open(file_name, "r") as in_file :
        exec(in_file.read(), {"math": math}, variable)

    dOmega0 = variable["dOmega0"]
    dSigma8 = variable["dSigma8"]

    return dOmega0,dSigma8

def main():
    """
    Main driver function.

    For each selected run:
      1. Read Omega_m and sigma_8 from control.par
      2. Locate all covo output files for that run
      3. Identify the reference correlation (fixed slice and mass bin)
      4. Extract the amplitude alpha from each covo file
      5. Store results in a unified table

    The final table is written to disk as a CSV file.
    """

    rows = []

    
    for r in RUNS:
        run_name = "run" + str(r).zfill(3)
        print("=== Processing", run_name, "===")

        #dOmega0, dSigma8 = run_control_par(run_name)
        cosmo = cosmo_from_table(int(r))
        #covo_files = sorted(COVO_DIR.glob("*" + run_name + "*"))
        covo_files = sorted(Path(".").glob(f"results_potential_plus_com_{run_name}_*/test"))
        
        print("  covo_files:", len(covo_files))
        if len(covo_files) > 0:
            print("  example:", covo_files[0])
        runsW_root = Path("/share/testde/ucapwhi/GowerStreetExtendedSims/runsW")
        #zfile_run = runsW_root / run_name / "z_values.txt"
        #slice_to_z = build_slice_to_z(zfile_run, which="mid")
        # Read all covo files once
        records = []
        for fp in covo_files:
            df, path = read_covo_output(fp)
            run, slice_id, mass_bin = parse_run_slice_mass(path)
            records.append((df, slice_id, mass_bin))
            #print("  ", Path(path).name, "-> slice:", slice_id, "mass_bin:", mass_bin)
        # Find reference correlation
        ref_df = None
        for df, slice_id, mass_bin in records:
            if slice_id == REF_SLICE and mass_bin == REF_MASS:
                ref_df = df
                break

        # Extract amplitude
        for df, slice_id, mass_bin in records:
          if slice_id not in allowed_slices:
              continue
          slice_key = str(int(slice_id))     # "00040" -> "40"
          mass_key  = str(mass_bin).strip()

          key_bias = (int(r), slice_key, mass_key)
          #key_bias_ref = (int(r), slice_key, str(REF_MASS).strip())
          bias = bias_lookup[key_bias]
          #bias_ref = bias_lookup[key_bias_ref]
          z = z_from_table(r, slice_id)

          #A, sigma_A, n_used = extract_amplitude_from_df(df,which_vec=WHICH_VEC,rmin=RMIN,rmax=RMAX,shift=0.5,
          #              ref_df=ref_df,bias=bias,bias_ref=bias_ref,debias=True,)
          A, sigma_A, n_used = extract_amplitude_from_df_norefver(df,which_vec=WHICH_VEC,rmin=RMIN,rmax=RMAX,
             shift=0.5,bias=bias,debias=True,)
          M_mid = parse_mass_mid(mass_bin)
          print("DEBUG", r, slice_id, mass_bin, "bias =", bias, "A =", A, flush=True)  
          rows.append({"run": r,
              "slice": slice_id,
              "z": z,
              "vec": WHICH_VEC,
              "mass_bin": mass_bin,
              "rmin": RMIN,
              "A": A,
              "M_mid": M_mid,
              "sigma_A": sigma_A,
              "n_rbins_used": n_used,
              "Omega_m": cosmo["Omega_m"],
              "sigma8": cosmo["sigma8"],
              "Omega_b": cosmo["Omega_b"],
              "little_h": cosmo["little_h"],
              "n_s": cosmo["n_s"],
              "w0": cosmo["w0"],
              "wa": cosmo["wa"],
              "m_nu": cosmo["m_nu"],})
    out = pd.DataFrame(rows)

    fit_rows = []

    for run, df_run in out.groupby("run"):
        try:
            fit = fit_alpha_beta(df_run, M0=1e13)

            fit_rows.append({"run": run,
              "logA0": fit["logA0"],
              "alpha_fit": fit["alpha"],
              "beta_fit": fit["beta"],
              "err_logA0": fit["err_logA0"],
              "err_alpha": fit["err_alpha"],
              "err_beta": fit["err_beta"],
              "N_used": fit["N_used"],})
        except np.linalg.LinAlgError:
            print(f"WARNING: singular fit for run {run}, skipping.")
            fit_rows.append({"run": run,
                        "logA0": np.nan,
                        "alpha_fit": np.nan,
                        "beta_fit": np.nan,
                        "err_logA0": np.nan,
                        "err_alpha": np.nan,
                        "err_beta": np.nan,
                        "N_used": 0,})


    # Save output table
    
    fit_df = pd.DataFrame(fit_rows)
    out = out.merge(fit_df, on="run", how="left")
    

    #out_name = f"amplitude_table_vec{WHICH_VEC}_rmin{int(RMIN)}.csv"
    #out.to_csv(out_name, index=False)
    out.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("Number of rows:", len(out))

    
if __name__ == "__main__":
    main()
