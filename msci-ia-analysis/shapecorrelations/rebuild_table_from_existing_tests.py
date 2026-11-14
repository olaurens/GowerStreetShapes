#!/usr/bin/env python3
import subprocess
from pathlib import Path
import pandas as pd

BASE = Path("/share/testde/ucapwhi/GowerStreetExtendedSims")
RUNSW = BASE / "runsW"
HERE = Path(__file__).resolve().parent

WHICH_VEC = "a"
RMIN = 10.0

BUILD_PY = HERE / "build_amplitude_table.py"  
MASTER_CSV = HERE / f"amplitude_table_vec{WHICH_VEC}_rmin{RMIN}_noref_debiased.csv"
COSMO_INPUT = HERE / "amplitude_table_veca_rmin10.0_noref_with_halobias.csv"



def run(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, cwd=str(cwd) if cwd else None)

def run_name_of(r: int) -> str:
    return f"run{r:03d}"

#RUNS = sorted(int(p.name.split(".")[0].replace("run", "")) for p in RUNSW.glob("run*.tar.gz"))
RUNS = sorted(int(p.name.split(".")[0].replace("run", "")) 
              for p in RUNSW.glob("run[0-9][0-9][0-9].tar.gz"))

all_rows = []
for r in RUNS:
    run_name = run_name_of(r)
    print(f"\n===== {run_name} =====")

    run_dir = RUNSW / run_name
    tests = sorted(run_dir.glob(f"**/results_potential_plus_com_{run_name}_?????_M*/test"))
    if not tests:
        tests = sorted(HERE.glob(f"results_potential_plus_com_{run_name}_?????_M*/test"))


    per_csv = HERE / f"_tmp_{run_name}_vec{WHICH_VEC}_rmin{RMIN}_debiased.csv"
    #run(["python", str(BUILD_PY), str(r), str(per_csv), *map(str, tests)], cwd=HERE)
    run(["python", str(BUILD_PY),str(r),str(per_csv),str(COSMO_INPUT),*map(str, tests)], cwd=HERE)

    df_new = pd.read_csv(per_csv)
    all_rows.append(df_new)

if all_rows:
    df_out = pd.concat(all_rows, ignore_index=True)
    df_out.to_csv(MASTER_CSV, index=False)
    print(f"\nSaved: {MASTER_CSV}")
else:
    print("\nNo outputs produced.")
