#!/usr/bin/env python3
"""
Minimal pipeline for selected GowerStreet runs:
1) unpack run + fof
2) run covo (potential, z bins)
3) build amplitude table
4) append to master CSV
5) clean up unpacked + covo files
"""

import subprocess
import shutil
from pathlib import Path
import pandas as pd

# ======================
# User settings
# ======================
BASE = Path("/share/testde/ucapwhi/GowerStreetExtendedSims")
RUNSW = BASE / "runsW"

#RUNS = [92,119,144,189,207,221,236,240,243,255,257,311,340,344,371,378,407,437,510,776,790,823,890]
RUNS = sorted(int(p.name.split(".")[0].replace("run","")) for p in RUNSW.glob("run*.tar.gz"))


WHICH_VEC  = "a"
RMIN       = 10.0
MIN_COUNTS = 100

HERE = Path(__file__).resolve().parent
BUILD_PY   = HERE / "build_amplitude_table.py"
#COVO_SH    = HERE / "run_covo_potential_z_bin.sh"
COVO_SH    = HERE / "covo_batch_z_pipeline.sh"
TYPEFILE_PY = HERE / "fof_to_covo_massbins.py"
MASTER_CSV = HERE / f"amplitude_table_vec{WHICH_VEC}_rmin{RMIN}.csv"


def run(cmd, cwd=None):
    """
    Run an external shell command with logging and fail-fast behaviour.
    """
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, cwd=str(cwd) if cwd else None)

# detect already-done runs
done = set()
if MASTER_CSV.exists():
    df = pd.read_csv(MASTER_CSV)
    if "run" in df.columns:
        done = set(df["run"].astype(str).map(lambda x: x if x.startswith("run") 
        else f"run{int(float(x)):03d}").unique())

for r in RUNS:
    run_name = f"run{r:03d}"
    print(f"\n===== {run_name} =====")

    if run_name in done:
        print("Already processed, skipping.")
        continue

    run_dir = RUNSW / run_name


    # 1) unpack run + fof (in RUNSW so paths match archives)
    tar_main = RUNSW / f"{run_name}.tar.gz"
    tar_fof  = RUNSW / f"{run_name}.fof.tar.gz"
    if not run_dir.exists():
        run(["tar", "-xzf", tar_main], cwd=RUNSW)
        run(["tar", "-xzf", tar_fof], cwd=RUNSW)
    else:
        print("Run directory already exists; skipping untar.")

    # 2) run covo
    run(["python", TYPEFILE_PY, run_name], cwd=HERE)
    run(["bash", COVO_SH, run_name], cwd=HERE)

    # 3) build amplitude table for this run
    tests = sorted(run_dir.glob(f"**/results_potential_plus_com_{run_name}_?????_M*/test"))
    if not tests:
        tests = sorted(HERE.glob(f"results_potential_plus_com_{run_name}_?????_M*/test"))

    per_csv = HERE / f"_tmp_{run_name}_vec{WHICH_VEC}_rmin{RMIN}.csv"
    run(["python", BUILD_PY, str(r), str(per_csv), *tests], cwd=HERE)

    # 4) append to master CSV
    df_new = pd.read_csv(per_csv)
    if MASTER_CSV.exists():
        df_old = pd.read_csv(MASTER_CSV)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(MASTER_CSV, index=False)

    # 5) cleanup
    print("Cleaning up...")
    p = run_dir.resolve()
    shutil.rmtree(p)
    #per_csv.unlink(missing_ok=True)

print("\nAll done.")

