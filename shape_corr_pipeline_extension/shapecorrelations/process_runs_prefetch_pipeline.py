#!/usr/bin/env python3
"""
Minimal pipeline for selected GowerStreet runs:
1) unpack run + fof (with optional prefetch/async untar)
2) run covo (potential, z bins)
3) build amplitude table
4) append to master CSV
5) clean up unpacked run dir
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

#RUNS = sorted(int(p.name.split(".")[0].replace("run", "")) for p in RUNSW.glob("run*.tar.gz"))

RUNS = sorted(int(p.name.replace("run", "").replace(".tar.gz", ""))
    for p in RUNSW.glob("run[0-9][0-9][0-9].tar.gz"))


WHICH_VEC  = "a"
RMIN       = 10.0

HERE = Path(__file__).resolve().parent
BUILD_PY    = HERE / "build_amplitude_table.py"
COVO_SH     = HERE / "covo_batch_z_pipeline.sh"
TYPEFILE_PY = HERE / "fof_to_covo_massbins.py"

MASTER_CSV = HERE / f"amplitude_table_vec{WHICH_VEC}_rmin{RMIN}_noref.csv"



def run(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, cwd=str(cwd) if cwd else None)

def run_name_of(r: int) -> str:
    return f"run{r:03d}"


# async untar 
untar_jobs = {}  # run_name -> Popen

def start_unpack(run_name: str):
    """Start untar in background if needed; return Popen or None."""
    run_dir = RUNSW / run_name
    if run_dir.exists():
        return None

    tar_main = RUNSW / f"{run_name}.tar.gz"
    tar_fof  = RUNSW / f"{run_name}.fof.tar.gz"

    if not tar_main.exists():
        raise FileNotFoundError(tar_main)
    if not tar_fof.exists():
        raise FileNotFoundError(tar_fof)

    # One background process that untars both archives sequentially
    cmd = [
        "bash", "-lc",
        f"tar -xzf '{tar_main.name}' && tar -xzf '{tar_fof.name}'"
    ]
    print(f">> (prefetch) untar {run_name} in background")
    p = subprocess.Popen(cmd, cwd=str(RUNSW))
    untar_jobs[run_name] = p
    return p
   
def ensure_unpacked(run_name: str):
    """Ensure run_name is fully untarred (main + fof). Wait if prefetched."""
    run_dir = RUNSW / run_name

    # If we scheduled a prefetch job, always wait for it to finish (even if run_dir already exists).
    p = untar_jobs.get(run_name)
    if p is not None:
        print(f">> waiting for prefetch untar: {run_name}")
        rc = p.wait()  # 0 = success
        if rc != 0:
            raise RuntimeError(f"Prefetch untar failed for {run_name} (exit {rc})")
        # job finished; remove handle
        untar_jobs.pop(run_name, None)

    # After waiting (or if not prefetched), ensure required files exist; if not, do a sync untar.
    fofstats = run_dir / "fof" / "run.00040.fofstats.0"
    if not fofstats.exists():
        # Not unpacked (or unpack incomplete): do it synchronously now
        tar_main = RUNSW / f"{run_name}.tar.gz"
        tar_fof  = RUNSW / f"{run_name}.fof.tar.gz"
        run(["tar", "-xzf", tar_main.name], cwd=RUNSW)
        run(["tar", "-xzf", tar_fof.name], cwd=RUNSW)


# detect already-done runs
done = set()
if MASTER_CSV.exists():
    df = pd.read_csv(MASTER_CSV)
    if "run" in df.columns:
        done = set(df["run"].astype(str).map(lambda x: x if x.startswith("run") 
        else f"run{int(float(x)):03d}").unique())


# main loop
for i, r in enumerate(RUNS):
    run_name = run_name_of(r)
    print(f"\n===== {run_name} =====")

    #if run_name in done:
        #print("Already processed, skipping.")
        #continue

    # 0) ensure current run is unpacked (wait if prefetched)
    #ensure_unpacked(run_name)

    # 0b) prefetch next run untar
    #if i + 1 < len(RUNS):
    #    next_run = run_name_of(RUNS[i + 1])
        # only prefetch if not already done / not already scheduled
    #    if next_run not in done and next_run not in untar_jobs:
    #        start_unpack(next_run)

    run_dir = RUNSW / run_name

    # 1) build typefiles
    #run(["python", str(TYPEFILE_PY), run_name], cwd=HERE)

    # 2) run covo (z bins)
    #run(["bash", str(COVO_SH), run_name], cwd=HERE)

    # 3) build amplitude table for this run
    tests = sorted(run_dir.glob(f"**/results_potential_plus_com_{run_name}_?????_M*/test"))
    if not tests:
        tests = sorted(HERE.glob(f"results_potential_plus_com_{run_name}_?????_M*/test"))

    per_csv = HERE / f"_tmp_{run_name}_vec{WHICH_VEC}_rmin{RMIN}.csv"
    run(["python", str(BUILD_PY), str(r), str(per_csv), *map(str, tests)], cwd=HERE)

    # 4) append to master CSV
    df_new = pd.read_csv(per_csv)
    if MASTER_CSV.exists():
        df_old = pd.read_csv(MASTER_CSV)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(MASTER_CSV, index=False)

    # 5) cleanup run dir (free space)
    print("Cleaning up unpacked run directory...")
    #shutil.rmtree(run_dir.resolve())

print("\nAll done.")

