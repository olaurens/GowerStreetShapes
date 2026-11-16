module load dev_tools/jun2020/gcc-10.1.0
set -euo pipefail

# Use script directory as BASE 
SCRIPT="${BASH_SOURCE[0]:-$0}"
BASE="$(cd "$(dirname "$SCRIPT")" && pwd)"
cd "$BASE"

runs=("$@")

mass_bins=("7.5e+12-1.0e+13"
  "1.0e+13-2.5e+13"
  "2.5e+13-5.0e+13")

time_slices=(00040 00060 00080 00100)

# ----- parallelism control -----
# Max number of concurrent covo processes.
# If running under Slurm, this will use allocated CPUs; otherwise default to 6.
NPROC="${SLURM_CPUS_PER_TASK:-6}"

# Simple semaphore: wait until number of background jobs < NPROC
sem() {
  while (( $(jobs -pr | wc -l) >= NPROC )); do
    sleep 2
  done
}

for run in "${runs[@]}"; do
  for ts in "${time_slices[@]}"; do
    for bin in "${mass_bins[@]}"; do

      infile="${BASE}/covo_typefile_${run}_${ts}_potential_plus_com_M${bin}.csv"
      outdir="${BASE}/results_potential_plus_com_${run}_${ts}_M${bin}"
      mkdir -p "$outdir"

      echo "Running covo for $run @ snapshot $ts, M in [${bin}) ... (results in $outdir)"

      sem
      (
        cd "$outdir" && \
        ~/covo/covo ~/covo/covo.params_box "$infile" \
          > "covo_${run}_${ts}_M${bin}.log" 2>&1
      ) &
    done
  done

  # Important: wait for THIS run to finish before returning
  wait
  echo "All covo jobs finished for $run"
done

echo "All z-evolution covo jobs finished."

