# This script launches covo potential calculations for several snapshots of a
# single simulation run. For a chosen run (e.g., run243), the script loops over
# selected snapshots (e.g., 00040, 00070, 00100). For each snapshot, the script

# constructs the input CSV file and executes covo inside a dedicated output
# directory to keep results cleanly separated.
#
# Each run-snapshot pair produces its own directory:
#     results_potential_<run>_<snapshot>/
# where both covo output and the log file are stored.
#
# This workflow is useful for analysing redshift evolution while preventing
# file overwriting from different snapshots.
#
# Eason Tian
# December 2025
# Load the GCC module used to compile covo
module load dev_tools/jun2020/gcc-10.1.0

BASE=~/dm-research/shapecorrelations
cd "$BASE"
BASE="$(pwd)"

# Choose the simulation run to process
#runs=(run092 run119 run144 run189 run207 run221 run236 run240 run243 run255 run257 run311 run340 run344 run371 run378 run407 run437 run510 run776 run790 run823 run890)

runs=("$@")

mass_bins=(
  "7.5e+12-1.0e+13"
  "1.0e+13-2.5e+13"
  "2.5e+13-5.0e+13"
)
for run in "${runs[@]}"; do
  for bin in "${mass_bins[@]}"; do
      # Loop over selected snapshots (time slices) for this run
      for ts in 00040 00060 00080 00100 ; do
      #for ts in 00050  ; do
          Mmin="${bin%-*}"
          Mmax="${bin#*-}"

          infile="${BASE}/covo_typefile_${run}_${ts}_potential_plus_com_M${bin}.csv"

          # Create an output directory for each run + snapshot
          outdir="${BASE}/results_potential_plus_com_${run}_${ts}_M${bin}"
          mkdir -p "$outdir"
          echo "Running covo for $run @ snapshot $ts, M in [${bin}) ... (results in $outdir)"

          bash -c "cd '$outdir' && ~/covo/covo ~/covo/covo.params_box '$infile'" \
            > "${outdir}/covo_${run}_${ts}_M${bin}.log" 2>&1 
      done
  done
done  
echo "All z-evolution covo jobs launched."
