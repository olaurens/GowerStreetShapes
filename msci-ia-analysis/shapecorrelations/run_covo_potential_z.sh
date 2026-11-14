# This script launches covo potential calculations for several snapshots of a
# single simulation run. For a chosen run (e.g., run243), the script loops over
# selected snapshots (e.g., 00040, 00070, 00100). For each snapshot, the script
# constructs the input CSV file and executes covo inside a dedicated output
# directory to keep results cleanly separated.
#
# Each run's snapshot pair produces its own directory:
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

# Choose the simulation run to process
run=run243

# Loop over selected snapshots (time slices) for this run
for ts in 00040 00060 00080 00100; do

    infile="${BASE}/covo_typefile_${run}_${ts}_potential.csv"
    
    # Create an output directory for each run + snapshot
    outdir="${BASE}/results_potential_${run}_${ts}"
    mkdir -p "$outdir"

    echo "Running covo for $run at snapshot $ts ... (results → $outdir)"

    # Run covo inside its own directory
    nohup bash -c "cd '$outdir' && ~/covo/covo ~/covo/covo.params_box '$infile'" \
        > "${outdir}/covo_${run}_${ts}_potential.log" 2>&1 &
done

echo "All z-evolution covo jobs launched."
