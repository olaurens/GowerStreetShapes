# ------------------------------------------------------------------------------
# This script launches covo potential calculations for multiple simulation runs
# at snapshot 00100. For each run, the script creates a dedicated output
# directory and executes covo inside it, ensuring clean separation of results
# and log files across different runs.
#
# The GCC module used to compile covo is loaded to guarantee compatibility with
# libstdc++. The script then loops over selected runs (e.g., run243, run273, …)
# and constructs the correct input CSV file for each run. Every covo execution
# is launched in the background via 'nohup', with all output redirected into a
# per-run log file stored inside the corresponding result directory.
#
# Eason Tian
# December 2025

# Load the same GCC module that was used when compiling covo

module load dev_tools/jun2020/gcc-10.1.0

BASE=~/dm-research/shapecorrelations
cd "$BASE"

# Batch-run potential calculations
for run in run243 run273 run355 run378 run394; do
    
    infile="${BASE}/covo_typefile_${run}_00100_potential.csv"
    
    workdir="${BASE}/results_potential_${run}_00100"
    mkdir -p "$workdir"
    
    nohup bash -c "cd '$workdir' && ~/covo/covo ~/covo/covo.params_box '$infile'" \
        > "${workdir}/covo_${run}_00100_potential.log" 2>&1 &echo "Running covo for $run ... (results will be stored in: $workdir)"
done

echo "All covo jobs launched."
