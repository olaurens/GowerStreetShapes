"""
Plot a 2D x-y slice (fixed potential catalogue only).
Reads:
    covo_typefile_run243_00100_potential.csv
and saves:
    slice_xy_potential.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy import constants as const
from astropy import units as u
import math

def run_control_par(run) :# reading control.par file from run
    #file_name = f"/Users/orianenyembo/Internship_2025_UCL/control.par"
    file_name = f"/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/{run}/control.par"
    variable = {}
    print("About to read {}".format(file_name))
    with open(file_name, "r") as in_file :
        exec(in_file.read(), {"math": math}, variable)
    dBoxSize = variable["dBoxSize"]
    nGrid = variable["nGrid"]
    dOmega0 = variable["dOmega0"]
    h = variable["h"]
    return dBoxSize, nGrid, dOmega0, h

def fof_file_format_experiment(time_slice,run) :
    fof_type = np.dtype([
       ('position_of_deepest_potential', np.float32, (3,)),
       ('deepest_potential', np.float32, 1),
       ('shrinking_sphere_centre', np.float32, (3,)),
       ('position_of_centre_of_mass', np.float32, (3,)),
       ('velocity_of_centre_of_mass', np.float32, (3,)),
       ('angular_momentum', np.float32, (3,)),
       ('moment_of_inertia', np.float32, (6,)),
       ('velocity_dispersion', np.float32, 1),
       ('r_max', np.float32, 1),
       ('mass', np.float32, 1),
       ('mass_env_0', np.float32, 1),
       ('mass_env_1', np.float32, 1), 
       ('half_mass_radius', np.float32, 1)])

    #file_name = f"/export/koki1/visitor1/Gowersims/runs/runsW/run243fof/fof/run.{time_slice}.fofstats.0"
    #file_name = f"/Users/orianenyembo/Internship_2025_UCL/GowerStreetExtendedSims/runsW/run243/fof/run.{time_slice}.fofstats.0"
    file_name = f"/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/{run}/fof/run.{time_slice}.fofstats.0"
    print("About to read {}".format(file_name))
    with open(file_name, "rb") as in_file:
        data = np.fromfile(in_file, dtype = fof_type)
    print(data['position_of_deepest_potential'])
    print(data['position_of_centre_of_mass'])
    print(data.shape)
    return data

def plot_web(time_slice,run):
    data = fof_file_format_experiment(time_slice,run)
    box_size, nGrid, dOmega0, h= run_control_par(run)
    pos = data['position_of_centre_of_mass']* box_size

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

   #choose the slice
    Z0=0.0
    DZ=10.0
    half = DZ / 2.0
    span = (z > Z0 - half) & (z < Z0 + half)

    xs = x[span]
    ys = y[span]

    print(f"Slicing with z in [{-5, 5}] Mpc/h")
    print(f"Selected haloes: {xs.size}")

    # Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys, s=1, alpha=0.4)

    plt.xlabel("x [Mpc/h]")
    plt.ylabel("y [Mpc/h]")
    plt.title(f"com slice: z from -5 to 5")

    plt.axis("equal")
    plt.tight_layout()

    outname = f"slice_xy_com_{run}_{time_slice}.png"
    plt.savefig(outname, dpi=300)
    plt.close()

    print(f"Saved figure to {outname}")

runs = ['run243', 'run273', 'run355', 'run378', 'run394']
#times = [str(i).zfill(5) for i in range(1, 101)]
#for time_slice in times :
for run in runs :
    plot_web(time_slice='00100', run=run)
