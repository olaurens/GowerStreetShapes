"""
This script converts FOF halo catalogues into the covo input format required
to compute intrinsic-alignment correlation functions. 

Compared to the original version(written by Oriane Laurens, June 2025), this
script no longer selects halos via a simple top-percentile mass cut. Instead,
it allows the user to define fixed halo mass bins so that the resulting catalogs
are consistent across snapshots,enabling a clean measurement of redshift evolution.

Eason Tian
December 2025
"""
# IMPORTS
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
from numba import njit, prange
from astropy import units as u
import math
mp.rcParams['agg.path.chunksize'] = 10000 # to adjust size (suggested by a previous error)

# READING THE FOF FILE

def fof_file_format_experiment(time_slice, run) :
    fof_type = np.dtype([
       ('position_of_deepest_potential', np.float32, (3,)),
       ('deepest_potential', np.float32, (1,)),
       ('shrinking_sphere_centre', np.float32, (3,)),
       ('position_of_centre_of_mass', np.float32, (3,)),
       ('velocity_of_centre_of_mass', np.float32, (3,)),
       ('angular_momentum', np.float32, (3,)),
       ('moment_of_inertia', np.float32, (6,)),
       ('velocity_dispersion', np.float32, (1,)),
       ('r_max', np.float32, 1),
       ('mass', np.float32, 1),
       ('mass_env_0', np.float32, 1),
       ('mass_env_1', np.float32, 1), 
       ('half_mass_radius', np.float32, 1)])
    file_name = f"/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/{run}/fof/run.{time_slice}.fofstats.0"
    print("About to read {}".format(file_name))
    with open(file_name, "rb") as in_file:
        data = np.fromfile(in_file, dtype = fof_type)
    print(data.shape)
    return data

# GET BOX SIZE
def box_size_function(run) :# reading control.par file from run
    #file_name = f"/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/{run}/control.par"
    file_name = f"/share/testde/ucapwhi/GowerStreetExtendedSims/runsW/{run}/control.par"
    #file_name = f"/import/home/visitor1/control.par"
    variable = {}
    print("About to read {}".format(file_name))
    with open(file_name, "r") as in_file :
        exec(in_file.read(), {"math": math}, variable)
    dBoxSize = variable["dBoxSize"]
    return dBoxSize

# GET RATIOS FROM SEMI-AXIS AND INERTIA MATRIX
def axis_vect(time_slice, run) :
    data = fof_file_format_experiment(time_slice, run)
    inertia = data['moment_of_inertia'] #'loading' inertia tensor file data
    eigenvects = []
    mass = data['mass']
    quantile = np.percentile(mass, 90) #90th percentile
    for mass_row, inertia_row in zip(mass, inertia) :
        """Construction of a, b, c from inertia matrix
        We have 6 independent components
        Diagonal terms : d1=Ixx; d2=Iyy; d3=Izz
        Triangle terms : t1=Ixy=Iyx; t2=Ixz=Izx; t3=Iyz=Izy
        + Output is two ratios, first c/a then b/a"""
        if mass_row > quantile : #only selects above the 90-th centile so only the top 10%
            d1, t1, t2, d2, t3, d3 = inertia_row #get matrix independent components from file
            I = np.array([[d1,t1,t2], [t1,d2,t3], [t2,t3,d3]]) #reconstructing inertia matrix
            eigenval, eigenvect = np.linalg.eigh(I) #get eigen values sorted with a>=b>=c
            eigenvects.append(eigenvect)
    return eigenvects

#CALCULATE THE EIGENVECTOR AND EIGENVALUE
@njit(parallel=True)
def cal(inertia, mass, quantile):
    n = mass.shape[0]
    # predistribute matrix
    A = np.empty((n, 3), np.float32)
    B = np.empty((n, 3), np.float32)
    keep = 0
    for i in prange(n):
        if mass[i] <= quantile:
            continue
        d1, t1, t2, d2, t3, d3 = inertia[i]
        # construct inertia matrix
        I = np.array([[d1,t1,t2], [t1,d2,t3], [t2,t3,d3]])
        # eigenvalue
        evals, evecs = np.linalg.eigh(I)
        a_vec = evecs[:, 2]#longest eigenvalue=a
        c_vec = evecs[:, 0]
        norm_a = math.sqrt(a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2)
        norm_c = math.sqrt(c_vec[0]**2 + c_vec[1]**2 + c_vec[2]**2)
        if norm_a > 0.0 and norm_c > 0.0:
            A[i, 0] = a_vec[0] / norm_a
            A[i, 1] = a_vec[1] / norm_a
            A[i, 2] = a_vec[2] / norm_a
            B[i, 0] = c_vec[0] / norm_c
            B[i, 1] = c_vec[1] / norm_c
            B[i, 2] = c_vec[2] / norm_c
            keep += 1   
    return A,B
#try different types of postion argument
def get_positions(data, box_size, mode="potential_plus_com"):
    if mode == "com":
        pos = data["position_of_centre_of_mass"]
    elif mode == "shrink":
        pos = data["shrinking_sphere_centre"]
    elif mode == "potential":
        pos = data["position_of_deepest_potential"]
    elif mode == "potential_plus_com":
        pos = (data["position_of_deepest_potential"] 
               + data["position_of_centre_of_mass"])
    else:
        raise ValueError(f"Unknown position mode: {mode}")
    return pos * box_size

# STORE DATA FOR COVO
def get_data(time_slice,run,pos_mode="potential_plus_com", mass_bin=None):
    data = fof_file_format_experiment(time_slice, run)
    box_size = box_size_function(run)
    
    RHO_CRIT = 2.77536627208e11  # [Msun / (Mpc/h)^3], critical density factor
    mass = (box_size**3)*(RHO_CRIT)*data['mass']# # Convert to physical mass in Msun/h
    
    print(f"=== {run} @ {ts} ===, N_halo = {mass.size}, mass min  = {mass.min():.3e}, mass max  = {mass.max():.3e}")
    inertia = data['moment_of_inertia']
    angular_momentum = data['angular_momentum']
    posmode="potential_plus_com"
    pos = get_positions(data, box_size, mode=pos_mode)
    mmin, mmax = mass_bin
    mass_mask = (mass >= mmin) & (mass < mmax)

    # using mask to select properties inside this mass range
    mass_sel = mass[mass_mask]
    inertia_sel = inertia[mass_mask]
    pos_sel = pos[mass_mask]

    A,B = cal(inertia_sel, mass_sel, quantile=0)#ignore quantile
    halo_data = np.column_stack((pos_sel, A,B))
    data_file = pd.DataFrame(halo_data)
    data_file.to_csv(f'covo_typefile_{run}_{time_slice}_{pos_mode}_M{mmin:.1e}-{mmax:.1e}.csv', index=False, header=False)
# EXECUTION
#runs = ["run207","run236"]
##runs = ["run092", "run119", "run144", "run189", "run207",
        #"run221", "run236", "run240", "run243", "run255",
        #"run257", "run311", "run340", "run344", "run371",
        #"run378", "run407", "run437", "run510", "run776",
        #"run790", "run823", "run890",]

runs = [sys.argv[1]]

time_slices = [ '00040','00060', '00080', '00100']

#time_slices = ['00050']
#time_slices = ['00100']
#compute mass bins
#exponents = [12, 13, 14, 15]
# Physical mass bins in Msun/h, split by factors of ~2.5
mass_bins = [
    (7.5e12, 1e13),
    (1e13, 2.5e13),
    (2.5e13, 5e13),
    ]#discard halos with the smallest and largest mass, as they either explode or halo number is too small 
print("Using physical mass bins (Msun/h):", mass_bins)

for run in runs:
        for ts in time_slices:
            for mode in [ "potential_plus_com"]: 
                for mmin, mmax in mass_bins:
                    print(f"=== {run} @ {ts}, M in [{mmin:.1e}, {mmax:.1e}) ===")
                    get_data(ts,run,pos_mode="potential_plus_com",mass_bin=(mmin, mmax))
