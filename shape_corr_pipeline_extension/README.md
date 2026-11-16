# GowerStreetShapes

Welcome to the Gower Street Shapes repository. Here you can find all of the codes I made to study Dark Matter halos shape and intrinsic alignment from Gower Street simulations outputs (FOF Outputs).

**Contact :** _Oriane.Laurens[at]etu.univ-grenoble-alpes.fr/oriane.laurens[at]outlook.fr_

## Understanding shapes
- **Halo Shaper** is a little side project that helps visualize the influence of semi-axis ratios parameters on an ellipsoid. It includes a projection code that allows to go from a 3D ellipsoid to a 2D ellipse projected with a "stick" (length = ellipticity module; angle = orientation, half of ellipticity argument).
  
## Halo Shape Distribution 
- The **Shape** * code generates a csv file with semi axis of halos at each redshift for every chosen runs. It can be used before the two codes below to avoid great computing time when trying to plot. If those are not needed specifically, run a more relevant code.
- The **Shape Distribution** * code allows to plot the distribution of halo shape for a given run and time slice (and corresponding redshift). If multiple time slices are generated for a single run, using ffmpeg helps visualising the evolution with redshift with a video. You can change it to use the previous Shape code's outputs.
- The **Ratios Moments** * code is made to calculate the main first order moments such as mean, variance and skewness with error bars to visualise statistically the evolution across redshift rather than visually with the video that can be hard to use to compare different runs. You can change it to use the previous Shape code's outputs.

## Shape correlations
- **FOF to Covo** is a small script that converts a FOF output file into a csv readable by Covo, a 3D correlation code developped by Kai Hoffman, available on Bitbucket.
- The **Correlation Functions** code is made to plot the correlation functions of chosen parameters from a Covo output.

**Note on covo usage** : To run it, type : ./covo covo.params_parameter-file-ID

  _\* = You better use an HPC to run these ones on multiple redshifts. I haven't found a way to optimise/parallelise this efficiently but the processing is very slow as it contains a loop to read each row of a binary file when reading the FOF output._

## Shape Correlation Pipeline Extensions

The scripts in this section were developed during the 2025-2026 UCL MSci project by Eason Tian, extending the original shape-correlation workflow and introducing cosmology-dependent amplitude modelling and Gaussian Process emulation.

### Amplitude Extraction

- The **amp_lib.py** code is a mini library thatcontains helper functions used throughout the amplitude extraction and fitting process.
- The **amplitude_extraction.py** is made to fit correlation functions relative to a reference template and extract a scalar alignment amplitude for each run, redshift slice and mass bin
- The **build_amplitude_table.py** code gathers all extracted amplitudes together with the corresponding cosmological parameters into a single table that can be used for further analysis
- **add_halo_bias_to_amplitude_table.py** allows halo bias estimates to be incorporated into the amplitude tables.
- **merge_master_cosmo.py** combines cosmological information and amplitude measurements into consolidated datasets

These scripts extract alignment amplitudes from COVO correlation-function outputs and build consolidated tables for cosmological analysis.

### Emulator and Cosmology Dependence

- These **gpr_*.py** codes are made to study how alignment amplitudes vary with cosmological parameters and to build Gaussian Process emulators capable of predicting amplitudes for unseen cosmologies.
- **validate_gp_reconstruct_A.py** performs leave-one-out tests by withholding one simulation run at a time and comparing emulator predictions against the measured amplitudes

These scripts train and validate Gaussian Process emulators describing the dependence of alignment parameters on cosmological quantities.

### Automated Processing

- **process_run_and_cleanup.py** automates the processing of individual simulation runs and organises the resulting outputs.
- **process_runs_prefetch_pipeline.py** is designed to manage batches of simulation runs and prepare the data products required for amplitude extraction and emulator training.
- These **run_covo_potential*.sh** scripts automate the execution of COVO on multiple runs, redshift slices and mass bins.

Automation tools for large-scale processing of simulation runs and COVO outputs.

### Correlation Function Variants

Several correlation-function scripts are retained because they correspond to different analysis stages and visualisation variants developed during the project:

- **correlation_function.py** is the original implementation from oriane's code.
- **correlation_function_modified.py** is the main modified version used during the final analysis.
- **correlation_function_with_z.py** includes redshift-dependent analysis.
- **correlation_function_modified_heatmap.py** generates heatmap visualisations.
- **correlation_function_modified_inset.py** generates figures with inset panels.

These files document the evolution of the analysis pipeline and were used for different diagnostic and presentation purposes.

## Contributors

Original repository structure and halo-shape analysis tools:
- Oriane Laurens

Shape-correlation extensions, amplitude modelling and emulator development:
- Eason Tian
