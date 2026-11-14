from pathlib import Path
from io import StringIO
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Read covo output
# =========================================================
def read_covo_output(file_name):
    """
    Read a covo output file.

    Column convention:
      col 0 : bin index
      col 1 : r
      col 3 : r12_v1a
      col 4 : sigma(r12_v1a)
      col 5 : r12_v1b
      col 6 : sigma(r12_v1b)
    """
    file_path = Path(file_name).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    return data, file_path


# =========================================================
# Parse run / slice / mass_bin from path
# Example parent dir:
# results_potential_plus_com_run243_00040_M1.0e+13-2.5e+13
# =========================================================
def parse_run_slice_mass(path: Path):
    parent = Path(path).resolve().parent.name

    m = re.search(r"run(\d+)_([0-9]{5})_M(.+)$", parent)
    if not m:
        raise ValueError(f"Could not parse run/slice/mass from path: {path}")

    run = int(m.group(1))
    slice_id = m.group(2)
    mass_bin = m.group(3)
    return run, slice_id, mass_bin


# =========================================================
# Read z_values.txt and build slice -> z map
# =========================================================
def build_slice_to_z(z_values_path: Path, which="mid"):
    """
    Reads z_values.txt and returns:
        {"00040": z40, "00060": z60, ...}

    Expected columns:
      Step,z_far,z_near,delta_z,...
    ignoring lines starting with '#'
    """
    z_values_path = Path(z_values_path).expanduser().resolve()
    if not z_values_path.exists():
        raise FileNotFoundError(f"z_values.txt not found: {z_values_path}")

    names = [
        "Step", "z_far", "z_near", "delta_z",
        "cmd_far", "cmd_near", "delta_cmd",
        "cmd_box_far", "cmd_box_near", "delta_cmd_box",
        "cmvolume", "cmvolume_boxvolume"
    ]

    lines = z_values_path.read_text().splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    dfz = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=names)

    if which == "mid":
        zrep = 0.5 * (dfz["z_far"] + dfz["z_near"])
    elif which == "near":
        zrep = dfz["z_near"]
    elif which == "far":
        zrep = dfz["z_far"]
    else:
        raise ValueError("which must be 'mid', 'near', or 'far'")

    slice_to_z = {
        f"{int(step):05d}": float(z)
        for step, z in zip(dfz["Step"], zrep)
    }
    return slice_to_z


def get_slice_to_z_for_run(run, runsW_root=Path("/share/testde/ucapwhi/GowerStreetExtendedSims/runsW"), which="mid"):
    """
    run can be int like 243 or string like '243' / 'run243'
    """
    if isinstance(run, str):
        if run.startswith("run"):
            run = int(run[3:])
        else:
            run = int(run)

    zfile_run = Path(runsW_root) / f"run{run:03d}" / "z_values.txt"
    return build_slice_to_z(zfile_run, which=which)


# =========================================================
# Plot: same slice, different mass bins
# z is same for all curves -> best in title
# =========================================================
def plot_data_correl_mb(data, file_path, ax0, rmin, ax0_inset=None):
    stem = file_path.parent.name
    mass_part = stem.split("_M", 1)[1]

    shift = 0.5
    r = data.iloc[:, 1]
    r12_v1a = data.iloc[:, 3] - shift
    r12_v1a_std = data.iloc[:, 4]

    mask_r = r > rmin

    ax0.errorbar(
        r, r12_v1a, yerr=r12_v1a_std,
        label=rf"vector $a$, $M={mass_part}\,[M_\odot\,h^{{-1}}]$"
    )

    if ax0_inset is not None:
        m = mask_r & (r12_v1a > 0)
        if np.any(m):
            ax0_inset.plot(r[m], r12_v1a[m], label=rf"$M={mass_part}$")


# =========================================================
# Plot: same mass bin, different slices
# z changes for each curve -> best in legend
# =========================================================
def plot_data_correl_rs(data, label_text, ax0, rmin, ax0_inset=None):
    shift = 0.5

    r = data.iloc[:, 1]
    r12_v1a = data.iloc[:, 3] - shift
    r12_v1a_std = data.iloc[:, 4]

    mask_r = r > rmin

    ax0.errorbar(
        r, r12_v1a, yerr=r12_v1a_std,
        label=label_text
    )

    if ax0_inset is not None:
        m = mask_r & (r12_v1a > 0)
        if np.any(m):
            ax0_inset.plot(r[m], r12_v1a[m], label=label_text)


# =========================================================
# Run: multi mass bins at one slice
# =========================================================
def run_multiM(file_paths):
    data0, path0 = read_covo_output(file_paths[0])
    run, slice_id, mass_bin = parse_run_slice_mass(path0)

    slice_to_z = get_slice_to_z_for_run(run, which="mid")
    z = slice_to_z.get(f"{int(slice_id):05d}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax.inset_axes([0.6, 0.15, 0.35, 0.35])

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Large r (log y)", fontsize=9)
    ax2.grid(True, alpha=0.3)

    rmin = 20

    for fp in file_paths:
        data, this_path = read_covo_output(fp)
        plot_data_correl_mb(data, this_path, ax, rmin=rmin, ax0_inset=ax2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel("Correlation")

    if z is None:
        ax.set_title(
            rf"Correlation function of major axis $a$ "
            rf"(run{run:03d}, snapshot {slice_id})"
        )
    else:
        ax.set_title(
            rf"Correlation function of major axis $a$ "
            rf"(run{run:03d}, snapshot {slice_id}, $z={z:.3f}$)"
        )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.8)

    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent
    outpath = script_dir / f"corr_run{run:03d}_{slice_id}_multiM_largeR.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {outpath}")


# =========================================================
# Run: multi slices at one mass bin
# =========================================================
def run_multiZ(file_paths):
    data0, path0 = read_covo_output(file_paths[0])
    run, slice_id, mass_bin = parse_run_slice_mass(path0)

    slice_to_z = get_slice_to_z_for_run(run, which="mid")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax.inset_axes([0.6, 0.15, 0.35, 0.35])

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Large r (log y)", fontsize=9)
    ax2.grid(True, alpha=0.3)

    rmin = 20

    for fp in file_paths:
        data, this_path = read_covo_output(fp)
        run_this, slice_this, mass_bin_this = parse_run_slice_mass(this_path)

        z_this = slice_to_z.get(f"{int(slice_this):05d}")
        if z_this is None:
            label_this = rf"vector $a$, snapshot {slice_this}"
        else:
            label_this = rf"vector $a$, snapshot {slice_this} ($z={z_this:.3f}$)"

        plot_data_correl_rs(data, label_this, ax, rmin=rmin, ax0_inset=ax2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel("Correlation")
    ax.set_title(
        rf"Correlation function of major axis $a$ "
        rf"(run{run:03d}, $M={mass_bin}\,[M_\odot\,h^{{-1}}]$)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent
    outpath = script_dir / f"corr_run{run:03d}_M{mass_bin}_multiZ_largeR.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {outpath}")


# =========================================================
# Main
# Usage:
#   python correlation_function_with_z.py multiM file1 file2 file3 ...
#   python correlation_function_with_z.py multiZ file1 file2 file3 ...
# =========================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python correlation_function_with_z.py multiM <file1> <file2> ...")
        print("  python correlation_function_with_z.py multiZ <file1> <file2> ...")
        sys.exit(1)

    mode = sys.argv[1]
    file_paths = [Path(f).expanduser().resolve() for f in sys.argv[2:]]

    if mode == "multiM":
        run_multiM(file_paths)
    elif mode == "multiZ":
        run_multiZ(file_paths)
    else:
        raise ValueError("mode must be 'multiM' or 'multiZ'")
