#! /usr/bin/env python3
import argparse
from dataclasses import dataclass
import string
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from turbulence_statistics import LAADType, Neighbourhood, RelativeTo


@dataclass
class Dataset:
    prefix: str
    shape: tuple[int, int, int]
    slice_index: int
    dx: float
    re: float


latex_textwidth = 5.77
scale_factor = 2
fig_width = scale_factor * latex_textwidth

MIN_SIZE = scale_factor * 5
SMALL_SIZE = scale_factor * 7  # pt
MEDIUM_SIZE = scale_factor * 10  # pt
BIGGER_SIZE = scale_factor * 11  # pt

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('legend', title_fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('xtick.major', pad=4)
plt.rc('ytick.major', pad=4)
plt.rc('axes', labelpad=12)
plt.rc('axes', titlepad=8)

plt.rc('text.latex', preamble=r"""
\usepackage{microtype}
\usepackage{mlmodern}
\usepackage{siunitx}

\newcommand{\Rethroat}{\text{Re}_{\text{throat}}}
""")
plt.rc("text", usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

plt.rc("savefig", dpi=300)

fda_dataset_list = [
    Dataset("laad_fda_re500", (60, 60, 1000), 30, 0.0002, 500),
    Dataset("laad_fda_re2000", (60, 60, 1000), 30, 0.0002, 2000),
    Dataset("laad_fda_re3500", (60, 60, 1000), 30, 0.0002, 3500),
    Dataset("laad_fda_re5000", (60, 60, 1000), 30, 0.0002, 5000),
    Dataset("laad_fda_re6500", (60, 60, 1000), 30, 0.0002, 6500),
]

karman_extent_cm = [0, 200, 0, 50]
karman_dx_cm = 1

fda_extent_cm = [0, 20, 0, 1.2]
fda_dx = 0.0002

window_sizes = np.arange(1, 41)
base_figure_dir = Path(__file__).parent / "figures" / "thesis"


def tick_formatter(num_decimals: int) -> FuncFormatter:
    def custom_formatter(x, pos):
        if x == 0:
            return "0"
        
        return f"{x:.{num_decimals}f}"

    return FuncFormatter(custom_formatter)


def laad_type_str(laad_type):
    components = [str(laad_type.neighbourhood_shape), str(laad_type.relative_to)]
    dirname = "_".join(components)
    return dirname


def laad_type_dir(base_dir: Path, laad_type: LAADType) -> Path:
    return base_dir / laad_type_str(laad_type)


def get_data(file_prefix: str, laad_type: LAADType, window_size: int, dataset: str = "fda", index: str = "laad") -> np.ndarray:
    base_dir = Path(__file__).parent / "output" / dataset / index
    laad_dir = laad_type_dir(base_dir, laad_type)
    file_path = laad_dir / f"{file_prefix}_{window_size}.npy"
    if not file_path.exists():
        raise ValueError("File does not exist:", file_path.resolve())
    return np.load(file_path)


def get_window_size_dataset(file_prefix: str, laad_type: LAADType, dataset: str = "fda") -> np.ndarray:
    dataset_list = []

    for window_size in window_sizes:
        dataset_list.append(get_data(file_prefix, laad_type, window_size, dataset))

    return np.stack(dataset_list, axis=0)


def classify_points(threshold: float, ref_1: np.ndarray, ref_2: np.ndarray) -> np.ndarray:
    result = np.zeros((*ref_1.shape, 3), dtype=np.int32)
    result[np.logical_and(ref_1 < threshold, ref_2 < threshold), 0] = 255  # red, indicating low value, flat
    result[np.logical_and(ref_1 < threshold, ref_2 > threshold), 1] = 255  # green, indicating rising value
    result[np.logical_and(ref_1 > threshold, ref_2 > threshold), 2] = 255  # blue, indicating high value, flat
    result[np.logical_and(ref_1 > threshold, ref_2 < threshold), 0] = 255  # yellow, indicating high to low
    result[np.logical_and(ref_1 > threshold, ref_2 < threshold), 1] = 255  # yellow, indicating high to low
    result[np.isnan(ref_1), 0] = 255
    result[np.isnan(ref_1), 1] = 255
    result[np.isnan(ref_1), 2] = 255
    return result


def laad_raw_karman(laad_type: LAADType):
    figure_path = base_figure_dir / "Chapter5" / "laad_raw_karman.pdf"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, layout="compressed", figsize=(fig_width, 7.5), sharex=True)
    window_sizes = [1, 5, 10]

    for ax, ws in zip(axes, window_sizes):
        ax.set_title(f"Window size = \SI{{{ws * karman_dx_cm}}}{{\cm}}")
        data = get_data("laad_karman_re1000", laad_type, ws, dataset="karman")
        im = ax.imshow(data.T, extent=karman_extent_cm, vmin=0, vmax=1)
        ax.set_yticks([0, 25, 50])
    cb = fig.colorbar(im, ax=axes)
    cb.set_label("Index value")
    cb.set_ticks(np.arange(0, 1.1, 0.1))

    fig.suptitle("Locally-averaged angular deviation for the Kárman vortex street at $z = 5 \,\si{cm}$")

    fig.savefig(figure_path)


def inner_product_raw_comparison(laad_type: LAADType):
    # General plot code
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 4.2), layout="constrained", height_ratios=[0.475, 0.475, 0.05])
    fig.suptitle("Local sum of inner products for the FDA nozzle at $x = 0 \, \si{cm}$")

    # Top subplot
    ds = fda_dataset_list[0]
    metric_values = get_data(ds.prefix, laad_type, 5, index="inner_product")
    axes[0].set_title("$\Rethroat = 500$")
    axes[0].set_xticks(np.arange(0, 21, 1), labels="")
    axes[0].set_yticks([0, 0.6, 1.2])
    axes[0].yaxis.set_major_formatter(tick_formatter(1))
    axes[0].imshow(metric_values, vmin=0, vmax=1, extent=fda_extent_cm)
    
    # Bottom subplot
    ds = fda_dataset_list[1]
    metric_values = get_data(ds.prefix, laad_type, 5, index="inner_product")
    axes[1].set_title("$\Rethroat = 2000$")
    axes[1].set_xticks(np.arange(0, 21, 1))
    axes[1].set_yticks([0, 0.6, 1.2])
    axes[1].yaxis.set_major_formatter(tick_formatter(1))
    im = axes[1].imshow(metric_values, vmin=0, vmax=1, extent=fda_extent_cm)

    # Colorbar
    fig.colorbar(im, cax=axes[2], orientation="horizontal", location="bottom").set_label("Index value")

    # Save and wrap up
    path = base_figure_dir / "Chapter5" / "inner_product_raw_comparison.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def vec_diff_raw_comparison(laad_type: LAADType):
    # General plot code
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 4.2), layout="constrained", height_ratios=[0.475, 0.475, 0.05])
    fig.suptitle("Local sum of squared differences for the FDA nozzle at $x = 0 \,\si{cm}$")

    # Top subplot
    ds = fda_dataset_list[0]
    metric_values = get_data(ds.prefix, laad_type, 5, index="vec_diff")
    axes[0].set_title("$\Rethroat = 500$")
    axes[0].set_xticks(np.arange(0, 21, 1), labels="")
    axes[0].set_yticks([0, 0.6, 1.2])
    axes[0].yaxis.set_major_formatter(tick_formatter(1))
    axes[0].imshow(metric_values, vmin=0, vmax=1, extent=fda_extent_cm)
    
    # Bottom subplot
    ds = fda_dataset_list[1]
    metric_values = get_data(ds.prefix, laad_type, 5, index="vec_diff")
    axes[1].set_xticks(np.arange(0, 21, 1))
    axes[1].set_yticks([0, 0.6, 1.2])
    axes[1].yaxis.set_major_formatter(tick_formatter(1))
    axes[1].set_title("$\Rethroat = 2000$")
    im = axes[1].imshow(metric_values, vmin=0, vmax=1, extent=fda_extent_cm)

    # Colorbar
    fig.colorbar(im, cax=axes[2], orientation="horizontal", location="bottom").set_label("Index value")

    # Save and wrap up
    path = base_figure_dir / "Chapter5" / "vec_diff_raw_comparison.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def laad_raw_comparison(laad_type: LAADType):
    # General plot code
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 4.2), layout="constrained", height_ratios=[0.475, 0.475, 0.05])
    fig.suptitle("Locally-averaged angular deviation for the FDA nozzle at $x = 0 \,\si{cm}$")

    # Top subplot
    ds = fda_dataset_list[0]
    metric_values = get_data(ds.prefix, laad_type, 5, index="laad")
    axes[0].set_title("$\Rethroat = 500$")
    axes[0].set_xticks(np.arange(0, 21, 1), labels="")
    axes[0].set_yticks([0, 0.6, 1.2])
    axes[0].yaxis.set_major_formatter(tick_formatter(1))
    axes[0].imshow(metric_values, vmin=0, vmax=1, extent=fda_extent_cm)
    
    # Bottom subplot
    ds = fda_dataset_list[1]
    metric_values = get_data(ds.prefix, laad_type, 5, index="laad")
    axes[1].set_title("$\Rethroat = 2000$")
    axes[1].set_xticks(np.arange(0, 21, 1))
    axes[1].set_yticks([0, 0.6, 1.2])
    axes[1].yaxis.set_major_formatter(tick_formatter(1))
    im = axes[1].imshow(metric_values, vmin=0, vmax=1, extent=fda_extent_cm)

    # Colorbar
    fig.colorbar(im, cax=axes[2], orientation="horizontal", location="bottom").set_label("Index value")

    # Save and wrap up
    path = base_figure_dir / "Chapter5" / "laad_raw_comparison.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def laad_centre_vs_average():
    # Figure path setup
    figure_dir = base_figure_dir / "Chapter5"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"laad_centre_vs_average.pdf"

    # Other parameters
    window_sizes = [1, 4, 7, 10]
    ds = fda_dataset_list[1]
    extent = [9, 20, 0, 1.2]

    # Figure setup
    # fig, axes = plt.subplots(2*len(window_sizes), 1, figsize=(fig_width, 14))
    fig = plt.figure(figsize=(fig_width, 12.2), layout="constrained")
    fig.suptitle(f"LAAD w.r.t. centre (top) versus average (bottom) vector for $\Rethroat = {ds.re:.0f}$")
    z_lb = 450  # Lower bound index for fda dataset
    z_ub = 900  # Upper bound index for fda dataset

    gs = GridSpec(nrows=4, ncols=2, figure=fig, hspace=0, width_ratios=[0.975, 0.025])
    subfigs = [fig.add_subfigure(gs[i, 0]) for i in range(4)]

    # subfigs = fig.subfigures(nrows=len(window_sizes), ncols=1, hspace=0)
    for subfig, ws in zip(subfigs, window_sizes):
        ws_si_cm = (2 * ws + 1) * fda_dx * 100
        subfig.suptitle(f"window size = {ws_si_cm:.2f}\,\si{{cm}}")
        (left_ax, right_ax) = subfig.subplots(nrows=2, ncols=1, sharex=True)
        centre_ds = get_data(
            ds.prefix,
            LAADType(Neighbourhood.BALL, RelativeTo.CENTRE),
            ws,
            index="laad"
        )
        average_ds = get_data(
            ds.prefix,
            LAADType(Neighbourhood.BALL, RelativeTo.AVERAGE),
            ws,
            index="laad"
        )
        
        left_ax.set_xticks(np.arange(9, 21, 1))
        left_ax.set_yticks([0, 0.6, 1.2])
        left_ax.yaxis.set_major_formatter(tick_formatter(1))
        left_ax.imshow(centre_ds[:, z_lb:z_ub], vmin=0, vmax=1, extent=extent)

        right_ax.set_yticks([0, 0.6, 1.2])
        right_ax.yaxis.set_major_formatter(tick_formatter(1))
        im = right_ax.imshow(average_ds[:, z_lb:z_ub], vmin=0, vmax=1, extent=extent)

    ax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(im, cax=ax, orientation="vertical")
    cb.set_ticks(np.arange(0, 1.1, 0.1))
    cb.set_label("Index value")
        
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)



def sphere_vs_cube():
        # Figure path setup
    figure_dir = base_figure_dir / "Chapter5"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"sphere_vs_cube.pdf"

    # Other parameters
    window_sizes = [1, 5, 10]
    ds = fda_dataset_list[1]
    extent = [9, 20, 0, 1.2]

    # Figure setup
    # fig, axes = plt.subplots(2*len(window_sizes), 1, figsize=(fig_width, 14))
    fig = plt.figure(figsize=(fig_width, 9), layout="constrained")
    fig.suptitle(f"LAAD w.r.t. spherical (top) versus cubic (middle) neighbourhood\n and the absolute difference (bottom) for $\Rethroat = {ds.re:.0f}$")
    z_lb = 450  # Lower bound index for fda dataset
    z_ub = 900  # Upper bound index for fda dataset

    gs = GridSpec(nrows=len(window_sizes), ncols=2, figure=fig, hspace=0, width_ratios=[0.975, 0.025])
    subfigs = [fig.add_subfigure(gs[i, 0]) for i in range(len(window_sizes))]

    for subfig, ws in zip(subfigs, window_sizes):
        ws_si_cm = (2 * ws + 1) * fda_dx * 100
        subfig.suptitle(f"window size = {ws_si_cm:.2f}\,\si{{cm}}")
        (ax1, ax2) = subfig.subplots(nrows=2, ncols=1, sharex=True)
        sphere_ds = get_data(
            ds.prefix,
            LAADType(Neighbourhood.BALL, RelativeTo.AVERAGE),
            ws,
            index="laad"
        )
        cube_ds = get_data(
            ds.prefix,
            LAADType(Neighbourhood.CUBE, RelativeTo.AVERAGE),
            ws,
            index="laad"
        )
        
        ax1.set_xticks(np.arange(9, 21, 1))
        ax1.set_yticks([0, 0.6, 1.2])
        ax1.yaxis.set_major_formatter(tick_formatter(1))
        ax1.imshow(sphere_ds[:, z_lb:z_ub], vmin=0, vmax=1, extent=extent)

        ax2.set_yticks([0, 0.6, 1.2])
        ax2.yaxis.set_major_formatter(tick_formatter(1))
        im2 = ax2.imshow(cube_ds[:, z_lb:z_ub], vmin=0, vmax=1, extent=extent)

        # ax3.set_yticks([0, 0.6, 1.2])
        # ax3.yaxis.set_major_formatter(tick_formatter(1))
        # im3 = ax3.imshow(np.abs((cube_ds[:, z_lb:z_ub] - sphere_ds[:, z_lb:z_ub])) > 0.15, vmin=0, vmax=1, extent=extent)
        data = np.abs(cube_ds[:, :] - sphere_ds[:, :])
        print("Window size: ", ws)
        print("Max: ", np.nanmax(data))
        print("Percentile 99: ", np.nanpercentile(data, 99))
        print("Percentile 90: ", np.nanpercentile(data, 80))
        print(f" {np.nanpercentile(data, 95):.2f} & {np.nanpercentile(data, 99):.2f} & {np.nanmax(data):.2f}\\\\")
        print("---")
    
    cb_ax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(im2, cax=cb_ax, orientation="vertical")
    cb.set_label("Index value")
    cb.set_ticks(np.arange(0, 1.1, 0.1))
        
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_window_size_graphs(laad_type: LAADType, datasets: list[Dataset]):
    # Define the storage location
    figure_dir = base_figure_dir / "Chapter5"
    figure_dir.mkdir(parents=True, exist_ok=True)

    downstream_points = [500, 600, 700, 800]

    for dataset in datasets:
        # Define the figure path
        figure_path = figure_dir / f"{dataset.prefix}_{laad_type_str(laad_type)}_graphs.pdf"

        # Get the dataset
        datasets = get_window_size_dataset(dataset.prefix, laad_type)
        si_cube_side_lengths_cm = (2 * window_sizes + 1) * dataset.dx * 100

        # Compare the behaviour as the window size grows for various points
        fig, axes = plt.subplots(3, 2, figsize=(fig_width, 7.5), layout="constrained", height_ratios=[1, 1, 0.2], sharey=False)
        shape_string = "spherical" if laad_type.neighbourhood_shape == Neighbourhood.BALL else "cubic"
        ref_vec_string = "average" if laad_type.relative_to == RelativeTo.AVERAGE else "centre"
        fig.suptitle(f"Growing window size behaviour for the FDA nozzle for $x=\SI{{0}}{{\cm}}$, $\Rethroat = {dataset.re:.0f}$,\n{shape_string} neighbourhood, and relative to the {ref_vec_string} vector")
        handles = []
        axes[2][0].axis('off')
        axes[2][1].axis('off')
        for ax, z, tag in zip(axes.flatten(), downstream_points, string.ascii_uppercase):
            handles = []
            z_si_cm = z * fda_dx * 100
            ax.set_title(f"$z={z_si_cm:.0f}\,\si{{cm}}$")
            ax.set_xlabel("Window size (cm)")
            ax.set_xlim([0, 0.7])
            ax.set_xticks(np.arange(0, 0.8, 0.1))
            ax.xaxis.set_major_formatter(tick_formatter(1))
            ax.yaxis.set_major_formatter(tick_formatter(1))
            ax.set_ylim([0, 1])
            ax.grid(which="both")

            ax.annotate(tag, (0.13, 0.83), fontweight="bold", fontsize=30)
            ax.axhline(y=0.2, linestyle="dotted", color="black")
            ax.axvline(x=0.06, linestyle="dotted", color="black")
            ax.axvline(x=0.38, linestyle="dotted", color="black")

            for y in np.arange(10, 60, 10):
                values = datasets[:, y, z]
                y_si_cm = y * fda_dx * 100
                handles.extend(ax.plot(si_cube_side_lengths_cm, values, label=f"$y={y_si_cm}\,\si{{cm}}$"))
        fig.legend(handles=handles, ncols=5, bbox_to_anchor=(0.5, 0.03), loc='center')
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)


def window_size_classifications(laad_type: LAADType, datasets: list[Dataset]):
    # Define the window sizes used for comparison
    ref_1_index = 0
    ref_2_index = 8
    thresholds = [0.05, 0.1, 0.2, 0.3]
    
    # Define the storage location
    figure_dir = base_figure_dir / "Chapter5"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        # Define the figure path
        ref_1_size_cm = (2*window_sizes[ref_1_index]+1) * dataset.dx * 100
        ref_2_size_cm = (2*window_sizes[ref_2_index]+1) * dataset.dx * 100

        # Define the figure path
        figure_path = figure_dir / f"{dataset.prefix}_{laad_type_str(laad_type)}_classified.pdf"
        # Get the datasets
        datasets = get_window_size_dataset(dataset.prefix, laad_type)
        
        # Get the values for the two window_size we'll compare
        ref_1 = datasets[ref_1_index, :, :]
        ref_2 = datasets[ref_2_index, :, :]
        
        # Define the plot
        legend_elements = [
            Rectangle((0,0), 1, 1, edgecolor='black', facecolor=(1, 0, 0), label=f"Low at \SI{{{ref_1_size_cm:.2f}}}{{\cm}}, low at \SI{{{ref_2_size_cm:.2f}}}{{\cm}}"),
            Rectangle((0,0), 1, 1, edgecolor='black', facecolor=(0, 1, 0), label=f"Low at \SI{{{ref_1_size_cm:.2f}}}{{\cm}}, high at \SI{{{ref_2_size_cm:.2f}}}{{\cm}}"),
            Rectangle((0,0), 1, 1, edgecolor='black', facecolor=(1, 1, 0), label=f"High at \SI{{{ref_1_size_cm:.2f}}}{{\cm}}, low at \SI{{{ref_2_size_cm:.2f}}}{{\cm}}"),
            Rectangle((0,0), 1, 1, edgecolor='black', facecolor=(0, 0, 1), label=f"High at \SI{{{ref_1_size_cm:.2f}}}{{\cm}}, high at \SI{{{ref_2_size_cm:.2f}}}{{\cm}}"),
            # Rectangle((0,0), 1, 1, edgecolor='black', facecolor=(1, 1, 1), label="Outside domain"),
        ]

        fig, axes = plt.subplots(len(thresholds) + 1, 1, figsize=(fig_width, 6.5), layout="constrained", height_ratios=[1, 1, 1, 1, 0.75])
        axes[-1].axis("off")
        shape_string = "spherical" if laad_type.neighbourhood_shape == Neighbourhood.BALL else "cubic"
        ref_vec_string = "average" if laad_type.relative_to == RelativeTo.AVERAGE else "centre"
        fig.suptitle(f"Points classified by LAAD index behaviour w.r.t. window size\nfor $x=\SI{{0}}{{\cm}}$, $\Rethroat={dataset.re:.0f}$, {shape_string} neighbourhood, relative to the {ref_vec_string} vector")

        for ax, threshold in zip(axes, thresholds):
            if ax != axes[3]:
                ax.set_xticks(np.arange(0, 21, 1), labels="")
            else:
                ax.set_xticks(np.arange(0, 21, 1))
                ax.xaxis.set_major_formatter(tick_formatter(0))
            classified_points = classify_points(threshold, ref_1, ref_2)
            ax.set_yticks([0, 0.6, 1.2])
            ax.yaxis.set_major_formatter(tick_formatter(1))
            ax.imshow(classified_points, extent=fda_extent_cm)
            ax.set_title(f"Threshold = {threshold}")

        fig.legend(handles=legend_elements, loc='center', ncols=2, bbox_to_anchor=(0.5, 0.075))
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)


def plot_window_size_classifications_karman(laad_type: LAADType):
    # Define the window sizes used for comparison
    ref_1_index = 0
    ref_2_index = 7
    thresholds = [0.05, 0.1, 0.2, 0.3,]

    prefix = "laad_karman_re1000"
    
    dx_cm = 1

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)

    ref_1_size = (2*window_sizes[ref_1_index]+1) * dx_cm
    ref_2_size = (2*window_sizes[ref_2_index]+1) * dx_cm

    # Define the figure path
    figure_path = base_figure_dir / "Chapter5" / f"{prefix}_classified.pdf"
    # Get the datasets
    datasets = get_window_size_dataset(prefix, laad_type, dataset="karman")
    # Get the values for the two window_size we'll compare
    ref_1 = datasets[ref_1_index, :, :]
    ref_2 = datasets[ref_2_index, :, :]
    # Define the plot
    legend_elements = [
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(1, 0, 0), label=f"Low at \SI{{{ref_1_size}}}{{\cm}}, low at \SI{{{ref_2_size}}}{{\cm}}"),
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(0, 1, 0), label=f"Low at \SI{{{ref_1_size}}}{{\cm}}, high at \SI{{{ref_2_size}}}{{\cm}}"),
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(1, 1, 0), label=f"High at \SI{{{ref_1_size}}}{{\cm}}, low at \SI{{{ref_2_size}}}{{\cm}}"),
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(0, 0, 1), label=f"High at \SI{{{ref_1_size}}}{{\cm}}, high at \SI{{{ref_2_size}}}{{\cm}}"),
        # Rectangle((0,0), 1, 1, color=(0, 0, 0), label="Outside domain"),
    ]
    fig, axes = plt.subplots(3,2, figsize=(fig_width, 5.5), layout="constrained", height_ratios=[1, 1, 0.4])
    shape_string = "spherical" if laad_type.neighbourhood_shape == Neighbourhood.BALL else "cubic"
    ref_vec_string = "average" if laad_type.relative_to == RelativeTo.AVERAGE else "centre"
    fig.suptitle(f"Points classified by LAAD index behaviour w.r.t. window size\nfor $z=\SI{{5}}{{\cm}}$, for a {shape_string} neighbourhood, relative to the {ref_vec_string} vector")

    for ax, threshold in zip(axes.flatten(), thresholds):
        # Compute plot with classifications
        classified_points = classify_points(threshold, ref_1, ref_2)
        ax.imshow(np.swapaxes(classified_points, 0, 1), extent=[0, classified_points.shape[0] * dx_cm, 0, classified_points.shape[1] * dx_cm])
        ax.set_xticks(np.arange(0, 201, 25))
        ax.set_yticks(np.arange(0, 51, 10))
        # ax.xaxis.set_major_formatter(tick_formatter(1))
        # ax.yaxis.set_major_formatter(tick_formatter(1))
        ax.set_title(f"Threshold={threshold}")
    axes[-1, 0].axis("off")
    axes[-1, 1].axis("off")
    fig.legend(handles=legend_elements, ncols=2, loc='center', bbox_to_anchor=(0.5, 0.075))
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_window_size_classifications_jhtdb(laad_type: LAADType):
    # Define the window sizes used for comparison
    ref_1_index = 0
    ref_2_index = 32
    thresholds = [0.05, 0.1, 0.15, 0.2,]

    prefix = "laad_jhtdb201"
    dx = 0.00613592315

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)

    ref_1_size = (2*window_sizes[ref_1_index]+1) * dx
    ref_2_size = (2*window_sizes[ref_2_index]+1) * dx

    # Define the figure path
    figure_path = base_figure_dir / "Chapter5" / f"{prefix}_classified.pdf"
    # Get the datasets
    datasets = get_window_size_dataset(prefix, laad_type, dataset="jhtdb")
    # Get the values for the two window_size we'll compare
    ref_1 = datasets[ref_1_index, :, :]
    ref_2 = datasets[ref_2_index, :, :]
    # Define the plot
    legend_elements = [
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(1, 0, 0), label=f"Low at \SI{{{ref_1_size:.4f}}}{{\m}}, low at \SI{{{ref_2_size:.4f}}}{{\m}}"),
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(0, 1, 0), label=f"Low at \SI{{{ref_1_size:.4f}}}{{\m}}, high at \SI{{{ref_2_size:.4f}}}{{\m}}"),
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(1, 1, 0), label=f"High at \SI{{{ref_1_size:.4f}}}{{\m}}, low at \SI{{{ref_2_size:.4f}}}{{\m}}"),
        Rectangle((0,0), 1, 1, edgecolor="black", facecolor=(0, 0, 1), label=f"High at \SI{{{ref_1_size:.4f}}}{{\m}}, high at \SI{{{ref_2_size:.4f}}}{{\m}}"),
        # Rectangle((0,0), 1, 1, color=(0, 0, 0), label="Outside domain"),
    ]
    fig, axes = plt.subplots(2,4, figsize=(fig_width, 4.5), layout="constrained", height_ratios=[1, 0.2])
    shape_string = "spherical" if laad_type.neighbourhood_shape == Neighbourhood.BALL else "cubic"
    ref_vec_string = "average" if laad_type.relative_to == RelativeTo.AVERAGE else "centre"
    fig.suptitle(f"Points classified by LAAD index behaviour w.r.t. window size\nfor $z=\pi\,\si{{m}}$, for a {shape_string} neighbourhood, relative to the {ref_vec_string} vector")

    for ax, threshold in zip(axes.flatten(), thresholds):
        # Compute plot with classifications
        classified_points = classify_points(threshold, ref_1, ref_2)
        ax.imshow(classified_points, extent=[0, classified_points.shape[0] * dx, 0, classified_points.shape[1] * dx])
        ax.set_xticks(np.arange(0, 1.3, 0.2))
        ax.set_yticks(np.arange(0, 1.3, 0.2))
        ax.xaxis.set_major_formatter(tick_formatter(1))
        ax.yaxis.set_major_formatter(tick_formatter(1))
        ax.set_title(f"Threshold={threshold}")
    axes[-1, 0].axis("off")
    axes[-1, 1].axis("off")
    axes[-1, 2].axis("off")
    axes[-1, 3].axis("off")
    fig.legend(handles=legend_elements, ncols=2, loc='center', bbox_to_anchor=(0.5, 0.075))
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    # Make sure the figure directory exists
    base_figure_dir.exists()
    # Run all the plots
    laad_raw_karman(laad_type=LAADType(relative_to=RelativeTo.AVERAGE,neighbourhood_shape=Neighbourhood.BALL))
    inner_product_raw_comparison(laad_type=LAADType(relative_to=RelativeTo.AVERAGE, neighbourhood_shape=Neighbourhood.BALL))
    vec_diff_raw_comparison(laad_type=LAADType(relative_to=RelativeTo.AVERAGE, neighbourhood_shape=Neighbourhood.BALL))
    laad_raw_comparison(laad_type=LAADType(relative_to=RelativeTo.AVERAGE, neighbourhood_shape=Neighbourhood.BALL))

    plot_window_size_graphs(
        laad_type=LAADType(relative_to=RelativeTo.CENTRE,neighbourhood_shape=Neighbourhood.BALL),
        datasets=[fda_dataset_list[1]]
    )
    plot_window_size_graphs(
        laad_type=LAADType(relative_to=RelativeTo.AVERAGE,neighbourhood_shape=Neighbourhood.BALL),
        datasets=[fda_dataset_list[1]]
    )

    laad_centre_vs_average()
    sphere_vs_cube()

    window_size_classifications(
        laad_type=(LAADType(relative_to=RelativeTo.AVERAGE, neighbourhood_shape=Neighbourhood.BALL)),
        datasets=fda_dataset_list[:2]
    )
    window_size_classifications(
        laad_type=(LAADType(relative_to=RelativeTo.CENTRE, neighbourhood_shape=Neighbourhood.BALL)),
        datasets=[fda_dataset_list[1]]
    )

    plot_window_size_classifications_jhtdb(laad_type=LAADType(relative_to=RelativeTo.AVERAGE, neighbourhood_shape=Neighbourhood.BALL))
    plot_window_size_classifications_karman(laad_type=LAADType(relative_to=RelativeTo.AVERAGE, neighbourhood_shape=Neighbourhood.BALL))
