#! /usr/bin/env python3
import argparse
from dataclasses import dataclass
from matplotlib.patches import Rectangle
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


base_figure_dir = Path(__file__).parent / "figures" / "fda" / "laad"

dataset_list = [
    Dataset("laad_fda_re500", (60, 60, 1000), 30, 0.0002),
    Dataset("laad_fda_re2000", (60, 60, 1000), 30, 0.0002),
    Dataset("laad_fda_re3500", (60, 60, 1000), 30, 0.0002),
    Dataset("laad_fda_re5000", (60, 60, 1000), 30, 0.0002),
    Dataset("laad_fda_re6500", (60, 60, 1000), 30, 0.0002),
]

window_sizes = np.arange(1, 41)


def laad_type_dir(base_dir: Path, laad_type: LAADType) -> Path:
    output_path = base_dir

    components = [str(laad_type.neighbourhood_shape), str(laad_type.relative_to)]
    dirname = "_".join(components)

    return output_path / dirname


def get_data(file_prefix: str, laad_type: LAADType, window_size: int) -> np.ndarray:
    base_dir = Path(__file__).parent / "output" / "fda" / "laad"
    laad_dir = laad_type_dir(base_dir, laad_type)
    file_path = laad_dir / f"{file_prefix}_{window_size}.npy"
    if not file_path.exists():
        raise ValueError("File does not exist:", file_path.resolve())
    return np.load(file_path)


def plot_all_window_sizes(laad_type: LAADType):
    # Define the storage location
    figure_dir = laad_type_dir(base_figure_dir / "laad_all_images", laad_type)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for dataset in dataset_list:
        for window_size in window_sizes:
            # Create destination path
            figure_path = figure_dir / f"{dataset.prefix}_{window_size}.png"        

            # Retrieve the data from disk
            metric_values = get_data(dataset.prefix, laad_type, window_size)
            # Plot the result
            fig, axis = plt.subplots(1, 1)
            fig.tight_layout()
            axis.set_title("LAAD of FDA nozzle at $x = 0$ m")
            im = axis.imshow(metric_values, vmin=0)
            fig.colorbar(im, ax=axis, orientation="horizontal", location="bottom")
            fig.savefig(figure_path, dpi=300)
            plt.close(fig)



def get_window_size_dataset(file_prefix: str, laad_type: LAADType) -> np.ndarray:
    dataset_list = []

    for window_size in window_sizes:
        dataset_list.append(get_data(file_prefix, laad_type, window_size))

    return np.stack(dataset_list, axis=0)  # Has shape (40, 60, 1000)


def plot_window_size_graphs(laad_type: LAADType):
    # Define the storage location
    figure_dir = laad_type_dir(base_figure_dir / "window_size_analysis", laad_type)
    figure_dir.mkdir(parents=True, exist_ok=True)

    downstream_points = [500, 600, 700, 800]


    for dataset in dataset_list:
        # Define the figure path
        figure_path = figure_dir / f"{dataset.prefix}_graphs.png"

        # Get the dataset
        datasets = get_window_size_dataset(dataset.prefix, laad_type)
        si_cube_side_lengths = (2 * window_sizes + 1) * dataset.dx

        # Compare the behaviour as the window size grows for various points
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.set_tight_layout(tight=True)

        for ax, z in zip(axes.flatten(), downstream_points):
            ax.set_title(f"{z=}")
            ax.set_xlabel("Window cube side length (m)")
            ax.set_xlim([0, 0.016])
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.set_ylim([0, 1])
            ax.grid(which="both")

            for y in np.arange(10, 60, 10):
                values = datasets[:, y, z]
                ax.plot(si_cube_side_lengths, values, label=y)
            ax.legend(loc="upper left")
        fig.savefig(figure_path)
        plt.close(fig)


def classify_points(threshold: float, ref_1: np.ndarray, ref_2: np.ndarray) -> np.ndarray:
    result = np.zeros((60, 1000, 3), dtype=np.int32)
    result[np.logical_and(ref_1 < threshold, ref_2 < threshold), 0] = 255  # red, indicating low value, flat
    result[np.logical_and(ref_1 < threshold, ref_2 > threshold), 1] = 255  # green, indicating rising value
    result[np.logical_and(ref_1 > threshold, ref_2 > threshold), 2] = 255  # blue, indicating high value, flat
    result[np.logical_and(ref_1 > threshold, ref_2 < threshold), 0] = 255  # yellow, indicating high to low
    result[np.logical_and(ref_1 > threshold, ref_2 < threshold), 1] = 255  # yellow, indicating high to low
    return result


def plot_window_size_classifications(laad_type: LAADType):
    figure_dir = laad_type_dir(base_figure_dir / "window_size_analysis", laad_type)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Define the window sizes used for comparison
    ref_1_index = 2
    ref_2_index = 7
    thresholds = [0.1, 0.2, 0.35, 0.4, 0.5,]

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)

    for dataset in dataset_list:
        ref_1_size = (2*window_sizes[ref_1_index]+1) * dataset.dx
        ref_2_size = (2*window_sizes[ref_2_index]+1) * dataset.dx

        # Define the figure path
        figure_path = figure_dir / f"{dataset.prefix}_classified.png"
        # Get the datasets
        datasets = get_window_size_dataset(dataset.prefix, laad_type)
        # Get the values for the two window_size we'll compare
        ref_1 = datasets[ref_1_index, :, :]
        ref_2 = datasets[ref_2_index, :, :]
        # Define the plot
        legend_elements = [
            Rectangle((0,0), 1, 1, color=(1, 0, 0), label=f"Low at {ref_1_size:.6f}, low at {ref_2_size:.6f}"),
            Rectangle((0,0), 1, 1, color=(0, 1, 0), label=f"Low at {ref_1_size:.6f}, high at {ref_2_size:.6f}"),
            Rectangle((0,0), 1, 1, color=(0, 0, 1), label=f"High at {ref_1_size:.6f}, high at {ref_2_size:.6f}"),
            Rectangle((0,0), 1, 1, color=(1, 1, 0), label=f"High at {ref_1_size:.6f}, low at {ref_2_size:.6f}"),
            Rectangle((0,0), 1, 1, color=(0, 0, 0), label="Outside domain"),
        ]
        fig, axes = plt.subplots(len(thresholds), 1, figsize=(15,18))

        legends = []
        for ax, threshold in zip(axes, thresholds):
            # Compute plot with classifications
            classified_points = classify_points(threshold, ref_1, ref_2)
            ax.imshow(classified_points, extent=[0, classified_points.shape[1] * dataset.dx, 0, classified_points.shape[0] * dataset.dx])
            lgd = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.25))
            legends.append(lgd)
            ax.set_title(f"Growing window size behaviour, threshold={threshold}", fontsize=24)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=300, bbox_extra_artists=legends, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    all_plot_choices = ["all", "metric_raw", "ws_graph", "ws_classification"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plot_types",
        type=str,
        default="all",
        choices=all_plot_choices,
        nargs="*",
        help="Choose which plots to generate. If not specified, all plots are generated."
    )
    parser.add_argument(
        "--relative_to",
        type=str,
        default="both",
        choices=["average", "centre", "both"],
        help="use LAAD data relative to the average vector in a window, or relative to the center window"
    )
    parser.add_argument("--shape", type=str, default="both", choices=["cube", "ball", "both"])
    args = parser.parse_args()
    
    chosen_plots = all_plot_choices if "all" in args.plot_types else args.plot_types

    # Parse the arguments to internal objects
    relative_params = list(RelativeTo) if args.relative_to == "both" else [RelativeTo(args.relative_to)]
    shape_params = list(Neighbourhood) if args.relative_to == "both" else [Neighbourhood(args.shape)]

    # Create a list of all configurations to run with
    laad_types = []
    for rel in relative_params:
        for shape in shape_params:
            laad_types.append(LAADType(neighbourhood_shape=shape, relative_to=rel))

    if "metric_raw" in chosen_plots:
        for laad_type in laad_types:
            print(f"Plotting visualization for LAAD for laad_type {laad_type}")
            plot_all_window_sizes(laad_type)

    if "ws_graph" in chosen_plots:
        for laad_type in laad_types:
            print(f"Plotting graphs showing the behaviour of LAAD for varying window size with laad_type {laad_type}")
            plot_window_size_graphs(laad_type)

    if "ws_classification" in chosen_plots:
        for laad_type in laad_types:
            print(f"Plot images where each point is classified by the behaviour of LAAD for varying window size for laad_type {laad_type}")
            plot_window_size_classifications(laad_type)
