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


base_figure_dir = Path(__file__).parent / "figures" / "jhtdb" / "laad"

dataset_list = [
    Dataset("laad_jhtdb201", (201, 201, 201), 100, 0.00613592315),
]

window_sizes = np.arange(1, 41)


def laad_type_dir(base_dir: Path, laad_type: LAADType) -> Path:
    output_path = base_dir

    components = [str(laad_type.neighbourhood_shape), str(laad_type.relative_to)]
    dirname = "_".join(components)

    return output_path / dirname


def get_data(file_prefix: str, laad_type: LAADType, window_size: int) -> np.ndarray:
    base_dir = Path(__file__).parent / "output" / "jhtdb" / "laad"
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
            axis.set_title("LAAD of JHTDB dataset at $z = \pi$ m")
            im = axis.imshow(metric_values, vmin=0)
            fig.colorbar(im, ax=axis, orientation="horizontal", location="bottom")
            fig.savefig(figure_path, dpi=300)
            plt.close(fig)



def get_window_size_dataset(file_prefix: str, laad_type: LAADType) -> np.ndarray:
    dataset_list = []

    for window_size in window_sizes:
        dataset_list.append(get_data(file_prefix, laad_type, window_size))

    return np.stack(dataset_list, axis=0)  # Has shape (40, 256, 256)


def plot_window_size_graphs(laad_type: LAADType):
    # Define the storage location
    figure_dir = laad_type_dir(base_figure_dir / "window_size_analysis", laad_type)
    figure_dir.mkdir(parents=True, exist_ok=True)

    points = [25, 50, 100 , 150]


    for dataset in dataset_list:
        # Define the figure path
        figure_path = figure_dir / f"{dataset.prefix}_graphs.png"

        # Get the dataset
        datasets = get_window_size_dataset(dataset.prefix, laad_type)
        si_cube_side_lengths = (2 * window_sizes + 1) * dataset.dx

        # Compare the behaviour as the window size grows for various points
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Index behaviour for varying window size ({laad_type.neighbourhood_shape},{laad_type.relative_to})", fontsize=24)

        for ax, x in zip(axes.flatten(), points):
            ax.set_title(f"x={x*dataset.dx:.6f}", fontsize=18)
            ax.set_xlabel("Window cube side length (m)", fontsize=16)
            # ax.set_xlim([0, 0.016])
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            # ax.set_ylim([0, 1])
            ax.grid(which="both")

            for y in points:
                values = datasets[:, x, y]
                ax.plot(si_cube_side_lengths, values, label=np.around(y*dataset.dx, decimals=6))
            ax.legend(loc="upper left", fontsize=16)
        fig.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)


def classify_points(threshold: float, ref_1: np.ndarray, ref_2: np.ndarray) -> np.ndarray:
    result = np.zeros((201, 201, 3), dtype=np.int32)
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
    ref_1_index = 0
    ref_2_index = 39
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25,]


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
            # Rectangle((0,0), 1, 1, color=(0, 0, 0), label="Outside domain"),
        ]
        fig, axes = plt.subplots(3,2, figsize=(15,15))
        fig.suptitle(f"Points classified by behaviour when varying window size ({laad_type.neighbourhood_shape},{laad_type.relative_to})", fontsize=24)
        points = np.array([25, 50, 100 , 150]) * dataset.dx
        xx, yy = np.meshgrid(points, points)

        for ax, threshold in zip(axes.flatten(), thresholds):
            # Compute plot with classifications
            classified_points = classify_points(threshold, ref_1, ref_2)
            ax.imshow(classified_points, extent=[0, classified_points.shape[0] * dataset.dx, 0, classified_points.shape[1] * dataset.dx])
            ax.set_xticks(np.arange(0, 6.1, 1))
            ax.set_yticks(np.arange(0, 6.1, 1))
            ax.set_title(f"Threshold={threshold}", fontsize=24)
            ax.scatter(xx, yy, marker='x', color='black')
        axes[2, 1].axis("off")
        lgd = fig.legend(handles=legend_elements, loc='lower right', fontsize=18, bbox_to_anchor=(0.9, 0.1))
        fig.tight_layout()
        fig.savefig(figure_path, bbox_extra_artists=(lgd,), dpi=300)
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
