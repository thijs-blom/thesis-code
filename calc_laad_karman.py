#! /usr/bin/env python3
import argparse
import meshio
import numpy as np
from pathlib import Path
from turbulence_statistics import Axis, index_inner_product, index_vec_diff, laad, LAADType, RelativeTo, Neighbourhood
from datetime import datetime


n_proc = 1

data_path = Path(__file__).parent / "data" / "karman"
base_output_dir = Path(__file__).parent / "output" / "karman"

parameter_list = [
    # (input_file, output_prefix shape, z_slice)
    ("re1000_200_50_10.vtu", "laad_karman_re1000", (200, 50, 10), 5)
]

window_sizes = np.arange(1, 41)


def get_fields(data_file: Path, nx: int, ny: int, nz: int):
    mesh = meshio.read(data_file)

    u = mesh.point_data["u"].reshape((nx, ny, nz), order="F")
    v = mesh.point_data["v"].reshape((nx, ny, nz), order="F")
    w = mesh.point_data["w"].reshape((nx, ny, nz), order="F")

    return u, v, w


def compute_index(laad_type: LAADType, output_dir: Path, index_func):
    for params in parameter_list:
        print(f"Processing {params[0]}")
        print("---------------------------")
        u, v, w = get_fields(data_path / params[0], *params[2])
        for window_size in window_sizes:
            start = datetime.now()
            result = index_func(
                u,
                v,
                w,
                window_size,
                laad_type,
                n_proc=n_proc,
                slice_axis=Axis.Z,
                slice_value=params[3],
            )
            end = datetime.now()
            elapsed = (end - start).total_seconds()
            print(f"Computed results for {window_size=} in {elapsed} seconds")
            np.save(output_dir / f"{params[1]}_{window_size}.npy", result)


def laad_type_dir(base_dir: Path, laad_type: LAADType) -> Path:
    output_path = base_dir

    components = [str(laad_type.neighbourhood_shape), str(laad_type.relative_to)]
    dirname = "_".join(components)

    return output_path / dirname


if __name__ == "__main__":
    all_indices = ["laad", "vec_diff", "inner_product"]
    index_funcs = {
        "laad": laad,
        "vec_diff": index_vec_diff,
        "inner_product": index_inner_product
    }
    possible_index_choices = ["laad", "vec_diff", "inner_product", "all"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--relative_to", type=str, default="both", choices=["average", "centre", "both"])
    parser.add_argument("--shape", type=str, default="both", choices=["cube", "ball", "both"])
    parser.add_argument("--index", type=str, default="all", choices=possible_index_choices)
    args = parser.parse_args()

    chosen_indices = []
    if args.index == "all":
        chosen_indices = all_indices
    else:
        chosen_indices.append(args.index)
    # Make sure paths exist
    if not data_path.exists():
        raise ValueError("Data directory does not exist")
    # Make sure input files exist
    for params in parameter_list:
        if not (data_path / params[0]).exists():
            raise ValueError(f"Input file {params[0]} does not exist")

    # Make sure output path exists
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Parse the arguments to internal objects
    relative_params = list(RelativeTo) if args.relative_to == "both" else [RelativeTo(args.relative_to)]
    shape_params = list(Neighbourhood) if args.shape == "both" else [Neighbourhood(args.shape)]

    # Create a list of all configurations to run with
    laad_types = []
    for rel in relative_params:
        for shape in shape_params:
            laad_types.append(LAADType(neighbourhood_shape=shape, relative_to=rel))

    for laad_type in laad_types:
        for index in chosen_indices:
            output_dir = laad_type_dir(base_output_dir / index, laad_type)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Computing index {index} for configuration: {laad_type}")
            func = index_funcs[index]
            compute_index(laad_type, output_dir, func)
