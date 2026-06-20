"""
comet_noise_pipeline.py
================================================================================
Run a three-stage LAS pipeline:

1. Run comet-tail filtering on the input LAS.
2. Run scene-context noise scoring on the comet-filtered LAS.
"""

import argparse
import os

import numpy as np

import comet
import scene_context_noise


DEFAULT_INPUT_FILE = r"C:\Users\njzy1\Desktop\data\input_file\test\test1.las"
DEFAULT_FIRST_COMET_OUTPUT_FILE = r"C:\Users\njzy1\Desktop\data\result_first_comet_clean.las"
DEFAULT_OUTPUT_FILE = r"C:\Users\njzy1\Desktop\data\result_pipeline.las"

DEFAULT_COMET_PARAMS = {
    "angular_thresh": 0.003,
    "min_radial_thresh": 0.02,
    "radial_thresh": 0.25,
    "max_neighbors": 100,
    "min_neighbors": 20,
    "batch_size": 50_000,
}

def run_pipeline(
    input_path,
    first_comet_output_path,
    output_path,
    noise_threshold=0.5,
    default_comet_params=None,
):
    default_comet_params = default_comet_params or DEFAULT_COMET_PARAMS

    print(f"Reading {input_path} ...")
    las = scene_context_noise._read_las_safe(input_path)
    xyz = np.ascontiguousarray(las.xyz, dtype=np.float32)
    print(f"{len(xyz):,} points loaded\n")

    print("[Stage 1/2] Running comet filtering...")
    comet_default = comet.classify_comet_points(xyz, **default_comet_params)
    comet.save_filtered_las(las, comet_default, first_comet_output_path)

    print(f"\n[Stage 2/2] Running scene-context noise scoring on {first_comet_output_path}...")
    scene_context_noise.run(
        first_comet_output_path,
        output_path,
        threshold=noise_threshold,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comet filtering, then scene-context noise scoring."
    )
    parser.add_argument("input_file", nargs="?", default=DEFAULT_INPUT_FILE, help="Input .las/.laz file")
    parser.add_argument("output_file", nargs="?", default=DEFAULT_OUTPUT_FILE, help="Output .las/.laz file")
    parser.add_argument("--first-comet-output", default=DEFAULT_FIRST_COMET_OUTPUT_FILE)
    parser.add_argument("--noise-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found")
        raise SystemExit(1)

    run_pipeline(
        args.input_file,
        args.first_comet_output,
        args.output_file,
        noise_threshold=args.noise_threshold,
    )
