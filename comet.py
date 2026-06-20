"""
comet_classifier_angular.py
================================================================================
Classify LiDAR points using Ray-Casting / Angular matching (fixed parameters).
"""
import os
import sys
import gc
import argparse
import numpy as np
import laspy
from scipy.spatial import cKDTree

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'

NORMAL = np.int8(0)
TAIL   = np.int8(1)


def _find_ray_candidates(xyz, angular_thresh, min_radial_thresh, max_radial_thresh,
                         max_neighbors, min_neighbors, batch_size):
    n = len(xyz)

    print("   Converting XYZ to unit sphere...")
    r = np.linalg.norm(xyz, axis=1)
    r_safe = np.where(r == 0, 1e-9, r)
    unit_xyz = xyz / r_safe[:, np.newaxis]

    chord_thresh = 2 * np.sin(angular_thresh / 2)

    print("   Building 3D unit-sphere cKD-tree...")
    tree = cKDTree(unit_xyz, leafsize=100)

    is_comet = np.zeros(n, dtype=bool)
    print(f"   Scanning rays in batches of {batch_size:,}...")

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        if batch_start % 1_000_000 == 0 and batch_start > 0:
            print(f"    {batch_start:,} / {n:,}  ({100*batch_start/n:.0f}%)")
            gc.collect()

        batch_unit_xyz = unit_xyz[batch_start:batch_end]
        batch_r        = r[batch_start:batch_end]

        _, nbr_indices = tree.query(
            batch_unit_xyz, k=max_neighbors,
            distance_upper_bound=chord_thresh, workers=-1
        )

        for local_i, nbrs in enumerate(nbr_indices):
            valid_nbrs = nbrs[nbrs < n]

            if len(valid_nbrs) < min_neighbors:
                continue

            target_r = batch_r[local_i]
            ray_dir  = batch_unit_xyz[local_i]
            nbr_xyz  = xyz[valid_nbrs]

            proj             = nbr_xyz @ ray_dir
            dist_from_target = np.abs(proj - target_r)

            total_within_max = np.sum(dist_from_target <= max_radial_thresh)
            total_within_min = np.sum(dist_from_target <  min_radial_thresh)
            count_in_zone    = total_within_max - total_within_min

            if count_in_zone >= min_neighbors:
                is_comet[batch_start + local_i] = True

    return is_comet


def _refine_by_neighbor_tail_rate(xyz, is_comet, radius, batch_size):
    n = len(xyz)
    tree = cKDTree(xyz, leafsize=100)

    MAX_NEIGHBORS = 100
    tail_rates = np.zeros(n, dtype=float)

    print("   Computing neighbor tail rates...")
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        if batch_start % 1_000_000 == 0 and batch_start > 0:
            print(f"    {batch_start:,} / {n:,}  ({100*batch_start/n:.0f}%)")
            gc.collect()

        batch_xyz = xyz[batch_start:batch_end]

        _, nbr_indices = tree.query(
            batch_xyz, k=MAX_NEIGHBORS,
            distance_upper_bound=radius, workers=-1
        )

        for local_i, nbrs in enumerate(nbr_indices):
            global_i   = batch_start + local_i
            valid_nbrs = nbrs[(nbrs < n) & (nbrs != global_i)]
            if len(valid_nbrs) == 0:
                continue
            tail_rates[global_i] = np.sum(is_comet[valid_nbrs]) / len(valid_nbrs)

    has_neighbors = tail_rates > 0
    avg_rate = np.mean(tail_rates[has_neighbors])
    print(f"   Average neighbor tail rate: {avg_rate:.4f}")

    is_comet_refined  = is_comet.copy()
    is_comet_refined |= tail_rates >= 0.5
    return is_comet_refined


def classify_comet_points(xyz, angular_thresh, min_radial_thresh,
                          radial_thresh, max_neighbors, min_neighbors,
                          batch_size):
    labels = np.full(len(xyz), NORMAL, dtype=np.int8)

    print("Scanning point cloud for ray artifacts...")
    is_comet = _find_ray_candidates(xyz, angular_thresh, min_radial_thresh,
                                    radial_thresh, max_neighbors, min_neighbors, batch_size)
    is_comet = _refine_by_neighbor_tail_rate(xyz, is_comet, radius=1, batch_size=batch_size)

    n_candidates = int(is_comet.sum())
    print(f"Found {n_candidates:,} comet tail points ({100 * n_candidates / len(xyz):.1f}%).")

    labels[is_comet] = TAIL
    return labels


def save_classified_las(las_src, labels, output_path):
    out = laspy.LasData(las_src.header.copy())
    out.points = las_src.points.copy()
    if not hasattr(out, "comet_class"):
        out.add_extra_dim(laspy.ExtraBytesParams(
            name="comet_class", type=np.int8, description="0=normal 1=tail",
        ))
    out.comet_class = labels
    out.write(output_path)
    print(f"Saved -> {output_path}")


def save_filtered_las(las_src, labels, output_path):
    keep_mask = labels == NORMAL
    removed = int((~keep_mask).sum())

    out = laspy.LasData(las_src.header.copy())
    out.points = las_src.points[keep_mask].copy()
    out.write(output_path)

    print(
        f"Removed {removed:,} comet tail points; "
        f"kept {int(keep_mask.sum()):,} points."
    )
    print(f"Saved filtered LAS -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify LiDAR comet-tail artifacts in a LAS/LAZ file."
    )
    parser.add_argument("input_file", help="Input .las/.laz file")
    parser.add_argument("output_file", help="Output .las/.laz file")
    parser.add_argument("--angular-thresh", type=float, default=0.003)
    parser.add_argument("--min-radial-thresh", type=float, default=0.02)
    parser.add_argument("--radial-thresh", type=float, default=0.25)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--min-neighbors", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_FILE  = args.input_file
    OUTPUT_FILE = args.output_file

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        sys.exit(1)

    print(f"Reading {INPUT_FILE} ...")
    las = laspy.read(INPUT_FILE)

    xyz = np.ascontiguousarray(las.xyz, dtype=np.float32)
    print(f"{len(xyz):,} points loaded\n")

    labels = classify_comet_points(
        xyz,
        angular_thresh=args.angular_thresh,
        min_radial_thresh=args.min_radial_thresh,
        radial_thresh=args.radial_thresh,
        max_neighbors=args.max_neighbors,
        min_neighbors=args.min_neighbors,
        batch_size=args.batch_size,
    )
    save_filtered_las(las, labels, OUTPUT_FILE)
    print("\nDone.")
