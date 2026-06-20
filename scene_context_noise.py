"""
scene_context_noise.py
================================================================================
Per-point noise-probability scoring via local spatial context.

Input : a (pre-filtered) .las file
Output: same file + extra float32 dim 'noise_prob' in [0, 1]
        (optionally also uint8 'noise_label' if a threshold is supplied)

Two complementary signals
--------------------------
1. Density sparsity
     Distance to the K-th nearest neighbour, normalised by the global median.
     Points in sparse patches have a large d_k  →  higher noise score.

2. Local geometric coherence  (PCA eigenvalues of k-NN neighbourhood)
     Linearity    (l1-l2)/l1       high  →  tail-like line    →  noise
     Low curvature = 1 - 3*l3/(l1+l2+l3)                      →  noise
     Planarity    (l2-l3)/l1       high  →  smooth surface     →  real

Final  noise_prob = w_density * density_signal
                  + w_line    * linearity
                  + w_lowcurv * low_curvature
"""

import os
import sys
import gc
from typing import Optional, Tuple

import numpy as np
import laspy
from scipy.spatial import cKDTree

# ── Tunable defaults ──────────────────────────────────────────────────────────
K_DENSITY  = 20      # k-NN used for density estimation
K_EIGEN    = 20      # k-NN used for PCA geometry features
W_DENSITY  = 0.50    # weight: density sparsity
W_LINE     = 0.35    # weight: tail-like linear structure
W_LOWCURV  = 0.15    # weight: low curvature / non-random scatter
BATCH_SIZE = 50_000


# ── Signal 1: density sparsity ────────────────────────────────────────────────

def _density_signal(xyz: np.ndarray, k: int, batch_size: int) -> np.ndarray:
    """
    Returns score in [0, 1]: higher = sparser local neighbourhood = more noise.

    Uses d_k (distance to k-th non-self nearest neighbour) normalised by a
    range-stratified median: points are binned by distance from the scanner
    origin so that naturally sparser far-field points are not penalised.
    Linear ramp: bin_median -> 0,  4x bin_median -> 1.
    """
    n    = len(xyz)
    tree = cKDTree(xyz, leafsize=64)
    dk   = np.empty(n, np.float32)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        if s and s % 500_000 == 0:
            print(f"    density  {s:,}/{n:,}")
            gc.collect()
        d, _ = tree.query(xyz[s:e], k=k + 1, workers=-1)  # k+1: skip self
        dk[s:e] = d[:, k]                                  # k-th non-self dist

    # Range-stratified normalisation: 20 equal-count bins by distance from origin.
    # Each bin gets its own median so far-field natural sparsity is not penalised.
    r      = np.linalg.norm(xyz, axis=1).astype(np.float32)
    N_BINS = 20
    edges  = np.percentile(r, np.linspace(0, 100, N_BINS + 1))
    edges[-1] += 1e-3          # ensure the farthest point falls inside last bin
    local_med = np.empty(n, np.float32)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r >= lo) & (r < hi)
        if mask.sum() > 0:
            local_med[mask] = np.median(dk[mask])

    signal = np.clip((dk / (local_med + 1e-9) - 1.0) / 3.0, 0.0, 1.0)
    return signal.astype(np.float32)


# ── Signal 2: local geometric coherence ───────────────────────────────────────

def _eigen_features(
    xyz: np.ndarray, k: int, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised batched PCA on k non-self nearest neighbours.
    Returns (linearity, low_curvature, planarity), each in [0, 1].
    """
    n    = len(xyz)
    tree = cKDTree(xyz, leafsize=64)

    linearity = np.empty(n, np.float32)
    low_curvature = np.empty(n, np.float32)
    planarity = np.empty(n, np.float32)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        if s and s % 500_000 == 0:
            print(f"    PCA      {s:,}/{n:,}")
            gc.collect()

        _, idx = tree.query(xyz[s:e], k=k + 1, workers=-1)  # (B, k+1)
        pts    = xyz[idx[:, 1:]]                             # (B, k, 3) skip self
        c      = pts - pts.mean(axis=1, keepdims=True)       # centred
        cov    = np.einsum('bki,bkj->bij', c, c) / (k - 1)  # (B, 3, 3)

        ev = np.linalg.eigvalsh(cov)   # (B, 3) ascending eigenvalues
        ev = np.maximum(ev, 0.0)
        l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]           # l1 largest

        d = l1 + 1e-9
        total = l1 + l2 + l3 + 1e-9
        planarity [s:e] = (l2 - l3) / d
        linearity [s:e] = (l1 - l2) / d
        low_curvature[s:e] = 1.0 - np.clip(3.0 * l3 / total, 0.0, 1.0)

    return linearity, low_curvature, planarity


# ── Combine into noise probability ────────────────────────────────────────────

def score_noise(
    xyz:        np.ndarray,
    k_density:  int   = K_DENSITY,
    k_eigen:    int   = K_EIGEN,
    w_density:  float = W_DENSITY,
    w_line:     float = W_LINE,
    w_lowcurv:  float = W_LOWCURV,
    batch_size: int   = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (noise_prob, density_signal, linearity, low_curvature), each float32 in [0, 1].
    noise_prob = w_density*density + w_line*linearity + w_lowcurv*low_curvature
    """
    print("[1/2] Density sparsity signal...")
    dens_sig = _density_signal(xyz, k_density, batch_size)

    print("[2/2] Eigenvalue (geometric coherence) features...")
    lin, low_curv, _plan = _eigen_features(xyz, k_eigen, batch_size)

    noise_prob = (
        w_density * dens_sig    +
        w_line    * lin         +
        w_lowcurv * low_curv
    )
    return (
        np.clip(noise_prob, 0.0, 1.0).astype(np.float32),
        dens_sig,
        lin.astype(np.float32),
        low_curv.astype(np.float32),
    )


# ── I/O pipeline ──────────────────────────────────────────────────────────────

def _read_las_safe(path: str) -> "laspy.LasData":
    """
    Read a LAS file. If it contains duplicate extra-byte field names
    (e.g. a file processed twice by comet-old.py), repair the raw bytes
    in-memory before handing off to laspy.
    """
    _orig_exc = None
    try:
        return laspy.read(path)
    except ValueError as exc:
        if "occurs more than once" not in str(exc):
            raise
        _orig_exc = exc

    import io, struct

    _W      = {0:0,1:1,2:1,3:2,4:2,5:4,6:4,7:8,8:8,9:4,10:8}
    VLR_HDR = 54   # LAS VLR header size in bytes
    EB_REC  = 192  # one extra-byte specification record

    with open(path, "rb") as fh:
        raw = bytearray(fh.read())

    ver_minor = raw[25]
    hdr_size  = struct.unpack_from("<H", raw, 94)[0]

    if ver_minor < 4:
        offset_pts = struct.unpack_from("<I", raw, 96)[0]
        pt_rec_len = struct.unpack_from("<H", raw, 105)[0]
        n_points   = struct.unpack_from("<I", raw, 107)[0]
    else:
        offset_pts = struct.unpack_from("<Q", raw, 96)[0]
        pt_rec_len = struct.unpack_from("<H", raw, 109)[0]
        n_points   = struct.unpack_from("<Q", raw, 251)[0]

    # ── Locate Extra Bytes VLR by scanning for b"LASF_Spec" ─────────────────
    # More robust than count-based walking when a prior VLR has a wrong length.
    # user_id sits at VLR_offset+2; record_id at VLR_offset+18; rlen at +20.
    eb_vlr_pos = eb_data_pos = eb_data_len = None
    needle = b"LASF_Spec"
    scan   = hdr_size
    while scan < offset_pts - VLR_HDR:
        idx = bytes(raw).find(needle, scan, offset_pts)
        if idx < 0:
            break
        vlr_start = idx - 2          # reserved field is 2 bytes before user_id
        if vlr_start < hdr_size:
            scan = idx + 1
            continue
        try:
            rid  = struct.unpack_from("<H", raw, vlr_start + 18)[0]
            rlen = struct.unpack_from("<H", raw, vlr_start + 20)[0]
        except struct.error:
            scan = idx + 1
            continue
        # Validate: record_id==4, data fits before point data, size is a multiple of 192
        if (rid == 4
                and rlen > 0
                and rlen % EB_REC == 0
                and vlr_start + VLR_HDR + rlen <= offset_pts):
            eb_vlr_pos  = vlr_start
            eb_data_pos = vlr_start + VLR_HDR
            eb_data_len = rlen
            break
        scan = idx + 1

    if eb_vlr_pos is None:
        # Emit diagnostics so the user can diagnose unusual file layouts
        print(f"  Debug: ver={raw[24]}.{ver_minor}  hdr_size={hdr_size}"
              f"  offset_pts={offset_pts}  pt_rec_len={pt_rec_len}")
        raise ValueError(f"No extra-bytes VLR found in {path}") from _orig_exc

    # ── Parse extra-byte records; compute each dim's offset in a point record ─
    n_recs      = eb_data_len // EB_REC
    total_extra = sum(_W.get(raw[eb_data_pos + i*EB_REC + 2], 0) for i in range(n_recs))
    std_size    = pt_rec_len - total_extra

    recs, cumoff = [], std_size
    for i in range(n_recs):
        base  = eb_data_pos + i * EB_REC
        dtype = raw[base + 2]
        name  = raw[base+4:base+36].rstrip(b"\x00").decode("ascii", errors="ignore")
        width = _W.get(dtype, 0)
        recs.append({"data": bytes(raw[base:base+EB_REC]),
                     "name": name, "width": width, "off": cumoff})
        cumoff += width

    seen, kept, dups = set(), [], []
    for r in recs:
        (kept if r["name"] not in seen else dups).append(r)
        seen.add(r["name"])

    if not dups:
        raise ValueError("Could not identify duplicate extra-byte dims") from _orig_exc

    removed_vlr_bytes = len(dups) * EB_REC
    removed_pt_bytes  = sum(d["width"] for d in dups)
    new_eb_len        = len(kept) * EB_REC
    new_pt_len        = pt_rec_len - removed_pt_bytes

    # ── Extract & fix point data BEFORE modifying raw ────────────────────────
    pt_raw   = np.frombuffer(
        raw[offset_pts : offset_pts + n_points * pt_rec_len], dtype=np.uint8
    ).copy().reshape(n_points, pt_rec_len)
    dup_cols = [c for d in dups for c in range(d["off"], d["off"] + d["width"])]
    pt_fixed = np.delete(pt_raw, dup_cols, axis=1)

    # ── Patch raw bytes ───────────────────────────────────────────────────────
    struct.pack_into("<H", raw, eb_vlr_pos + 20, new_eb_len)
    raw[eb_data_pos : eb_data_pos + eb_data_len] = b"".join(r["data"] for r in kept)
    new_offset = offset_pts - removed_vlr_bytes
    if ver_minor < 4:
        struct.pack_into("<I", raw, 96,  new_offset)
        struct.pack_into("<H", raw, 105, new_pt_len)
    else:
        struct.pack_into("<Q", raw, 96,  new_offset)
        struct.pack_into("<H", raw, 109, new_pt_len)
    raw[new_offset : new_offset + n_points * pt_rec_len] = pt_fixed.tobytes()

    print(f"  (auto-repaired {len(dups)} duplicate extra-byte field(s))")
    return laspy.read(io.BytesIO(bytes(raw)))


def run(
    input_path:  str,
    output_path: str,
    threshold:   Optional[float] = None,
) -> None:
    """
    Read a LAS file, compute noise_prob per point, write to output.

    If threshold is given, also writes a uint8 'noise_label' field
    (1 = noise, 0 = clean).
    """
    print(f"Reading {input_path} ...")
    las = _read_las_safe(input_path)
    xyz = np.ascontiguousarray(las.xyz, dtype=np.float32)
    print(f"{len(xyz):,} points loaded\n")

    noise_prob, dens_sig, lin, low_curv = score_noise(xyz)

    thr = threshold if threshold is not None else 0.5
    n_noise = int((noise_prob >= thr).sum())
    print(
        f"\nPoints above threshold ({thr:.2f}): "
        f"{n_noise:,}  ({100.0 * n_noise / len(xyz):.1f}%)"
    )
    print(f"noise_prob  mean={noise_prob.mean():.3f}  "
          f"p50={np.percentile(noise_prob, 50):.3f}  "
          f"p90={np.percentile(noise_prob, 90):.3f}  "
          f"p99={np.percentile(noise_prob, 99):.3f}")

    out = laspy.LasData(las.header.copy())
    out.points = las.points.copy()

    out.add_extra_dim(laspy.ExtraBytesParams(
        name='noise_prob', type=np.float32,
        description='noise prob: 0=clean 1=noise',
    ))
    out.noise_prob = noise_prob

    out.add_extra_dim(laspy.ExtraBytesParams(
        name='density_signal', type=np.float32,
        description='sparsity: 0=dense 1=sparse',
    ))
    out.density_signal = dens_sig

    out.add_extra_dim(laspy.ExtraBytesParams(
        name='linearity', type=np.float32,
        description='PCA tail linearity',
    ))
    out.linearity = lin

    out.add_extra_dim(laspy.ExtraBytesParams(
        name='low_curvature', type=np.float32,
        description='1=low curvature',
    ))
    out.low_curvature = low_curv

    if threshold is not None:
        out.add_extra_dim(laspy.ExtraBytesParams(
            name='noise_label', type=np.uint8,
            description='1=noise (thresholded)',
        ))
        out.noise_label = (noise_prob >= threshold).astype(np.uint8)

    out.write(output_path)
    print(f"Saved -> {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    THRESHOLD = 0.5        # set to None to skip binary label output

    if len(sys.argv) < 3:
        print("Usage: python scene_context_noise.py <input.las> <output.las> [threshold]")
        sys.exit(1)

    INPUT, OUTPUT = sys.argv[1], sys.argv[2]
    if len(sys.argv) == 4:
        THRESHOLD = float(sys.argv[3])

    if not os.path.exists(INPUT):
        print(f"Error: {INPUT} not found")
        sys.exit(1)

    run(INPUT, OUTPUT, threshold=THRESHOLD)
    print("Done.")
