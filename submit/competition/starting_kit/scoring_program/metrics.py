"""
Distributional realism metrics for the UrbanTwin sim2real LiDAR challenge.

These mirror the protocol in Shahbaz & Agarwal, "High-Fidelity Digital Twins
for Bridging the Sim2Real Gap in LiDAR-Based ITS Perception" (IEEE T-ITS,
2026), with one deliberate deviation: where the paper computes MMD/EMD/FPD
in the SEED detector's 256-d backbone latent feature space, the challenge
computes them in the raw point-cloud space (3-d). This avoids embedding a
spconv/CUDA-only inference pipeline in the Codabench scoring program. CD
matches the paper exactly (it is already raw-space there).

Aggregation matches the paper:
    1. Concatenate all 50 frames per side (synthetic, real reference).
    2. Mask both to the track's POINT_CLOUD_RANGE.
    3. Subsample uniformly at random to N_SUBSAMPLE = 10 000 points each.
    4. Compute one CD, one MMD, one EMD, one FPD across the entire 10 000-point sets.

This produces a single scalar per metric per submission, exactly like Fig. 7
in the paper. RNG seed is fixed (1234) so the local evaluator and the server
produce identical numbers.

Notes / caveats:
    * MMD bandwidth uses the median heuristic on pairwise squared distances
      pooled across both samples, since the paper's gamma=1/256 is calibrated
      to 256-d features and would saturate the kernel in 3-d.
    * FPD on 3-d points fits a 3-variate Gaussian; with only 9 free parameters
      this metric is necessarily less expressive than the paper's 256-d FPD.
      Treat the absolute value as a sanity check; rankings remain meaningful.
"""

from __future__ import annotations

import numpy as np
import ot
from scipy import linalg
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


N_SUBSAMPLE = 10_000


def _mask_to_range(pts: np.ndarray, pc_range: tuple) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    m = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
        & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )
    return pts[m]


def _concat_and_mask(clouds: list, pc_range: tuple) -> np.ndarray:
    parts = [_mask_to_range(c, pc_range) for c in clouds]
    parts = [p for p in parts if p.shape[0] > 0]
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _subsample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] >= n:
        idx = rng.choice(pts.shape[0], n, replace=False)
    else:
        idx = rng.choice(pts.shape[0], n, replace=True)
    return pts[idx]


def chamfer_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Chamfer Distance with squared-L2 nearest-neighbor cost.

    CD(P, Q) = mean_{p in P} min_{q in Q} ||p - q||^2
            +  mean_{q in Q} min_{p in P} ||p - q||^2
    """
    tree_q = cKDTree(q)
    tree_p = cKDTree(p)
    d_pq, _ = tree_q.query(p, k=1)
    d_qp, _ = tree_p.query(q, k=1)
    return float(np.mean(d_pq ** 2) + np.mean(d_qp ** 2))


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray,
                            sample_size: int = 1000,
                            rng: np.random.Generator | None = None) -> float:
    """Bandwidth gamma = 1 / (2 * median(||x - y||^2)) over a small joint sample."""
    rng = rng or np.random.default_rng(0)
    nx = min(sample_size, X.shape[0])
    ny = min(sample_size, Y.shape[0])
    Xs = X[rng.choice(X.shape[0], nx, replace=False)]
    Ys = Y[rng.choice(Y.shape[0], ny, replace=False)]
    pooled = np.vstack([Xs, Ys])
    D2 = cdist(pooled, pooled, "sqeuclidean")
    med = float(np.median(D2[D2 > 0])) if (D2 > 0).any() else 1.0
    return 1.0 / max(2.0 * med, 1e-12)


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray,
                    rng: np.random.Generator | None = None) -> dict:
    """Unbiased RBF-MMD between two equal-or-near-equal-size point sets.

    Bandwidth selected by the median heuristic on a 1k joint subsample so
    the kernel doesn't saturate at the spatial scale of LiDAR point clouds.
    """
    n, m = X.shape[0], Y.shape[0]
    gamma = _median_heuristic_gamma(X, Y, rng=rng)
    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)
    mmd = (
        K_XX.sum() / (n * (n - 1))
        + K_YY.sum() / (m * (m - 1))
        - 2.0 * K_XY.mean()
    )
    return {"MMD": float(mmd), "MMD_gamma": float(gamma)}


def compute_emd(X: np.ndarray, Y: np.ndarray) -> float:
    """Wasserstein-1 (Earth Mover's) distance with Euclidean cost.

    Solves the exact transport via POT's network simplex (`ot.emd2`).
    For N <= 10 000 this completes in tens of seconds on CPU.
    Returns sqrt(emd2) so the value is in metres, matching the paper's
    reported EMD numbers.
    """
    a = np.full(X.shape[0], 1.0 / X.shape[0])
    b = np.full(Y.shape[0], 1.0 / Y.shape[0])
    M = cdist(X, Y, metric="euclidean")
    # Network simplex on 10k x 10k can need many iterations to fully converge;
    # 10M is comfortably above what real submissions hit in practice.
    emd2 = ot.emd2(a, b, M, numItermax=10_000_000)
    return float(np.sqrt(max(emd2, 0.0)))


def _frechet_gaussian(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def compute_fpd(X: np.ndarray, Y: np.ndarray) -> float:
    """Fréchet Point-cloud Distance (FID-style) between two point sets,
    treating each point as a sample from a 3-variate Gaussian."""
    mu1 = X.mean(axis=0)
    mu2 = Y.mean(axis=0)
    sigma1 = np.cov(X, rowvar=False)
    sigma2 = np.cov(Y, rowvar=False)
    return _frechet_gaussian(mu1, sigma1, mu2, sigma2)


def compute_realism(syn_clouds: list, real_clouds: list,
                    pc_range: tuple, seed: int = 1234) -> dict:
    """Run all four dataset-level metrics and return a flat result dict.

    Aggregation: concatenate 50 frames per side, mask to range, subsample
    each side to N_SUBSAMPLE points, then compute one scalar per metric.
    """
    rng = np.random.default_rng(seed)

    syn_all = _concat_and_mask(syn_clouds, pc_range)
    real_all = _concat_and_mask(real_clouds, pc_range)
    if syn_all.shape[0] == 0 or real_all.shape[0] == 0:
        return {
            "CD": float("inf"),
            "MMD": float("inf"),
            "EMD": float("inf"),
            "FPD": float("inf"),
            "n_syn_points_pre_subsample": int(syn_all.shape[0]),
            "n_real_points_pre_subsample": int(real_all.shape[0]),
            "MMD_gamma": 0.0,
        }

    Xs = _subsample(syn_all, N_SUBSAMPLE, rng)
    Xr = _subsample(real_all, N_SUBSAMPLE, rng)

    cd = chamfer_distance(Xs, Xr)
    mmd_out = compute_mmd_rbf(Xs, Xr, rng=rng)
    emd = compute_emd(Xs, Xr)
    fpd = compute_fpd(Xs, Xr)

    return {
        "CD": float(cd),
        "MMD": float(mmd_out["MMD"]),
        "EMD": float(emd),
        "FPD": float(fpd),
        "n_syn_points_pre_subsample": int(syn_all.shape[0]),
        "n_real_points_pre_subsample": int(real_all.shape[0]),
        "MMD_gamma": float(mmd_out["MMD_gamma"]),
    }
