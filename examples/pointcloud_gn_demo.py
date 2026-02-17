#!/usr/bin/env python3
"""
Synthetic point-cloud alignment demo with a Gauss-Newton solver.

Pipeline:
1) Sample random SMPL shape params beta_true.
2) Generate target point cloud from SMPL vertices.
3) Start from beta_init and optimize with point-to-plane residuals
   r_i = n_i^T (p_i(beta) - q_i), where q_i is NN in target cloud.
4) Visualize target + initial + final clouds (Open3D if available,
   otherwise matplotlib).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for this demo") from exc


@dataclass
class SMPLLinearModel:
    v: np.ndarray  # (3N,)
    v_dirs: np.ndarray  # (P, 3N)

    @property
    def num_vertices(self) -> int:
        return self.v.shape[0] // 3

    @property
    def num_betas(self) -> int:
        return self.v_dirs.shape[0]

    def vertices(self, beta: np.ndarray) -> np.ndarray:
        p = self.v + self.v_dirs.T @ beta
        return p.reshape(-1, 3)

    def vertex_dirs(self) -> np.ndarray:
        # (N, 3, P)
        return np.transpose(self.v_dirs.reshape(self.num_betas, -1, 3), (1, 2, 0))


def load_linear_smpl(npz_path: Path) -> SMPLLinearModel:
    data = np.load(npz_path)
    if "v" not in data or "vDirs" not in data:
        raise ValueError(f"{npz_path} missing 'v' or 'vDirs' keys")

    v = np.asarray(data["v"], dtype=np.float64).reshape(-1)
    v_dirs = np.asarray(data["vDirs"], dtype=np.float64)

    if v_dirs.ndim != 2:
        raise ValueError("Expected vDirs as a 2D array (P, 3N)")

    return SMPLLinearModel(v=v, v_dirs=v_dirs)


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    tree = cKDTree(points)
    _, nn_idx = tree.query(points, k=k)
    normals = np.zeros_like(points)

    for i in range(points.shape[0]):
        nbrs = points[nn_idx[i]]
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, np.argmin(eigvals)]

        # orient outward-ish by consistency with point direction
        if np.dot(n, points[i]) < 0:
            n = -n
        normals[i] = n

    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals / norms


def point_to_plane_gn(
    model: SMPLLinearModel,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    beta_init: np.ndarray,
    max_iters: int = 30,
    damping: float = 1e-5,
) -> Tuple[np.ndarray, list[float]]:
    beta = beta_init.copy()
    dirs = model.vertex_dirs()  # (N,3,P)
    tree = cKDTree(target_points)

    costs: list[float] = []
    for _ in range(max_iters):
        points = model.vertices(beta)
        _, nn_idx = tree.query(points, k=1)
        q = target_points[nn_idx]
        n = target_normals[nn_idx]

        # residual: (N,)
        r = np.sum(n * (points - q), axis=1)
        cost = float(0.5 * np.mean(r**2))
        costs.append(cost)

        # J_i = n_i^T * D_i, D_i in R^{3xP}
        J = np.einsum("ni,nip->np", n, dirs)  # (N, P)

        h = J.T @ J
        g = J.T @ r

        h += damping * np.eye(h.shape[0])
        delta = -np.linalg.solve(h, g)

        beta += delta

        if np.linalg.norm(delta) < 1e-8:
            break

    return beta, costs


def visualize_clouds(
    target: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    output_dir: Optional[Path] = None,
) -> None:
    try:
        import open3d as o3d  # type: ignore

        pcd_t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
        pcd_i = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(initial))
        pcd_f = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final))

        pcd_t.paint_uniform_color([0.2, 0.8, 0.2])  # target: green
        pcd_i.paint_uniform_color([0.2, 0.2, 0.9])  # initial: blue
        pcd_f.paint_uniform_color([0.9, 0.2, 0.2])  # final: red

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(output_dir / "target.ply"), pcd_t)
            o3d.io.write_point_cloud(str(output_dir / "initial.ply"), pcd_i)
            o3d.io.write_point_cloud(str(output_dir / "final.ply"), pcd_f)

        o3d.visualization.draw_geometries(
            [pcd_t, pcd_i, pcd_f],
            window_name="SMPL PointCloud GN (target=green, initial=blue, final=red)",
        )
        return
    except Exception:
        pass

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    step = max(1, target.shape[0] // 2500)
    ax.scatter(target[::step, 0], target[::step, 1], target[::step, 2], s=1, c="g", label="target")
    ax.scatter(initial[::step, 0], initial[::step, 1], initial[::step, 2], s=1, c="b", label="initial")
    ax.scatter(final[::step, 0], final[::step, 1], final[::step, 2], s=1, c="r", label="final")

    ax.set_title("SMPL PointCloud GN (target=green, initial=blue, final=red)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "pointcloud_gn.png", dpi=180)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to preprocessed SMPL .npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--beta-scale", type=float, default=0.04)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--damping", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=Path("examples/pointcloud_gn_outputs"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = load_linear_smpl(args.model)

    p = min(args.num_betas, model.num_betas)
    beta_true = np.zeros(model.num_betas)
    beta_true[:p] = rng.normal(scale=args.beta_scale, size=p)

    beta_init = np.zeros(model.num_betas)
    beta_init[:p] = rng.normal(scale=args.beta_scale, size=p)

    target = model.vertices(beta_true)
    initial = model.vertices(beta_init)

    target_normals = estimate_normals(target, k=20)

    beta_est, costs = point_to_plane_gn(
        model=model,
        target_points=target,
        target_normals=target_normals,
        beta_init=beta_init,
        max_iters=args.iters,
        damping=args.damping,
    )

    final = model.vertices(beta_est)

    print(f"iters: {len(costs)}")
    print(f"initial cost: {costs[0]:.6e}")
    print(f"final cost:   {costs[-1]:.6e}")
    print(f"||beta_true - beta_init||: {np.linalg.norm(beta_true - beta_init):.6e}")
    print(f"||beta_true - beta_est ||: {np.linalg.norm(beta_true - beta_est):.6e}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "estimation.npz",
        beta_true=beta_true,
        beta_init=beta_init,
        beta_est=beta_est,
        costs=np.asarray(costs),
    )

    visualize_clouds(target, initial, final, output_dir=args.out)


if __name__ == "__main__":
    main()
