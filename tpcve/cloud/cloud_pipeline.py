"""Препроцессинг облака точек: загрузка → flip-z/нормализация → min-range →
voxel downsample → SOR → классификация земля/растительность.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tpcve.cloud.generate_cloud import load_cloud, load_real_cloud
from tpcve.cloud.geometry import voxel_downsample_np, sor_np


@dataclass
class PreprocessConfig:
    units: str = "auto"
    flip_z: bool = False
    downsample: float = 0.0
    sor_std_ratio: float = 1.5
    sor_neighbors: int = 20
    min_range: float = 0.0
    height_threshold: float = 0.04
    verbose: bool = False


@dataclass
class PreprocessResult:
    vegetation: np.ndarray
    ground: np.ndarray
    n_input: int
    n_after_sor: int


def _log(cfg: PreprocessConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg)


def preprocess_cloud(path: str, cfg: PreprocessConfig) -> PreprocessResult:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".npz":
        data = load_cloud(str(p))
    else:
        data = load_real_cloud(str(p), units=cfg.units, verbose=cfg.verbose)

    pts = np.asarray(data["all_pts_noisy"]).copy()
    scanner_pos = np.asarray(data.get("scanner_pos", np.zeros(3)))
    n_input = len(pts)

    # Z-нормализация (только для не-NPZ — синтетика уже нормирована)
    if ext != ".npz":
        if cfg.flip_z:
            pts[:, 2] = -pts[:, 2]
        if len(pts):
            pts[:, 2] -= pts[:, 2].min()

    # Min-range фильтр (по XY)
    if cfg.min_range > 0 and len(pts):
        dxy = np.linalg.norm(pts[:, :2] - scanner_pos[:2], axis=1)
        pts = pts[dxy >= cfg.min_range]

    if cfg.downsample > 0:
        pts = voxel_downsample_np(pts, cfg.downsample)

    if len(pts) > cfg.sor_neighbors:
        pts = sor_np(pts, cfg.sor_neighbors, cfg.sor_std_ratio)

    pts_filtered = pts

    if len(pts_filtered):
        ground_mask = pts_filtered[:, 2] < cfg.height_threshold
        ground = pts_filtered[ground_mask]
        vegetation = pts_filtered[~ground_mask]
    else:
        ground = pts_filtered
        vegetation = pts_filtered

    _log(cfg, f"  {p.name}: in={n_input} → sor={len(pts_filtered)} "
              f"→ veg={len(vegetation)}")

    return PreprocessResult(
        vegetation=vegetation,
        ground=ground,
        n_input=n_input,
        n_after_sor=len(pts_filtered),
    )
