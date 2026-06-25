"""Препроцессинг облака точек: загрузка → flip-z/нормализация → min-range →
voxel downsample → SOR → классификация земля/растительность.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from tpcve.cloud.generate_cloud import load_cloud, load_real_cloud
from tpcve.cloud.geometry import voxel_downsample_np, sor_np

CACHE_VERSION = 1


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
    cache_dir: str | None = None


@dataclass
class PreprocessResult:
    vegetation: np.ndarray
    ground: np.ndarray
    n_input: int
    n_after_sor: int


def _log(cfg: PreprocessConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg)


def _config_hash(cfg: PreprocessConfig) -> str:
    raw = json.dumps({k: v for k, v in asdict(cfg).items()
                      if k not in ("cache_dir", "verbose")},
                     sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_key(path: str) -> str:
    return hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()


def _load_cache(path: str, cfg: PreprocessConfig) -> PreprocessResult | None:
    if not cfg.cache_dir:
        return None
    cache_file = Path(cfg.cache_dir) / f"{_cache_key(path)}.npz"
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file)
        stored_cfg_hash = str(data.get("config_hash", b""))
        if stored_cfg_hash != _config_hash(cfg):
            cache_file.unlink(missing_ok=True)
            return None
        return PreprocessResult(
            vegetation=data["vegetation"],
            ground=data["ground"],
            n_input=int(data["n_input"]),
            n_after_sor=int(data["n_after_sor"]),
        )
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None


def _save_cache(path: str, cfg: PreprocessConfig, res: PreprocessResult) -> None:
    if not cfg.cache_dir:
        return
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(path)}.npz"
    np.savez_compressed(cache_file,
                        vegetation=res.vegetation,
                        ground=res.ground,
                        n_input=res.n_input,
                        n_after_sor=res.n_after_sor,
                        config_hash=_config_hash(cfg))


def preprocess_cloud(path: str, cfg: PreprocessConfig) -> PreprocessResult:
    cached = _load_cache(path, cfg)
    if cached is not None:
        return cached

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

    res = PreprocessResult(
        vegetation=vegetation,
        ground=ground,
        n_input=n_input,
        n_after_sor=len(pts_filtered),
    )
    _save_cache(path, cfg, res)
    return res
