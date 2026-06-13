"""Демо предсказания биомассы для одного облака.

Берёт пять regression-CSV (voxel / layered-alpha / CHM / height-percentile /
count), выбирает первую строку (в наших CSV они уже отсортированы по убыванию
R²), парсит из неё параметры метода и коэффициенты best_model. Выбирает
медианное по биомассе облако из --list (как visualize_methods.py), прогоняет на
нём каждый метод, подставляет полученный скаляр в регрессию и печатает таблицу
с абс./отн. ошибкой.

Пример:
    uv run python predict_biomass.py \\
        --list data/some_list.txt \\
        --voxel-csv  results/regression_csv/voxel/<...>.csv \\
        --alpha-csv  results/regression_csv/alpha/<...>.csv \\
        --chm-csv    results/regression_csv/chm/<...>.csv \\
        --height-csv results/regression_csv/height/<...>.csv \\
        --count-csv  results/regression_csv/count/<...>.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import open3d as o3d
import pandas as pd
from dotenv import load_dotenv

from methods.chm import chm_volume
from cloud_pipeline import PreprocessConfig, preprocess_cloud
from geometry import alpha_layered
from visualize_methods import pick_median_biomass
from volume_methods import voxel_volume


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.env_file and os.path.exists(pre_args.env_file):
        load_dotenv(pre_args.env_file, override=True)
    elif os.path.exists(".env"):
        load_dotenv(".env", override=True)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--list", dest="list_file",
                     help="Список plot'ов (формат batch_process.parse_list_line); "
                          "выбирается медианное по биомассе облако")
    src.add_argument("--cloud", dest="cloud_file",
                     help="Путь к одному облаку (без GT биомассы)")
    p.add_argument("--base-dir", default=os.getenv("TPCVE_BASE_DIR", "data"))
    p.add_argument("--voxel-csv", required=True)
    p.add_argument("--alpha-csv", required=True)
    p.add_argument("--chm-csv", required=True)
    p.add_argument("--height-csv", required=True)
    p.add_argument("--count-csv", required=True)

    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "2.0")))
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04)
    p.add_argument("--output-json", default=None)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Regression CSV → predict callable + params

def _build_predict(row: pd.Series) -> tuple[str, Callable[[float], float], str]:
    model = str(row["best_model"])
    if model == "power":
        a = float(row["power_a"])
        b = float(row["power_b"])
        coefs = f"a={a:.4g}, b={b:.4g}"

        def predict(x: float) -> float:
            return a * (x ** b) if x > 0 else float("nan")
    elif model in ("linear", "huber"):
        slope = float(row[f"{model}_slope"])
        intercept = float(row[f"{model}_intercept"])
        coefs = f"slope={slope:.4g}, b0={intercept:.4g}"

        def predict(x: float) -> float:
            return slope * x + intercept
    else:
        raise ValueError(f"Неизвестная модель best_model={model!r}")
    return model, predict, coefs


def _load_voxel(csv_path: str) -> dict:
    row = pd.read_csv(csv_path).iloc[0]
    voxel_mm = float(row["voxel_mm"])
    model, predict, coefs = _build_predict(row)
    train_r2 = float(row[f"{model}_r2"])
    return {
        "kind": "voxel",
        "params_str": f"size={voxel_mm:g}mm",
        "params": {"voxel_size_m": voxel_mm / 1000.0},
        "model": model, "predict": predict, "coefs": coefs,
        "train_r2": train_r2,
    }


def _load_alpha(csv_path: str) -> dict:
    row = pd.read_csv(csv_path).iloc[0]
    mode = str(row["mode"])
    if mode != "layered":
        raise ValueError(f"alpha CSV: поддерживается только mode=layered, "
                         f"получили {mode!r}")
    voxel_mm = float(row["voxel_mm"])
    alpha = float(row["alpha"])
    dz_mm = float(row["layer_dz_mm"])
    model, predict, coefs = _build_predict(row)
    train_r2 = float(row[f"{model}_r2"])
    return {
        "kind": "alpha",
        "params_str": (f"a={alpha:g} dz={dz_mm:g}mm"
                       + (f" pre-vox={voxel_mm:g}mm" if voxel_mm > 0 else "")),
        "params": {"voxel_mm": voxel_mm, "alpha": alpha, "dz_m": dz_mm / 1000.0},
        "model": model, "predict": predict, "coefs": coefs,
        "train_r2": train_r2,
    }


def _load_chm(csv_path: str) -> dict:
    row = pd.read_csv(csv_path).iloc[0]
    cell_mm = float(row["cell_size_mm"])
    percentile = float(row["percentile"])
    model, predict, coefs = _build_predict(row)
    train_r2 = float(row[f"{model}_r2"])
    return {
        "kind": "chm",
        "params_str": f"cell={cell_mm:g}mm p={percentile:g}",
        "params": {"cell_size_m": cell_mm / 1000.0, "percentile": percentile},
        "model": model, "predict": predict, "coefs": coefs,
        "train_r2": train_r2,
    }


def _load_height(csv_path: str) -> dict:
    row = pd.read_csv(csv_path).iloc[0]
    percentile = float(row["percentile"])
    model, predict, coefs = _build_predict(row)
    train_r2 = float(row[f"{model}_r2"])
    return {
        "kind": "height",
        "params_str": f"p={percentile:g}",
        "params": {"percentile": percentile},
        "model": model, "predict": predict, "coefs": coefs,
        "train_r2": train_r2,
    }


def _load_count(csv_path: str) -> dict:
    row = pd.read_csv(csv_path).iloc[0]
    source = str(row["source"])
    if source not in ("raw", "pre"):
        raise ValueError(f"count CSV: ожидался source ∈ {{raw,pre}}, "
                         f"получили {source!r}")
    model, predict, coefs = _build_predict(row)
    train_r2 = float(row[f"{model}_r2"])
    return {
        "kind": "count",
        "params_str": f"source={source}",
        "params": {"source": source},
        "model": model, "predict": predict, "coefs": coefs,
        "train_r2": train_r2,
    }


# ----------------------------------------------------------------------------
# Применение метода к облаку

def _compute_x(method: dict, veg: np.ndarray, pre) -> float:
    kind = method["kind"]
    p = method["params"]
    if kind == "voxel":
        vol, _ = voxel_volume(veg, p["voxel_size_m"])
        return float(vol)
    if kind == "alpha":
        pts = veg
        if p["voxel_mm"] > 0 and len(pts):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd = pcd.voxel_down_sample(voxel_size=p["voxel_mm"] / 1000.0)
            pts = np.asarray(pcd.points)
        _, vol = alpha_layered(pts, p["alpha"], p["dz_m"], with_rings=False)
        return float(vol)
    if kind == "chm":
        vol, _ = chm_volume(veg, p["cell_size_m"], p["percentile"])
        return float(vol)
    if kind == "height":
        if len(veg) == 0:
            return 0.0
        return float(np.percentile(veg[:, 2], p["percentile"]))
    if kind == "count":
        return float(pre.n_input if p["source"] == "raw" else len(veg))
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# main

def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)

    if args.list_file:
        cloud_path, biomass_gt, _ = pick_median_biomass(
            args.list_file, base_dir, stage=None)
    else:
        cloud_path = Path(args.cloud_file).resolve()
        biomass_gt = None

    methods = [
        _load_voxel(args.voxel_csv),
        _load_alpha(args.alpha_csv),
        _load_chm(args.chm_csv),
        _load_height(args.height_csv),
        _load_count(args.count_csv),
    ]

    cfg = PreprocessConfig(
        units=args.units, flip_z=args.flip_z, downsample=args.downsample,
        sor_std_ratio=args.sor_std_ratio, sor_neighbors=args.sor_neighbors,
        min_range=args.min_range, height_threshold=args.height_threshold,
        verbose=False,
    )
    pre = preprocess_cloud(str(cloud_path), cfg)
    veg = pre.vegetation

    try:
        rel = cloud_path.relative_to(base_dir.resolve())
    except ValueError:
        rel = cloud_path

    print(f"Cloud:        {rel}")
    print(f"Points:       in={pre.n_input}  sor={pre.n_after_sor}  "
          f"veg={len(veg)}")
    if biomass_gt is not None:
        print(f"True biomass: {biomass_gt:.3f} g/m²")
    else:
        print("True biomass: N/A (single-cloud mode)")
    print()

    units = {"voxel": "m³", "alpha": "m³", "chm": "m³",
             "height": "m", "count": "pts"}
    has_gt = biomass_gt is not None
    if has_gt:
        header = (f"{'method':<6} | {'params':<28} | {'model':<6} | "
                  f"{'x_pred':<14} | {'y_pred (g/m²)':>13} | "
                  f"{'abs_err (g/m²)':>14} | {'rel_err':>8} | {'train R²':>8}")
    else:
        header = (f"{'method':<6} | {'params':<28} | {'model':<6} | "
                  f"{'x_pred':<14} | {'y_pred (g/m²)':>13} | {'train R²':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for m in methods:
        x = _compute_x(m, veg, pre)
        y = float(m["predict"](x))
        unit = units.get(m["kind"], "")
        if has_gt:
            abs_err = y - biomass_gt
            rel_err_pct = abs_err / biomass_gt * 100 if biomass_gt > 0 else float("nan")
            print(f"{m['kind']:<6} | {m['params_str']:<28} | {m['model']:<6} | "
                  f"{x:>10.4f} {unit:<3} | {y:>13.2f} | {abs_err:>+14.2f} | "
                  f"{rel_err_pct:>+7.2f}% | {m['train_r2']:>8.3f}")
            results.append({
                "method": m["kind"], "params": m["params_str"],
                "model": m["model"], "coefs": m["coefs"],
                "x_pred": x, "y_pred": y,
                "abs_err": abs_err, "rel_err_pct": rel_err_pct,
                "train_r2": m["train_r2"],
            })
        else:
            print(f"{m['kind']:<6} | {m['params_str']:<28} | {m['model']:<6} | "
                  f"{x:>10.4f} {unit:<3} | {y:>13.2f} | {m['train_r2']:>8.3f}")
            results.append({
                "method": m["kind"], "params": m["params_str"],
                "model": m["model"], "coefs": m["coefs"],
                "x_pred": x, "y_pred": y,
                "train_r2": m["train_r2"],
            })

    if args.output_json:
        out = {
            "cloud": str(rel),
            "biomass_true": biomass_gt if biomass_gt is not None else None,
            "n_input": pre.n_input, "n_after_sor": pre.n_after_sor,
            "n_vegetation": int(len(veg)), "predictions": results,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
