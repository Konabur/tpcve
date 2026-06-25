"""Вморозить регрессионную модель в standalone predict.py.

Читает regression CSV (результат analyze.py), извлекает лучшие модели
и создаёт файл predict.py с вшитыми коэффициентами.

Пример:
    python tools/freeze_model.py --stage Z31 --output predict.py
    python tools/freeze_model.py --voxel-csv results/...csv --output predict.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def extract_best_row(csv_path: str) -> dict | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    row = df.iloc[0]
    best_model = str(row["best_model"])
    r2 = float(row.get(f"{best_model}_r2", 0))

    if "voxel_mm" in df.columns and "alpha" in df.columns:
        kind = "alpha"
    elif "voxel_mm" in df.columns:
        kind = "voxel"
    elif "cell_size_mm" in df.columns:
        kind = "chm"
    elif "source" in df.columns:
        kind = "count"
    elif "percentile" in df.columns:
        kind = "percentile"
    else:
        print(f"  [warn] неизвестный формат CSV: {csv_path}", file=sys.stderr)
        return None

    params = {}
    if kind == "voxel":
        params["voxel_mm"] = float(row["voxel_mm"])
    elif kind == "alpha":
        params["voxel_mm"] = float(row.get("voxel_mm", 0))
        params["alpha"] = float(row["alpha"])
        params["dz_m"] = float(row.get("layer_dz_mm", 50)) / 1000.0
    elif kind == "chm":
        params["cell_size_mm"] = float(row["cell_size_mm"])
        params["percentile"] = float(row["percentile"])
    elif kind == "count":
        params["source"] = str(row["source"])
    elif kind == "percentile":
        params["percentile"] = float(row["percentile"])

    stage = None
    stem = Path(csv_path).stem
    for s in ("Z31", "Z65"):
        if f"_{s}_" in f"_{stem}_":
            stage = s
            break

    # Всегда берём linear — разница с power копеечная
    coefs = {"slope": float(row["linear_slope"]),
             "intercept": float(row["linear_intercept"])}
    lmse = float(row.get("linear_rmse", 0))
    lmse_pct = float(row.get("linear_rmse_pct", 0))

    label = _format_label(kind, params)
    return {
        "method": kind,
        "stage": stage,
        "params": params,
        "label": label,
        "x_col": str(row.get("x_col", "?")),
        "model_type": "linear",
        "coefs": coefs,
        "train_r2": round(float(row["linear_r2"]), 4),
        "train_rmse": round(lmse, 2),
        "train_rmse_pct": round(lmse_pct, 2),
    }


def _fmt_dict(d: dict) -> str:
    items = ", ".join(f"{k!r}: {v!r}" for k, v in d.items())
    return "{" + items + "}"


def _format_label(kind: str, params: dict) -> str:
    if kind == "voxel":
        return f"size={params['voxel_mm']:g}mm"
    if kind == "alpha":
        parts = []
        if params.get("voxel_mm", 0) > 0:
            parts.append(f"pre-vox={params['voxel_mm']:g}mm")
        parts.append(f"a={params['alpha']:g}")
        parts.append(f"dz={params['dz_m']*1000:g}mm")
        return " ".join(parts)
    if kind == "chm":
        return f"cell={params['cell_size_mm']:g}mm p={params['percentile']:g}"
    if kind == "count":
        return f"source={params['source']}"
    if kind == "percentile":
        return f"p={params['percentile']:g}"
    return str(params)


def make_models_code(models: list[dict]) -> str:
    lines = ["MODELS = ["]
    for m in models:
        lines.append("    {")
        lines.append(f'        "method": {m["method"]!r},')
        lines.append(f'        "stage": {m["stage"]!r},')
        lines.append(f'        "label": {m["label"]!r},')
        lines.append(f'        "params": {_fmt_dict(m["params"])},')
        lines.append(f'        "x_col": {m["x_col"]!r},')
        lines.append(f'        "model_type": {m["model_type"]!r},')
        lines.append(f'        "coefs": {_fmt_dict(m["coefs"])},')
        lines.append(f'        "train_r2": {m["train_r2"]:.4f},')
        lines.append(f'        "train_rmse": {m["train_rmse"]:.2f},')
        lines.append(f'        "train_rmse_pct": {m["train_rmse_pct"]:.2f},')
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines) + "\n"


PREDICT_TEMPLATE = '''\
#!/usr/bin/env python3
"""predict.py -- cгенерирован автоматически, не редактировать.

Запуск:
    python predict.py --cloud file.pcd
    python predict.py --list data/list.txt --base-dir data --stage Z31
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from tpcve.cloud.cloud_pipeline import PreprocessConfig, preprocess_cloud
from tpcve.cloud.volume_methods import voxel_volume
from tpcve.cloud.geometry import alpha_layered, voxel_downsample_np
from tpcve.methods.chm import chm_volume
from tpcve.core.io import parse_list_line, stage_from_path

# ---------------------------------------------------------------------------
# Модельные коэффициенты (сгенерированы из regression CSV)
# ---------------------------------------------------------------------------
%s


def _apply_model(m: dict, x: float) -> float:
    mt = m["model_type"]
    c = m["coefs"]
    if mt in ("linear", "huber"):
        return c["slope"] * x + c["intercept"]
    if mt == "power":
        return c["a"] * (x ** c["b"]) if x > 0 else float("nan")
    raise ValueError(f"Unknown model_type: {mt}")


def _compute_feature(m: dict, veg: np.ndarray, pre) -> float:
    kind = m["method"]
    p = m["params"]
    if kind == "voxel":
        vol, _ = voxel_volume(veg, p["voxel_mm"] / 1000.0)
        return float(vol)
    if kind == "alpha":
        pts = veg
        vm = p.get("voxel_mm", 0)
        if vm > 0 and len(pts):
            pts = voxel_downsample_np(pts, vm / 1000.0)
        _, vol = alpha_layered(pts, p["alpha"], p["dz_m"], with_rings=False)
        return float(vol)
    if kind == "chm":
        vol, _ = chm_volume(veg, p["cell_size_mm"] / 1000.0, p["percentile"])
        return float(vol)
    if kind == "percentile":
        if len(veg) == 0:
            return 0.0
        return float(np.percentile(veg[:, 2], p["percentile"]))
    if kind == "count":
        return float(pre.n_input if p.get("source") == "raw" else len(veg))
    raise ValueError(f"Unknown method: {kind}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.env_file and os.path.exists(pre_args.env_file):
        from dotenv import load_dotenv
        load_dotenv(pre_args.env_file, override=True)
    elif os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)

    p = argparse.ArgumentParser(
        description="Predict biomass from point cloud using pre-trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None)
    p.add_argument("--stage", default=None, choices=["Z31", "Z65", "all"],
                   help="Filter by growth stage; 'all' = all stages")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--list", dest="list_file",
                     help="List file (path + biomass per line)")
    src.add_argument("--cloud", dest="cloud_file", help="Single cloud file")
    p.add_argument("--base-dir", default=os.getenv("TPCVE_BASE_DIR", "data"))

    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower() in ("1", "true", "yes"))
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO", "2.0")))
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04)
    p.add_argument("--div2-z65", action=argparse.BooleanOptionalAction,
                   default=os.getenv("TPCVE_DIV2_Z65", "true").lower() in ("1", "true", "yes"),
                   help="Divide Z65 biomass by 2")
    p.add_argument("--output-json", default=None,
                   help="Save results as JSON")
    p.add_argument("--output-errors", default=None,
                   help="Save per-cloud predictions + errors as CSV (--list mode)")

    args = p.parse_args(argv)
    if not args.list_file and not args.cloud_file:
        p.error("Specify --cloud or --list")
    return args


# ---------------------------------------------------------------------------
# Single cloud
# ---------------------------------------------------------------------------

_DISPLAY_NAMES = {"voxel": "voxel", "alpha": "alpha", "chm": "chm",
                  "count": "count", "percentile": "height"}

_UNITS_LABEL = {"voxel": "m\u00b3", "alpha": "m\u00b3", "chm": "m\u00b3",
                "percentile": "m", "count": "pts"}


def _predict_one(veg, pre, stage) -> list[dict]:
    results = []
    for m in MODELS:
        if stage == "all":
            pass
        elif stage:
            if m["stage"] != stage:
                continue
        elif m["stage"]:
            continue
        x = _compute_feature(m, veg, pre)
        y = _apply_model(m, x)
        results.append({**m, "x_pred": x, "y_pred": y})
    stage_order = {"Z31": 0, "Z65": 1, None: 2}
    method_order = {"voxel": 0, "alpha": 1, "chm": 2, "height": 3, "count": 4}
    results.sort(key=lambda r: (stage_order.get(r.get("stage"), 99),
                                method_order.get(r["method"], 99)))
    return results


def _print_single(rel, pre, results, biomass_gt=None):
    print(f"Cloud:        {rel}")
    print(f"Points:       in={pre.n_input}  sor={pre.n_after_sor}  "
          f"veg={len(pre.vegetation)}")
    if biomass_gt is not None:
        print(f"Biomass:      {biomass_gt:.3f} g")
    print()

    has_gt = biomass_gt is not None
    if has_gt:
        h = (f"{'method':<6} | {'params':<28} | {'model':<6} | "
             f"{'x_pred':>10}     | {'y_pred (g)':>13} | "
             f"{'abs_err (g)':>14} | {'rel_err':>8} | {'train R\u00b2':>8} | {'RMSE (g)':>9} | {'RMSE%':>7}")
        sep = "-" * len(h)
    else:
        h = (f"{'method':<6} | {'params':<28} | {'model':<6} | "
             f"{'x_pred':>10}     | {'y_pred (g)':>13} | {'train R\u00b2':>8} | {'RMSE (g)':>9} | {'RMSE%':>7}")
        sep = "-" * len(h)

    print(h)
    print(sep)

    prev_stage = None
    for r in results:
        unit = _UNITS_LABEL.get(r["method"], "")
        dname = _DISPLAY_NAMES.get(r["method"], r["method"])
        x = r["x_pred"]
        y = r["y_pred"]
        label = r["label"]
        s = r.get("stage") or "combined"
        if s != prev_stage:
            print(f"--- {s} " + "-" * (len(sep) - len(s) - 5))
            prev_stage = s
        if has_gt:
            ae = y - biomass_gt
            re = ae / biomass_gt * 100 if biomass_gt > 0 else float("nan")
            print(f"{dname:<6} | {label:<28} | {r['model_type']:<6} | "
                  f"{x:>10.4f} {unit:<3} | {y:>13.2f} | {ae:>+14.2f} | "
                  f"{re:>+7.2f}% | {r['train_r2']:>8.3f} | {r['train_rmse']:>9.2f} | {r['train_rmse_pct']:>6.2f}%")
        else:
            print(f"{dname:<6} | {label:<28} | {r['model_type']:<6} | "
                  f"{x:>10.4f} {unit:<3} | {y:>13.2f} | {r['train_r2']:>8.3f} | {r['train_rmse']:>9.2f} | {r['train_rmse_pct']:>6.2f}%")

def _run_single(cloud_path: Path, args) -> tuple:
    cfg = PreprocessConfig(
        units=args.units, flip_z=args.flip_z, downsample=args.downsample,
        sor_std_ratio=args.sor_std_ratio, sor_neighbors=args.sor_neighbors,
        min_range=args.min_range, height_threshold=args.height_threshold,
        verbose=False,
    )
    pre = preprocess_cloud(str(cloud_path), cfg)
    return pre


# ---------------------------------------------------------------------------
# Batch (--list)
# ---------------------------------------------------------------------------

def _run_list(args) -> int:
    base_dir = Path(args.base_dir).resolve()
    stage = args.stage

    items = []
    with open(args.list_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rel, labels = parse_list_line(line)
            if stage and stage_from_path(rel) != stage:
                continue
            bm = float(labels["biomass"])
            if args.div2_z65 and stage_from_path(rel) == "Z65":
                bm /= 2.0
            items.append((base_dir / rel.lstrip("/\\\\"), bm, rel))

    if not items:
        print("No valid entries in --list")
        return 1

    if stage == "all":
        active = list(MODELS)
    else:
        active = [m for m in MODELS
                  if (stage and m["stage"] == stage)
                  or (not stage and not m["stage"])]

    errs = {id(m): {"m": m, "ae": [], "re": [], "preds": []}
            for m in active}

    n_total = len(items)
    for idx, (cloud_path, biomass_gt, rel) in enumerate(items):
        if not cloud_path.exists():
            print(f"  [{idx+1}/{n_total}] skip (not found): {rel}",
                  file=sys.stderr)
            continue
        try:
            cfg = PreprocessConfig(
                units=args.units, flip_z=args.flip_z,
                downsample=args.downsample,
                sor_std_ratio=args.sor_std_ratio,
                sor_neighbors=args.sor_neighbors,
                min_range=args.min_range,
                height_threshold=args.height_threshold,
                verbose=False,
            )
            pre = preprocess_cloud(str(cloud_path), cfg)
        except Exception as e:
            print(f"  [{idx+1}/{n_total}] err: {rel}: {e}", file=sys.stderr)
            continue

        veg = pre.vegetation
        for m in active:
            try:
                x = _compute_feature(m, veg, pre)
                y = _apply_model(m, x)
            except Exception:
                y = float("nan")
            mid = id(m)
            errs[mid]["preds"].append({
                "cloud": rel, "y_true": biomass_gt, "y_pred": y})
            if math.isfinite(y) and biomass_gt > 0:
                errs[mid]["ae"].append(abs(y - biomass_gt))
                errs[mid]["re"].append(abs(y - biomass_gt) / biomass_gt * 100)

        if (idx + 1) % 10 == 0 or idx == n_total - 1:
            print(f"  [{idx+1}/{n_total}] done")

    # Summary table
    print(f"\\n{'method':<6} | {'params':<28} | {'model':<6} | {'n':>5} | "
          f"{'MAE (g)':>10} | {'MRE (%)':>8} | {'RMSE (g)':>10} | {'train R\u00b2':>8} | {'train RMSE (g)':>14} | {'train RMSE%':>11}")
    print("-" * 125)

    for mid, data in errs.items():
        m = data["m"]
        ae = data["ae"]
        re = data["re"]
        n = len(ae)
        mae = sum(ae) / n if n else float("nan")
        mre = sum(re) / n if n else float("nan")
        rmse = math.sqrt(sum(a * a for a in ae) / n) if n else float("nan")
        dname = _DISPLAY_NAMES.get(m["method"], m["method"])
        print(f"{dname:<6} | {m['label']:<28} | {m['model_type']:<6} | "
              f"{n:>5} | {mae:>10.2f} | {mre:>7.2f}% | {rmse:>10.2f} | "
              f"{m['train_r2']:>8.4f} | {m['train_rmse']:>14.2f} | {m['train_rmse_pct']:>10.2f}%")

    # Save per-cloud errors
    if args.output_errors:
        rows = []
        for mid, data in errs.items():
            m = data["m"]
            for p in data["preds"]:
                rows.append({
                    "method": m["method"],
                    "params": str(m["params"]),
                    "model_type": m["model_type"],
                    "cloud": p["cloud"],
                    "y_true": p["y_true"],
                    "y_pred": p["y_pred"],
                })
        if rows:
            out_path = Path(args.output_errors)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\\nSaved per-cloud errors: {out_path}")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = _parse_args(argv)
    base_dir = Path(args.base_dir)

    if args.list_file:
        return _run_list(args)

    if args.cloud_file:
        cloud_path = Path(args.cloud_file).resolve()
        biomass_gt = None
    else:
        return 1

    try:
        rel = cloud_path.relative_to(base_dir.resolve())
    except ValueError:
        rel = cloud_path

    pre = _run_single(cloud_path, args)
    veg = pre.vegetation
    results = _predict_one(veg, pre, args.stage)
    _print_single(rel, pre, results, biomass_gt)

    if args.output_json:
        out = {
            "cloud": str(rel),
            "biomass_true": biomass_gt,
            "n_input": pre.n_input,
            "n_after_sor": pre.n_after_sor,
            "n_vegetation": int(len(veg)),
            "predictions": [
                {"method": r["method"], "params": r["params"],
                 "model_type": r["model_type"], "coefs": r["coefs"],
                 "x_pred": r["x_pred"], "y_pred": r["y_pred"],
                 "train_r2": r["train_r2"]}
                for r in results
            ],
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\\nSaved JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


METHOD_DIRS = {"voxel": "voxel", "alpha": "alpha", "chm": "chm",
               "count": "count", "percentile": "percentile"}


def _find_csvs(method: str, stage: str | None,
               results_dir: Path) -> list[Path]:
    """Найти regression CSV для метода по стадии.

    stage=None — только combined.
    stage='all' — все стадии (Z31, Z65, combined).
    stage='Z31'/'Z65' — только указанную.
    """
    d = results_dir / METHOD_DIRS[method]
    if not d.is_dir():
        return []
    if stage == "all":
        stages: list[str | None] = ["Z31", "Z65", None]
    else:
        stages = [stage]
    cands: list[Path] = []
    for s in stages:
        best = _pick_best_csv(d, s)
        if best:
            cands.append(best)
    return cands


def _pick_best_csv(d: Path, stage: str | None) -> Path | None:
    cands = []
    for p in sorted(d.glob("*.csv")):
        if p.stem.endswith("_test"):
            continue
        if stage is None:
            if "_Z31_" not in p.stem and "_Z65_" not in p.stem:
                cands.append(p)
        else:
            if f"_{stage}_" in f"_{p.stem}_":
                cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def main():
    p = argparse.ArgumentParser(
        description="Freeze regression model into standalone predict.py",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for method in ("voxel", "alpha", "chm", "percentile", "count"):
        p.add_argument(f"--{method}-csv", default=None,
                       help=f"Override: path to {method} regression CSV")
    p.add_argument("--stage", default=None,
                   choices=["Z31", "Z65", "all"],
                   help="Filter regression CSVs by growth stage; 'all' = all stages")
    p.add_argument("--results-dir", default="results",
                   help="Results root (default: results)")
    p.add_argument("--output", default="predict.py", help="Output file path")
    args = p.parse_args()

    models = []

    for method in ("voxel", "alpha", "chm", "percentile", "count"):
        explicit = getattr(args, f"{method}_csv")
        if explicit:
            csv_paths = [Path(explicit)]
        else:
            csv_paths = _find_csvs(method, args.stage,
                                   Path(args.results_dir) / "regression_csv")
        if not csv_paths:
            continue
        print(f"  {method}:")
        for csv_path in csv_paths:
            m = extract_best_row(str(csv_path))
            if m:
                models.append(m)
                s = m.get("stage") or "combined"
                print(f"    {s}: {csv_path.name} "
                      f"{m['model_type']} R\u00b2={m['train_r2']}")

    if not models:
        print("No models found", file=sys.stderr)
        return 1

    models_code = make_models_code(models)
    script = PREDICT_TEMPLATE.replace("%s", models_code, 1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")
    out_path.chmod(0o755)
    print(f"\nGenerated: {out_path.resolve()}")
    print(f"  models: {len(models)}")
    print(f"  size: {out_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
