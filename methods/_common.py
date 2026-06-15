"""Общий код batch/analyze: io, цикл long-batch, общий long-analyze."""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv
from tqdm import tqdm

from cloud_pipeline import PreprocessConfig, preprocess_cloud

LABEL_COLS = ["biomass", "col3", "col4", "col5"]


@dataclass
class InputItem:
    rel_path: str
    full_path: Path
    labels: dict


@dataclass
class BatchCfg:
    """Минимальный конфиг для collect_inputs/collect_for (заменяет type('Cfg', …))."""
    list_file: str | None
    input_dir: str | None
    base_dir: Path
    limit: int | None = None


def parse_list_line(line: str) -> tuple[str, dict]:
    """`<path> <biomass> <c3> <c4> <c5>` — путь может содержать пробелы."""
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Ожидалось >=5 токенов, получено {len(parts)}: {line!r}")
    *path_parts, biomass, c3, c4, c5 = parts
    rel_path = " ".join(path_parts)
    return rel_path, {"biomass": biomass, "col3": c3, "col4": c4, "col5": c5}


def collect_inputs(cfg, *, list_file: str | None = None) -> list[InputItem]:
    """list_file override позволяет переиспользовать конфиг для test-прохода."""
    items: list[InputItem] = []
    src_list = list_file if list_file is not None else cfg.list_file
    if src_list:
        with open(src_list, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                rel, labels = parse_list_line(line)
                full = cfg.base_dir / rel.lstrip("/\\")
                items.append(InputItem(rel, full, labels))
    elif cfg.input_dir and list_file is None:
        root = Path(cfg.input_dir)
        for f in sorted(root.rglob("*.pcd")):
            rel = str(f.relative_to(root))
            items.append(InputItem(rel, f, {k: "" for k in LABEL_COLS}))
    else:
        raise ValueError("Нужен --list или --input-dir")
    if cfg.limit:
        items = items[: cfg.limit]
    return items


def collect_for(cfg, list_file: str | None) -> list[InputItem]:
    """Обёртка для test-прохода: строит временный cfg-объект."""
    return collect_inputs(BatchCfg(
        list_file=list_file if list_file is not None else cfg.list_file,
        input_dir=None if list_file is not None else cfg.input_dir,
        base_dir=cfg.base_dir, limit=cfg.limit,
    ))


def load_done_files(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {row["file"] for row in csv.DictReader(f) if row.get("file")}


def load_done_keys(csv_path: Path, key_fn: Callable[[dict], str]) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {key_fn(row) for row in csv.DictReader(f) if row.get("file")}


# ---------------------------------------------------------------------------
# Выбор облака из списка (по стадии роста / медиане биомассы)

# Стадия роста определяется по подстроке в пути облака (дата съёмки).
STAGE_TOKENS = {"Z31": "0828", "Z65": "1002"}


def stage_from_path(path: str) -> str | None:
    for stage, tok in STAGE_TOKENS.items():
        if tok in path:
            return stage
    return None


def pick_median_biomass(list_path: str, base_dir: Path,
                        stage: str | None = None
                        ) -> tuple[Path, float, str | None]:
    """Из --list-файла выбрать облако с медианной биомассой.

    Возвращает (полный путь, биомасса, стадия). При stage != None берутся только
    облака этой стадии. Пустой результат → SystemExit.
    """
    rows: list[tuple[Path, float, str | None]] = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rel, labels = parse_list_line(line)
                bm = float(labels["biomass"])
            except (ValueError, KeyError):
                continue
            st = stage_from_path(rel)
            if stage is not None and st != stage:
                continue
            rows.append((base_dir / rel.lstrip("/\\"), bm, st))
    if not rows:
        msg = f"В {list_path} не нашлось валидных строк с биомассой"
        if stage is not None:
            msg += f" для стадии {stage} (подстрока '{STAGE_TOKENS[stage]}')"
        raise SystemExit(msg)
    rows.sort(key=lambda r: r[1])
    return rows[len(rows) // 2]


# ---------------------------------------------------------------------------
# Общие CLI/preprocess хелперы (имена и дефолты как в исходных batch-скриптах)

def load_env_from_argv(argv: Iterable[str] | None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.env_file and os.path.exists(pre_args.env_file):
        load_dotenv(pre_args.env_file, override=True)
    elif os.path.exists(".env"):
        load_dotenv(".env", override=True)


def add_common_batch_args(p: argparse.ArgumentParser, *,
                          sor_default: float = 2.0) -> None:
    """Общие для всех batch-методов флаги (имена и дефолты как сейчас)."""
    p.add_argument("--env-file", default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--list", dest="list_file")
    src.add_argument("--input-dir")
    p.add_argument("--list-test", default=None)
    p.add_argument("--base-dir", default=os.getenv("TPCVE_BASE_DIR", "data"))
    p.add_argument("--output-csv", default=None)
    p.add_argument("--units", default=os.getenv("TPCVE_UNITS", "auto"),
                   choices=["auto", "m", "cm", "mm"])
    p.add_argument("--flip-z", action="store_true",
                   default=os.getenv("TPCVE_FLIP_Z", "").lower()
                   in ("1", "true", "yes"))
    p.add_argument("--downsample", type=float,
                   default=float(os.getenv("TPCVE_DOWNSAMPLE", "0") or 0))
    p.add_argument("--sor-neighbors", type=int, default=20)
    p.add_argument("--sor-std-ratio", type=float,
                   default=float(os.getenv("TPCVE_SOR_STD_RATIO",
                                           str(sor_default))))
    p.add_argument("--min-range", type=float,
                   default=float(os.getenv("TPCVE_MIN_RANGE", "0") or 0))
    p.add_argument("--height-threshold", type=float, default=0.04)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True)


def preprocess_config_from_args(a) -> PreprocessConfig:
    return PreprocessConfig(
        units=a.units, flip_z=a.flip_z, downsample=a.downsample,
        sor_std_ratio=a.sor_std_ratio, sor_neighbors=a.sor_neighbors,
        min_range=a.min_range, height_threshold=a.height_threshold,
        verbose=a.verbose,
    )


def simple_error_rows(item: InputItem, msg: str) -> list[dict]:
    """Стандартная строка ошибки: одна запись file+labels+error.

    Используют voxel/percentile/chm. count переопределяет (две строки на source).
    """
    return [{"file": item.rel_path, **item.labels, "error": msg}]


def autoname_extra_from_args(a, *, sor_default: float = 2.0) -> dict:
    extra: dict = {}
    if abs(a.sor_std_ratio - sor_default) > 1e-9:
        extra["sor"] = a.sor_std_ratio
    if a.flip_z:
        extra["flipz"] = True
    if a.downsample > 0:
        extra["ds"] = a.downsample
    if a.min_range > 0:
        extra["r"] = a.min_range
    return extra


# ---------------------------------------------------------------------------
# Общий цикл long-batch (chm/count/percentile/voxel)

@dataclass
class LongBatchSpec:
    columns: list[str]
    row_key: Callable[[dict], str]
    error_rows: Callable[[InputItem, str], list[dict]]
    compute_rows: Callable[[InputItem, object, set], list[dict]]


def run_long_batch(spec: LongBatchSpec, *, items: list[InputItem],
                   csv_path: Path, resume: bool, preprocess: PreprocessConfig,
                   label: str = "train") -> int:
    print(f"[{label}] файлов на входе: {len(items)} -> {csv_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(csv_path, spec.row_key) if resume else set()
    mode = "a" if (resume and csv_path.exists()) else "w"
    t0 = time.time()
    n_done = n_err = 0
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=spec.columns,
                                extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            f.flush()
        bar = tqdm(items, unit="cloud", dynamic_ncols=True)
        for item in bar:
            bar.set_postfix_str(item.rel_path[-40:], refresh=False)
            if not item.full_path.exists():
                for r in spec.error_rows(item, f"not found: {item.full_path}"):
                    writer.writerow(r)
                f.flush()
                n_err += 1
                continue
            try:
                res = preprocess_cloud(str(item.full_path), preprocess)
            except Exception as e:
                for r in spec.error_rows(item, f"{type(e).__name__}: {e}"):
                    writer.writerow(r)
                f.flush()
                n_err += 1
                continue
            rows = spec.compute_rows(item, res, done_keys)
            for r in rows:
                writer.writerow(r)
                n_done += 1
            f.flush()
    print(f"\nГотово за {time.time() - t0:.1f}s. Строк добавлено: {n_done} "
          f"(ошибок файлов: {n_err}). CSV: {csv_path}")
    return 0


def run_batch_train_test(spec: LongBatchSpec, a, output_csv: Path) -> Path:
    """Общий хвост batch: train-проход + (опц.) test-проход. Возвращает output_csv.

    `a` — распарсенные общие batch-аргументы (add_common_batch_args). Sweep-парсинг
    и автоназвание остаются в методе; сюда приходит уже готовый spec и output_csv.
    """
    cfg = BatchCfg(a.list_file, a.input_dir, Path(a.base_dir), a.limit)
    pre = preprocess_config_from_args(a)
    run_long_batch(spec, items=collect_for(cfg, None), csv_path=output_csv,
                   resume=a.resume, preprocess=pre, label="train")
    if a.list_test:
        test_csv = output_csv.with_name(output_csv.stem + "_test"
                                        + output_csv.suffix)
        run_long_batch(spec, items=collect_for(cfg, a.list_test),
                       csv_path=test_csv, resume=a.resume, preprocess=pre,
                       label="test")
    return output_csv


def chain_analyze(mod, output_csv: Path, argv: Iterable[str] | None) -> None:
    """Запустить analyze метода сразу после batch, если включён --analyze.

    Аргументы для analyze выводятся из общих batch-флагов: путь к train-CSV,
    --test-csv (если был --list-test), --plots-dir (если --plots), --top. Вызывается
    и при прямом запуске метода (python -m methods.<name>), и диспетчером batch.py."""
    p = argparse.ArgumentParser(add_help=False)
    add_common_batch_args(p)
    mod.add_batch_args(p)
    a, _ = p.parse_known_args(argv)
    if not a.analyze:
        return
    an = [str(output_csv)]
    if a.list_test:
        an += ["--test-csv", str(output_csv.with_name(
            output_csv.stem + "_test" + output_csv.suffix))]
    if a.plots:
        an.append("--plots-dir")
    if a.top is not None:
        an += ["--top", str(a.top)]
    print(f"\n>>> analyze {getattr(mod, 'NAME', '?')}: {' '.join(an)}")
    mod.run_analyze(an)


def build_analyze_parser(description: str | None = None
                         ) -> argparse.ArgumentParser:
    """Стандартный парсер analyze (csv + общие флаги). Метод добавляет своё
    через add_analyze_args, затем parse_known_args."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("csv")
    p.add_argument("--test-csv", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--plots-dir", nargs="?", const="__auto__", default=None)
    p.add_argument("--target", default="biomass")
    p.add_argument("--top", type=int, default=None)
    return p


def standard_main(module, argv=None) -> int:
    """Единый main: batch метода, затем chain_analyze. Для всех методов одинаков."""
    csv_path = module.run_batch(argv)
    chain_analyze(module, csv_path, argv)
    return 0


# ---------------------------------------------------------------------------
# Общий long-analyze: регрессия biomass ~ x по группам строк CSV. Единая схема
# выходного regression-CSV: [method, *group_cols, x_col, <статистики fit_all>].

def _as_tuple(keys) -> tuple:
    return keys if isinstance(keys, tuple) else (keys,)


def run_long_analyze(args, *, value_cols: list[str], group_cols: list[str],
                     label_fn: Callable[[dict, str], str],
                     subfolder: str,
                     prep_df: Callable | None = None) -> int:
    """Единый analyze для всех методов.

    - `value_cols`: x-колонки (обычно одна; alpha может дать [V_voxel, V_random]).
    - `group_cols`: колонки группировки (свои у каждого метода).
    - `label_fn(meta, vc) -> str`: человекочитаемая метка группы (колонка `method`).
    - `prep_df(df)`: опц. in-place подготовка (напр. alpha: fillna layer_dz_mm).
    """
    import numpy as np
    import pandas as pd

    from tools.autoname import default_path
    from tools.regression import (bootstrap_test_r2_ci, compute_metrics,
                                   fit_all, flatten_for_csv, plot_fits)

    df = pd.read_csv(args.csv)
    if prep_df is not None:
        prep_df(df)
    if args.target not in df.columns:
        raise SystemExit(f"Нет колонки {args.target!r} в CSV")
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    vcs = [c for c in value_cols if c in df.columns]
    if not vcs:
        raise SystemExit(f"Нет ни одной value-колонки {value_cols} в CSV")

    df_test = None
    test_groups: dict[tuple, "pd.DataFrame"] = {}
    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        if prep_df is not None:
            prep_df(df_test)
        df_test[args.target] = pd.to_numeric(df_test[args.target], errors="coerce")
        for keys, grp in df_test.groupby(group_cols, dropna=False):
            test_groups[_as_tuple(keys)] = grp

    rows = []
    fit_cache: dict[tuple, dict] = {}
    meta_cache: dict[tuple, dict] = {}
    for keys, grp in df.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, _as_tuple(keys)))
        for vc in vcs:
            s = pd.to_numeric(grp[vc], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            if mask.sum() < 3:
                continue
            result = fit_all(s[mask].to_numpy(), y[mask].to_numpy())
            if result is None:
                continue
            label = label_fn(meta, vc)
            key = (_as_tuple(keys), vc)
            fit_cache[key] = result
            meta_cache[key] = meta
            rows.append({"method": label, **meta, "x_col": vc,
                         **flatten_for_csv(result)})

    if not rows:
        raise SystemExit("Нет валидных данных для регрессии")

    res_df = pd.DataFrame(rows)
    res_df["_max_r2"] = res_df[["linear_r2", "power_r2", "huber_r2"]].max(axis=1)
    res_df = (res_df.sort_values("_max_r2", ascending=False)
              .drop(columns="_max_r2").reset_index(drop=True))
    if args.top:
        res_df = res_df.head(args.top)

    print(f"\nЦель: {args.target} ~ x   (групп: {len(res_df)})")
    print("=" * 140)
    show_cols = ["method", "x_col", "best_model",
                 "linear_r2", "linear_rmse_pct",
                 "power_r2", "power_rmse_pct", "power_b",
                 "huber_r2", "huber_rmse_pct"]
    print(res_df[show_cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4g}"))
    print("=" * 140)

    stem = Path(args.csv).stem
    if args.output:
        out_csv = Path(args.output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = default_path("regression_csv", stem + "_regression",
                               subfolder=subfolder)
    res_df.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")

    test_xy: dict[tuple, tuple] = {}
    if df_test is not None:
        test_rows = []
        for r in res_df.itertuples(index=False):
            meta = {c: getattr(r, c) for c in group_cols}
            vc = r.x_col
            key = (tuple(meta[c] for c in group_cols), vc)
            result = fit_cache.get(key)
            grp = test_groups.get(tuple(meta[c] for c in group_cols))
            if result is None or grp is None or vc not in grp.columns:
                continue
            s = pd.to_numeric(grp[vc], errors="coerce")
            y = pd.to_numeric(grp[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            if mask.sum() == 0:
                continue
            xv, yv = s[mask].to_numpy(), y[mask].to_numpy()
            test_xy[key] = (xv, yv)
            row = {"method": r.method, **meta, "x_col": vc,
                   "n_test": int(mask.sum())}
            for name, f in result["all"].items():
                m = compute_metrics(yv, f["predict"](xv))
                row[f"{name}_test_r2"] = m["r2"]
                row[f"{name}_test_rmse"] = m["rmse"]
                row[f"{name}_test_rmse_pct"] = m["rmse_pct"]
                row[f"{name}_test_bias"] = m["bias"]
                ci_lo, ci_hi = bootstrap_test_r2_ci(f["predict"], xv, yv)
                row[f"{name}_test_r2_ci_lo"] = ci_lo
                row[f"{name}_test_r2_ci_hi"] = ci_hi
            test_rows.append(row)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            print(f"\nTest (групп с test: {len(test_df)}):")
            print("=" * 140)
            disp = test_df[["method", "x_col", "n_test"]].copy()
            for name in ("linear", "power", "huber"):
                disp[f"{name}_R2[CI95]"] = [
                    f"{a:.3f}[{b:.2f},{c:.2f}]" for a, b, c in zip(
                        test_df[f"{name}_test_r2"],
                        test_df[f"{name}_test_r2_ci_lo"],
                        test_df[f"{name}_test_r2_ci_hi"])]
                disp[f"{name}_RMSE%"] = test_df[f"{name}_test_rmse_pct"].round(1)
            print(disp.to_string(index=False))
            print("=" * 140)
            test_out = out_csv.with_name(out_csv.stem + "_test" + out_csv.suffix)
            test_df.to_csv(test_out, index=False)
            print(f"Test regression: {test_out}")

    if args.plots_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = (default_path("regression_plots", stem, ext="", subfolder=subfolder)
               if args.plots_dir == "__auto__" else Path(args.plots_dir))
        out.mkdir(parents=True, exist_ok=True)
        for r in res_df.itertuples(index=False):
            meta = {c: getattr(r, c) for c in group_cols}
            vc = r.x_col
            key = (tuple(meta[c] for c in group_cols), vc)
            result = fit_cache.get(key)
            if result is None:
                continue
            sub = df
            for c in group_cols:
                sub = sub[sub[c] == meta[c]]
            s = pd.to_numeric(sub[vc], errors="coerce")
            y = pd.to_numeric(sub[args.target], errors="coerce")
            mask = s.notna() & y.notna() & (s > 0)
            x, yv = s[mask].to_numpy(), y[mask].to_numpy()
            xv_test, yv_test = test_xy.get(key, (None, None))
            fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
            plot_fits(ax, x, yv, result, xlabel=str(vc), ylabel=args.target,
                      title=f"{args.target} ~ {r.method} [{vc}]",
                      x_test=xv_test, y_test=yv_test)
            best = result["best"]
            y_pred_tr = best["predict"](x)
            ax_r.scatter(y_pred_tr, yv - y_pred_tr, s=20, alpha=0.6,
                         color="#444", label=f"train (n={len(x)})")
            if xv_test is not None and len(xv_test) > 0:
                y_pred_te = best["predict"](xv_test)
                ax_r.scatter(y_pred_te, yv_test - y_pred_te, s=28, alpha=0.85,
                             color="#d62728", marker="^",
                             label=f"test (n={len(xv_test)})")
            ax_r.axhline(0, color="k", lw=0.8, ls="--")
            ax_r.set_xlabel(f"fitted {args.target}")
            ax_r.set_ylabel("residual (y − ŷ)")
            ax_r.set_title(f"Residuals [{result['best_model']}]")
            ax_r.legend(loc="best", fontsize=8)
            ax_r.grid(alpha=0.3)
            fig.tight_layout()
            fname = f"{r.method}__{vc}.png".replace("/", "_")
            fig.savefig(out / fname, dpi=130)
            plt.close(fig)
        print(f"Графики: {out}/")

    return 0
