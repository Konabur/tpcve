"""CLI/preprocess аргументы: общие batch-флаги, .env, конфиг preprocess, autoname."""
from __future__ import annotations

import argparse
import os
from typing import Iterable

from dotenv import load_dotenv

from tpcve.cloud.cloud_pipeline import PreprocessConfig
from tpcve.core.io import STAGE_TOKENS


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
    src = p.add_mutually_exclusive_group()
    src.add_argument("--list", dest="list_file",
                     default=os.getenv("TPCVE_LIST"))
    src.add_argument("--input-dir")
    p.add_argument("--list-test",
                   default=os.getenv("TPCVE_LIST_TEST"))
    p.add_argument("--stage", default=os.getenv("TPCVE_STAGE") or None,
                   choices=list(STAGE_TOKENS),
                   help="Брать только облака этой стадии роста (по дате в пути)")
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
    p.add_argument("--height-threshold", type=float,
                   default=float(os.getenv("TPCVE_HEIGHT_THRESHOLD", "0.04")),
                   help="Порог высоты для отделения земли от растительности (м)")
    p.add_argument("--preprocess-cache", default=None,
                   help="Dir to cache preprocessed clouds (speeds up multi-method runs)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--div2-z65", action=argparse.BooleanOptionalAction,
                   default=os.getenv("TPCVE_DIV2_Z65", "true").lower()
                   in ("1", "true", "yes"),
                   help="Делить биомассу Z65 на 2 (для приведения к Z31)")
    p.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--plots", action=argparse.BooleanOptionalAction,
                   default=True)


def preprocess_config_from_args(a) -> PreprocessConfig:
    return PreprocessConfig(
        units=a.units, flip_z=a.flip_z, downsample=a.downsample,
        sor_std_ratio=a.sor_std_ratio, sor_neighbors=a.sor_neighbors,
        min_range=a.min_range, height_threshold=a.height_threshold,
        verbose=a.verbose, cache_dir=getattr(a, "preprocess_cache", None),
    )


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
    if getattr(a, "stage", None):
        extra["stage"] = a.stage  # рендерится сразу после имени, до параметров
    return extra


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
