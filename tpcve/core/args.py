"""CLI/preprocess аргументы: общие batch-флаги, .env, конфиг preprocess, autoname."""
from __future__ import annotations

import argparse
import os
from typing import Iterable

from dotenv import load_dotenv

from tpcve.cloud.cloud_pipeline import PreprocessConfig


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
