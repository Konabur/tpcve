"""Батч/анализ-фреймворк, в который втыкаются плагины из `methods/`.

Публичный API пакета: методы делают `import core` и зовут `core.X`;
внешние скрипты берут конкретные имена (напр. `from core.io import
pick_median_biomass`). Сам код разнесён по фокусным модулям:

  - core.io           — сбор/парсинг входов (--list/--input-dir), выбор облака
  - core.args         — CLI/preprocess аргументы, autoname, analyze-парсер
  - core.long_batch   — цикл long-batch + чейнинг analyze
  - core.long_analyze — регрессия biomass ~ x по группам

`__init__` — только реэкспорт (фасад), реализаций здесь нет.
"""
from __future__ import annotations

from core.args import (add_common_batch_args, autoname_extra_from_args,
                       build_analyze_parser, load_env_from_argv,
                       preprocess_config_from_args)
from core.io import (LABEL_COLS, STAGE_TOKENS, BatchCfg, InputItem, collect_for,
                     collect_inputs, load_done_keys, parse_list_line,
                     pick_median_biomass, stage_from_path)
from core.long_analyze import run_long_analyze
from core.long_batch import (LongBatchSpec, chain_analyze, run_batch_train_test,
                             run_long_batch, simple_error_rows, standard_main)

__all__ = [
    # io
    "LABEL_COLS", "STAGE_TOKENS", "BatchCfg", "InputItem", "collect_for",
    "collect_inputs", "load_done_keys", "parse_list_line",
    "pick_median_biomass", "stage_from_path",
    # args
    "add_common_batch_args", "autoname_extra_from_args", "build_analyze_parser",
    "load_env_from_argv", "preprocess_config_from_args",
    # long_batch
    "LongBatchSpec", "chain_analyze", "run_batch_train_test", "run_long_batch",
    "simple_error_rows", "standard_main",
    # long_analyze
    "run_long_analyze",
]
