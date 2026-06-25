# TLS Point Cloud Volume Estimation

Оценка объёма растительного покрова по данным наземного лазерного сканирования
(TLS) и предсказание биомассы пшеницы. Три объёмных метода (вокселизация,
alpha-shape, CHM) сравниваются с двумя простыми бейзлайнами (число точек,
высотный перцентиль); по каждому признаку строится регрессия
**biomass ~ признак**.

## Стек

- Python 3.11+
- NumPy, SciPy — вычисления
- Open3D — обработка облаков точек, фильтрация
- scikit-learn — регрессия (linear / power / huber)
- matplotlib, Plotly — визуализация
- alphashape — alpha-shape метод
- python-dotenv — переменные окружения

## Структура проекта

Библиотечный код собран в пакете `tpcve/`; точки входа лежат в корне репозитория.

```
batch.py, analyze.py   точки входа (CLI)
tpcve/
  cloud/    генерация/загрузка облаков, геометрия, объёмные методы, preprocess
  core/     фреймворк batch/analyze (io, аргументы, long-batch/analyze)
  methods/  реестр + плагины признаков (voxel/alpha/chm/count/percentile)
tools/      утилиты (autoname, regression, optimize_r2)
scripts/    утилиты и демо (inspect_cloud, predict_biomass, visualize_methods)
experiments/ разовые прогоны (occlusion, voxel_size_sweep, downsample-сравнения)
data/       наборы данных с per-dataset settings.env
datasets/   субмодуль с датасетом Yanco TC 2019
```

## Установка

Варианты клонирования:

```bash
# только код (легко, ~2 МБ)
git clone https://github.com/Konabur/tpcve.git
cd tpcve
```

код + датасет сразу
```bash
git clone --recurse-submodules https://github.com/Konabur/tpcve.git
cd tpcve
```


### uv (рекомендуется)

```bash
uv sync
```

### venv + pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -e .
```

Для работы с LAS/LAZ:
```bash
pip install laspy
```

## Основной сценарий: batch + analyze

Конвейер из двух этапов; метод выбирается через `--method`:

- **`batch.py`** — считает признак(и) по набору облаков и пишет volume-CSV.
  При `--analyze` (включён по умолчанию) сразу запускает регрессию.
- **`analyze.py`** — регрессия biomass ~ признак по volume-CSV (без `--input-csv`
  берёт последний по времени).

### Методы

**Объёмные методы** — оценивают объём растительности (основной предмет работы):

| `--method` | признак | метод-флаги | подпапка `results/` |
|---|---|---|---|
| `voxel` | объём вокселизации | `--voxel-sizes` | `voxel/` |
| `alpha` | alpha-shape объём (3D/послойный) | `--voxel-sizes --alphas --layer-dz --with-random` | `alpha/` |
| `chm` | объём по сетке высот | `--cell-sizes --percentiles` | `chm/` |

**Бейзлайны** — простые скаляры без оценки объёма, для сравнения:

| `--method` | признак | метод-флаги | подпапка `results/` |
|---|---|---|---|
| `count` | число точек (raw/pre) | — | `count/` |
| `percentile` | перцентиль высоты | `--percentiles` | `percentile/` |

Метод-специфичные флаги: `batch.py --method <name> --help`.

### Входные данные: --list / --base-dir / --input-dir

Облака подаются двумя способами:

**1. `--list <файл>` + `--base-dir <папка>` — c метками биомассы (для регрессии).**

В list-файле каждая строка — облако:

```
<относительный-путь> <biomass> [col3 col4 col5]
```

Путь может содержать пробелы — он отделяется от меток по расширению (`.pcd`,
`.las`, …). Необязательные `col3..col5` игнорируются. `--base-dir` — корень, от
которого разрешается относительный путь из list-файла.

```
# train_list.txt
/20190828/Tony e-w_20190828_001/1-5-1-b.pcd 380.600000 3 1 3
```

→ `--base-dir <папка> --list train_list.txt` → загрузится
`<папка>/20190828/Tony e-w_20190828_001/1-5-1-b.pcd`.

**2. `--input-dir <папка>` — без меток (только подсчёт признаков).**

Рекурсивный поиск всех `.pcd` в папке. Биомасса не указывается — регрессия
невозможна, нужно гасить `--no-analyze`.

### Набор данных Yanco TC 2019

Работа ведётся на открытом наборе данных:

> Estavillo, Gonzalo; Anthony, Condon; Pan, Liyuan; Bull, Geoff; & Coe, Robert
> (2021): *Biomass and LiDAR data from wheat and triticale plots grown at Yanco
> (NSW) in 2019 to improve prediction of digital biomass.* v2. CSIRO. Data
> Collection. <https://doi.org/10.25919/xv6v-6h56>

Это TLS-сканы делянок пшеницы и тритикале (Yanco, NSW, 2019) с измеренной
наземной биомассой — целевой переменной регрессии.

**Подключение:**

Если при клонировании не использовали `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Структура внутри субмодуля:

```
datasets/yanco-2019-wheat-pcd/
  data/
    Yanco_TC_2019_HI-pcd/     # папки 20190828/ и 20191002/ с .pcd
    train_list.txt             # --list для train
    test_list.txt              # --list для test
  metadata/
```

Примеры использования (см. описание `--list`/`--base-dir`/`--input-dir` выше):

```bash
# регрессия: --list + --base-dir
python batch.py --method voxel \
  --base-dir datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd \
  --list datasets/yanco-2019-wheat-pcd/data/train_list.txt

# без меток: --input-dir
python batch.py --method count \
  --input-dir datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd \
  --no-analyze
```

Или через `.env` (см. `.env.example`).

**Стадии роста.** Списки покрывают две даты съёмки, соответствующие стадиям
развития по шкале Zadoks (см. `STAGE_TOKENS` в `tpcve/core/io.py`):

| Стадия | Дата съёмки | Папка в наборе |
|---|---|---|
| `Z31` | 2019-08-28 | `20190828/` |
| `Z65` | 2019-10-02 | `20191002/` |

`test_list.txt` / `train_list.txt` — это разбиение делянок на test/train,
включающее обе стадии; стадия каждой делянки определяется по дате в пути.

**Фильтр по стадии (`--stage`).** Удобный флаг под этот набор: оставляет только
облака нужной стадии — и для `--list`, и для `--input-dir`. Стадия определяется
по дате в пути (`STAGE_TOKENS` в `tpcve/core/io.py`), отдельная колонка не нужна.
Можно задать через `TPCVE_STAGE`. Когда флаг указан, имя выходного CSV получает
токен стадии сразу после имени источника (`train_list_Z31_v7...`).

```bash
# регрессия по обучающему списку набора (обе стадии)
python batch.py --method voxel \
  --base-dir datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd \
  --list datasets/yanco-2019-wheat-pcd/data/train_list.txt \
  --voxel-sizes 6,7,8,10

# только стадия Z65 (съёмка 2019-10-02)
python batch.py --method voxel \
  --base-dir datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd \
  --list datasets/yanco-2019-wheat-pcd/data/train_list.txt \
  --stage Z65 --voxel-sizes 6,7,8,10

# прогон по всей папке датасета (без меток биомассы)
python batch.py --method count \
  --input-dir datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd \
  --no-analyze
```

> Поддержка стадий (`--stage`, `STAGE_TOKENS`) и формат `--list` заточены под
> структуру набора Yanco TC 2019 для удобства воспроизведения. Для других
> наборов стадии можно не использовать (флаг опционален), а список делянок —
> оформить в том же формате `<путь> <biomass> <col3> <col4> <col5>`.

### Примеры

```bash
# voxel: объём по нескольким размерам вокселя + регрессия
python batch.py --method voxel --list data/train.txt --voxel-sizes 6,7,8,10

# несколько методов за один прогон
python batch.py --method voxel,chm --list data/train.txt --cell-sizes 20,50 --percentiles 95

# alpha-shape (послойно)
python batch.py --method alpha --list data/train.txt --alphas 10,20 --layer-dz 20

# только batch без регрессии
python batch.py --method count --input-dir data/Yanco-1-1-1-b --no-analyze

# регрессия по готовому volume-CSV (или берёт последний по времени)
python analyze.py --method voxel
python analyze.py --method chm --input-csv results/volume_csv/chm/x.csv
```

### Результаты

- `results/volume_csv/<m>/` — признаки из `batch.py`
- `results/regression_csv/<m>/` — таблицы регрессии (per-model: linear / power / huber)
- `results/regression_plots/<m>/<stem>/` — графики фитов

Имя выходного CSV строится автоматически из аргументов, если `--output-csv` не задан.

## Переменные окружения

Все аргументы можно задать через переменные с префиксом `TPCVE_`. `.env` файл
загружается автоматически (см. `.env.example`); CLI-аргументы имеют приоритет.

```bash
TPCVE_UNITS=mm
TPCVE_BASE_DIR=datasets/yanco-2019-wheat-pcd/data/Yanco_TC_2019_HI-pcd
TPCVE_STAGE=Z65
TPCVE_FLIP_Z=false
TPCVE_DOWNSAMPLE=0
TPCVE_SOR_STD_RATIO=2.0
TPCVE_MIN_RANGE=0
```

## Поддерживаемые форматы

- `.npz` — синтетические данные (с ground truth)
- `.las`, `.laz` — стандарт индустрии
- `.pcd`, `.ply`, `.xyz`, `.pts` — Open3D форматы
- `.db3` — ROS 2 bag файлы (rosbags)

Для реальных облаков:
- Автоопределение единиц измерения (м/см/мм)
