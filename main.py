"""
Эксперимент: оценка объёма растительного покрова на синтетических данных
наземного лазерного сканирования (TLS).

Версия 2:
- Реалистичная модель шума (больше выбросов, мультипереотражения)
- Моделирование окклюзии (характерная проблема TLS)
- 3D интерактивная визуализация (Plotly)
- Улучшенные графики сравнения методов
"""

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull
import json
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tpcve.cloud.generate_cloud import generate_full_cloud, load_cloud, save_cloud, load_real_cloud
import plotly.graph_objects as go

# Предварительный парсинг для получения --env-file
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--env-file', type=str, default=None)
pre_args, _ = pre_parser.parse_known_args()

# Загрузка env-файла: --env-file → .env → settings.env
if pre_args.env_file and os.path.exists(pre_args.env_file):
    load_dotenv(pre_args.env_file, override=True)
elif os.path.exists('.env'):
    load_dotenv('.env', override=True)
elif os.path.exists('settings.env'):
    load_dotenv('settings.env', override=True)

# ============================================================
# 1. ЗАГРУЗКА ИЛИ ГЕНЕРАЦИЯ ОБЛАКА ТОЧЕК
# ============================================================

parser = argparse.ArgumentParser(description='Анализ объёма растительного покрова TLS')
parser.add_argument('--env-file', type=str, default=None,
                   help='Путь к .env файлу с настройками (по умолчанию: .env или settings.env)')
parser.add_argument('--cloud', type=str, default=os.getenv('TPCVE_CLOUD'),
                   help='Путь к облаку точек (.npz, .las, .laz, .pcd, .ply, .xyz, .pts)')
parser.add_argument('--save-cloud', type=str, default=os.getenv('TPCVE_SAVE_CLOUD'),
                   help='Сохранить сгенерированное облако в .npz')
parser.add_argument('--gt-volume', type=float,
                   default=float(os.getenv('TPCVE_GT_VOLUME', '0')) or None,
                   help='Ground truth объём (м³) для реальных облаков')
parser.add_argument('--units', type=str, default=os.getenv('TPCVE_UNITS', 'auto'),
                   choices=['auto', 'm', 'cm', 'mm'],
                   help='Единицы измерения координат (auto=автоопределение, m=метры, cm=сантиметры, mm=миллиметры)')
parser.add_argument('--output-dir', type=str, default=os.getenv('TPCVE_OUTPUT_DIR', 'results'),
                   help='Папка для сохранения результатов')
parser.add_argument('--default-voxel-size', type=float,
                   default=float(os.getenv('TPCVE_DEFAULT_VOXEL_SIZE', '0.007')),
                   help='Дефолтный размер вокселя (м) для случая без GT')
parser.add_argument('--skip-hull-methods', action='store_true',
                   default=os.getenv('TPCVE_SKIP_HULL_METHODS', '').lower() in ('1', 'true', 'yes'),
                   help='Пропустить Convex Hull и Alpha Shape (ускорение)')
parser.add_argument('--flip-z', action='store_true',
                   default=os.getenv('TPCVE_FLIP_Z', '').lower() in ('1', 'true', 'yes'),
                   help='Инвертировать ось Z (если земля сверху, сенсор смотрит вниз)')
parser.add_argument('--downsample', type=float,
                   default=float(os.getenv('TPCVE_DOWNSAMPLE', '0') or 0),
                   help='Voxel downsample перед SOR (м), 0 = без даунсемплинга. Рекомендуется 0.003-0.005 для плотных реальных облаков')
parser.add_argument('--sor-std-ratio', type=float,
                   default=float(os.getenv('TPCVE_SOR_STD_RATIO', '1.5')),
                   help='SOR std_ratio (1.5=агрессивно для синтетики, 2.0-3.0 для реальных лидаров)')
parser.add_argument('--min-range', type=float,
                   default=float(os.getenv('TPCVE_MIN_RANGE', '0') or 0),
                   help='Удалить точки ближе этого расстояния (м) от сенсора (0 = без фильтра)')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_VOXEL_SIZE_NO_GT = args.default_voxel_size

print("=" * 60)
print("ЗАГРУЗКА ОБЛАКА ТОЧЕК")
print("=" * 60)

if args.cloud:
    ext = Path(args.cloud).suffix.lower()
    print(f"\nЗагрузка из {args.cloud}...")

    if ext == '.npz':
        data = load_cloud(args.cloud)
    else:
        data = load_real_cloud(args.cloud, units=args.units)
        print("  (реальное облако, ground truth недоступен)")
else:
    print("\nГенерация нового облака...")
    data = generate_full_cloud()
    if args.save_cloud:
        save_cloud(data, args.save_cloud)
        print(f"Сохранено в {args.save_cloud}")

all_pts_noisy = data['all_pts_noisy']
ground_pts = data['ground_pts']
vegetation_pts = data['vegetation_pts']
plant_params = data['plant_params']
scanner_pos = data['scanner_pos']
total_gt_volume = data['total_gt_volume']

# Применить переданный GT если указан
if args.gt_volume is not None:
    total_gt_volume = args.gt_volume
    print(f"  Используется переданный GT объём: {total_gt_volume:.6f} м³")

# Нормализация Z для реальных облаков: земля должна быть в Z=0, растительность выше.
# --flip-z используется если сенсор смотрел вниз (земля оказалась сверху).
# После флипа (или без него) сдвигаем min(Z) к 0.
if args.cloud and Path(args.cloud).suffix.lower() != '.npz':
    all_pts_noisy = all_pts_noisy.copy()
    if args.flip_z:
        all_pts_noisy[:, 2] = -all_pts_noisy[:, 2]
    all_pts_noisy[:, 2] -= all_pts_noisy[:, 2].min()
    print(f"  Z нормализован{' (flip)' if args.flip_z else ''}: "
          f"{all_pts_noisy[:,2].min():.3f}..{all_pts_noisy[:,2].max():.3f} м")

# Флаг наличия ground truth
has_gt = total_gt_volume > 0

# Для совместимости с сохранением результатов:
# в реальных облаках/старых NPZ эти поля могут отсутствовать
all_pts_clean = data.get('all_pts_clean', all_pts_noisy)
all_pts_occluded = data.get('all_pts_occluded', all_pts_noisy)
n_occluded = max(0, len(all_pts_clean) - len(all_pts_occluded))

# Сохранить облако в results/
save_cloud(data, f'{OUTPUT_DIR}/cloud.npz')
print(f"Облако сохранено в {OUTPUT_DIR}/cloud.npz")

print(f"\n  Точек земли (исходно): {len(ground_pts)}")
print(f"  Точек растительности (исходно): {len(vegetation_pts)}")
print(f"  Точек итого (с шумом): {len(all_pts_noisy)}")
print(f"  Растений: {len(plant_params)}")
if has_gt:
    print(f"  Ground truth объём: {total_gt_volume:.6f} м³")
else:
    print(f"  Ground truth объём: недоступен")


# ============================================================
# 2. ФИЛЬТРАЦИЯ (SOR)
# ============================================================

print("\n" + "=" * 60)
print("ФИЛЬТРАЦИЯ")
print("=" * 60)

# Min-range фильтр: удалить точки в цилиндре (по XY) вокруг сенсора.
if args.min_range > 0:
    dxy = np.linalg.norm(all_pts_noisy[:, :2] - scanner_pos[:2], axis=1)
    keep = dxy >= args.min_range
    n_removed = (~keep).sum()
    pts_for_filter = all_pts_noisy[keep]
    print(f"Min-range фильтр (XY, {args.min_range} м): удалено {n_removed:,} точек, "
          f"осталось {len(pts_for_filter):,}")
else:
    pts_for_filter = all_pts_noisy

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_for_filter)

# Voxel downsample перед SOR — обязателен для больших реальных облаков.
# SOR имеет сложность O(N·k·log N) и на 1M+ точек занимает минуты.
if args.downsample > 0:
    n_before = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=args.downsample)
    n_after = len(pcd.points)
    print(f"Voxel downsample ({args.downsample*1000:.1f} мм): {n_before:,} → {n_after:,} точек "
          f"({100*(1-n_after/n_before):.1f}% сжатие)")

# SOR фильтр
n_in = len(pcd.points)
print(f"SOR (k=20, σ={args.sor_std_ratio}): {n_in:,} точек...", flush=True)
pcd_sor, ind_sor = pcd.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=args.sor_std_ratio, print_progress=True
)
pts_after_sor = np.asarray(pcd_sor.points)
n_removed_sor = n_in - len(pts_after_sor)
print(f"SOR: удалено {n_removed_sor} ({100 * n_removed_sor / n_in:.1f}%), осталось {len(pts_after_sor):,}")

# Radius outlier removal убран — слишком агрессивный даже с мягкими параметрами
pts_filtered = pts_after_sor
print(f"Итого после фильтрации: {len(pts_filtered):,}")

# Сохранить отфильтрованное облако для повторных запусков
np.savez_compressed(f'{OUTPUT_DIR}/cloud_filtered.npz', points=pts_filtered)
print(f"Фильтрованное облако сохранено в {OUTPUT_DIR}/cloud_filtered.npz")


# ============================================================
# 3. ОТДЕЛЕНИЕ ЗЕМЛИ ОТ РАСТИТЕЛЬНОСТИ
# ============================================================

print("\n" + "=" * 60)
print("КЛАССИФИКАЦИЯ ЗЕМЛЯ / РАСТИТЕЛЬНОСТЬ")
print("=" * 60)

height_threshold = 0.04

ground_mask = pts_filtered[:, 2] < height_threshold
vegetation_mask = ~ground_mask

ground_classified = pts_filtered[ground_mask]
vegetation_classified = pts_filtered[vegetation_mask]

print(f"  Порог высоты: {height_threshold} м")
print(f"  Точек земли: {len(ground_classified)}")
print(f"  Точек растительности: {len(vegetation_classified)}")


# ============================================================
# 4. ОЦЕНКА ОБЪЁМА
# ============================================================

print("\n" + "=" * 60)
print("ОЦЕНКА ОБЪЁМА")
print("=" * 60)


def voxel_volume(points, voxel_size=0.01):
    """Воксельный метод."""
    indices = np.floor(points / voxel_size).astype(int)
    unique_voxels = set(map(tuple, indices))
    return len(unique_voxels) * (voxel_size ** 3), len(unique_voxels)


def convex_hull_volume(points):
    """Convex hull."""
    if len(points) < 4:
        return 0.0
    hull = ConvexHull(points)
    return hull.volume


def alpha_shape_volume(points, alpha_value):
    """Alpha shape."""
    try:
        import alphashape
        if len(points) > 5000:
            idx = np.random.choice(len(points), 5000, replace=False)
            points_sub = points[idx]
        else:
            points_sub = points
        alpha_sh = alphashape.alphashape(points_sub, alpha_value)
        return getattr(alpha_sh, 'volume', 0.0)
    except Exception as e:
        print(f"    Alpha shape error: {e}")
        return 0.0


# Воксельный метод — разные размеры (только адекватные)
voxel_sizes = [0.005, 0.006, 0.0065, 0.0068, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02]
voxel_results = {}
print("\nВоксельный метод:")
for vs in voxel_sizes:
    vol, n_vox = voxel_volume(vegetation_classified, voxel_size=vs)
    if has_gt:
        err = (vol - total_gt_volume) / total_gt_volume * 100
        voxel_results[vs] = {'volume': vol, 'n_voxels': n_vox, 'error_pct': err}
        print(f"  size={vs:.3f} м: V={vol:.6f} м³, вокселей={n_vox}, ошибка={err:+.1f}%")
    else:
        voxel_results[vs] = {'volume': vol, 'n_voxels': n_vox, 'error_pct': None}
        print(f"  size={vs:.3f} м: V={vol:.6f} м³, вокселей={n_vox}")

# Convex Hull и Alpha Shape (опционально)
if args.skip_hull_methods:
    print("\nConvex Hull и Alpha Shape: пропущены (--skip-hull-methods)")
    ch_volume = 0.0
    ch_error = None
    alpha_results = {}
else:
    # Convex Hull
    ch_volume = convex_hull_volume(vegetation_classified)
    if has_gt:
        ch_error = (ch_volume - total_gt_volume) / total_gt_volume * 100
        print(f"\nConvex Hull: V={ch_volume:.6f} м³, ошибка={ch_error:+.1f}%")
    else:
        ch_error = None
        print(f"\nConvex Hull: V={ch_volume:.6f} м³")

    # Alpha Shape
    alpha_values = [1.0, 5.0, 10.0, 20.0, 50.0]
    alpha_results = {}
    print("\nAlpha Shape:")
    for av in alpha_values:
        vol = alpha_shape_volume(vegetation_classified, av)
        if vol > 0:
            if has_gt:
                err = (vol - total_gt_volume) / total_gt_volume * 100
                alpha_results[av] = {'volume': vol, 'error_pct': err}
                print(f"  α={av}: V={vol:.6f} м³, ошибка={err:+.1f}%")
            else:
                alpha_results[av] = {'volume': vol, 'error_pct': None}
                print(f"  α={av}: V={vol:.6f} м³")
        else:
            print(f"  α={av}: не удалось вычислить")


# ============================================================
# 5. СВОДНАЯ ТАБЛИЦА
# ============================================================

if has_gt:
    print("\n" + "=" * 70)
    print(f"{'МЕТОД':<30} {'ОБЪЁМ (м³)':<14} {'ОШИБКА':<14} {'ПРИМЕЧАНИЕ'}")
    print("=" * 70)
    print(f"{'Ground Truth':<30} {total_gt_volume:<14.6f} {'—':<14} {'эталон'}")
    print("-" * 70)
else:
    print("\n" + "=" * 50)
    print(f"{'МЕТОД':<30} {'ОБЪЁМ (м³)':<14}")
    print("=" * 50)

all_results = []

for vs in voxel_sizes:
    r = voxel_results[vs]
    label = f"Voxel {vs*1000:.1f}мм"
    if has_gt:
        note = "✓ лучший" if abs(r['error_pct']) == min(abs(voxel_results[v]['error_pct']) for v in voxel_sizes) else ""
        print(f"{label:<30} {r['volume']:<14.6f} {r['error_pct']:+13.1f}% {note}")
    else:
        print(f"{label:<30} {r['volume']:<14.6f}")
    all_results.append({'method': label, 'volume': r['volume'],
                        'error_pct': r['error_pct'], 'type': 'voxel'})

if not args.skip_hull_methods:
    if has_gt:
        print("-" * 70)
        print(f"{'Convex Hull':<30} {ch_volume:<14.6f} {ch_error:+13.1f}% завышает (оболочка)")
    else:
        print("-" * 50)
        print(f"{'Convex Hull':<30} {ch_volume:<14.6f}")
    all_results.append({'method': 'Convex Hull', 'volume': ch_volume,
                        'error_pct': ch_error, 'type': 'hull'})

    for av in sorted(alpha_results.keys()):
        r = alpha_results[av]
        label = f"Alpha Shape α={av}"
        if has_gt:
            print(f"{label:<30} {r['volume']:<14.6f} {r['error_pct']:+13.1f}%")
        else:
            print(f"{label:<30} {r['volume']:<14.6f}")
        all_results.append({'method': label, 'volume': r['volume'],
                            'error_pct': r['error_pct'], 'type': 'alpha'})

if has_gt:
    print("=" * 70)
else:
    print("=" * 50)


# ============================================================
# 6. ГРАФИКИ
# ============================================================

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white'
})

# --- График 1: Pipeline обработки (вид сбоку XZ) ---
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Определяем общий диапазон Z для всех графиков
z_min = min(all_pts_noisy[:, 2].min(), pts_filtered[:, 2].min()) - 0.1
z_max = max(all_pts_noisy[:, 2].max(), pts_filtered[:, 2].max()) + 0.1

# 1a: Исходное (с шумом)
ax = fig.add_subplot(gs[0, 0])
idx = np.random.choice(len(all_pts_noisy), min(12000, len(all_pts_noisy)), replace=False)
ax.scatter(all_pts_noisy[idx, 0], all_pts_noisy[idx, 2], s=0.2, alpha=0.4, c='gray')
ax.set_title('1. Исходное облако\n(с шумом и выбросами)')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(z_min, z_max)
ax.set_aspect('equal', adjustable='box')

# 1b: Вид сбоку (для синтетики - окклюзия, для реальных - другой ракурс)
ax = fig.add_subplot(gs[0, 1])
idx = np.random.choice(len(all_pts_noisy), min(12000, len(all_pts_noisy)), replace=False)
if len(plant_params) > 0:
    # Синтетические данные - показываем окклюзию
    colors_occ = np.where(all_pts_noisy[idx, 0] < -0.3, 'orange', 'steelblue')
    ax.scatter(all_pts_noisy[idx, 0], all_pts_noisy[idx, 2], s=0.2, alpha=0.4, c=colors_occ)
    ax.axvline(x=-0.3, color='red', linestyle=':', alpha=0.5)
    ax.annotate('← сканер', xy=(-2.5, 0.5), fontsize=9, color='red')
    ax.set_title('2. Эффект окклюзии\n(задние ряды разрежены)')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Z, м')
else:
    # Реальные данные - показываем вид сбоку YZ
    ax.scatter(all_pts_noisy[idx, 1], all_pts_noisy[idx, 2], s=0.2, alpha=0.4, c='steelblue')
    ax.set_title('2. Вид сбоку (Y-Z)\n(другой ракурс)')
    ax.set_xlabel('Y, м')
    ax.set_ylabel('Z, м')
ax.set_ylim(z_min, z_max)
ax.set_aspect('equal', adjustable='box')

# 1c: После фильтрации
ax = fig.add_subplot(gs[0, 2])
idx = np.random.choice(len(pts_filtered), min(12000, len(pts_filtered)), replace=False)
ax.scatter(pts_filtered[idx, 0], pts_filtered[idx, 2], s=0.2, alpha=0.4, c='steelblue')
ax.set_title('3. После фильтрации\n(SOR + Radius)')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(z_min, z_max)
ax.set_aspect('equal', adjustable='box')

# 1d: Классификация
ax = fig.add_subplot(gs[1, 0])
idx_g = np.random.choice(len(ground_classified), min(5000, len(ground_classified)), replace=False)
idx_v = np.random.choice(len(vegetation_classified), min(7000, len(vegetation_classified)), replace=False)
ax.scatter(ground_classified[idx_g, 0], ground_classified[idx_g, 2],
           s=0.3, alpha=0.5, c='#8B4513', label='Земля')
ax.scatter(vegetation_classified[idx_v, 0], vegetation_classified[idx_v, 2],
           s=0.3, alpha=0.5, c='#228B22', label='Растительность')
ax.axhline(y=height_threshold, color='red', linestyle='--', alpha=0.7, label=f'Порог {height_threshold} м')
ax.set_title('4. Классификация')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(-0.1, 0.9)
ax.legend(markerscale=10, fontsize=9)

# 1e: Вид сверху
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(vegetation_classified[idx_v, 0], vegetation_classified[idx_v, 1],
                s=0.3, alpha=0.4, c=vegetation_classified[idx_v, 2], cmap='YlGn', vmin=0)
ax.set_title('5. Вид сверху\n(цвет = высота)')
ax.set_xlabel('X, м')
ax.set_ylabel('Y, м')
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, label='Z, м', shrink=0.8)

# 1f: Гистограмма высот
ax = fig.add_subplot(gs[1, 2])
ax.hist(vegetation_classified[:, 2], bins=50, color='#228B22', alpha=0.7, edgecolor='darkgreen')
if has_gt and len(plant_params) > 0:
    mean_h = np.mean([p['height'] for p in plant_params])
    ax.axvline(x=mean_h, color='red', linestyle='--', linewidth=2,
               label=f'Средняя высота GT = {mean_h:.2f} м')
    ax.legend(fontsize=9)
ax.set_title('6. Распределение высот')
ax.set_xlabel('Высота Z, м')
ax.set_ylabel('Кол-во точек')

plt.savefig(f'{OUTPUT_DIR}/01_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nСохранён: 01_pipeline.png")


# --- График 2: Сравнение воксельных методов ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Только воксельные методы 5-15мм (лучшие)
voxel_only = [r for r in all_results if r['type'] == 'voxel']
best_voxels = [r for r in voxel_only if 0.005 <= float(r['method'].split()[1].replace('мм', '')) / 1000 <= 0.015]
methods_best = [r['method'] for r in best_voxels]
volumes_best = [r['volume'] for r in best_voxels]
errors_best = [r['error_pct'] for r in best_voxels]

# График 1: Лучшие методы (5-15мм)
bars1 = ax1.bar(range(len(methods_best)), volumes_best, color='#2196F3',
                alpha=0.8, edgecolor='black', linewidth=0.5)

if has_gt:
    ax1.axhline(y=total_gt_volume, color='red', linestyle='--', linewidth=2,
                label=f'Ground Truth = {total_gt_volume:.4f} м³')

    for i, (bar, err) in enumerate(zip(bars1, errors_best)):
        color = '#228B22' if abs(err) < 50 else '#FF6600' if abs(err) < 150 else '#CC0000'
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0005,
                 f'{err:+.0f}%', ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color=color)
    ax1.legend(fontsize=10)

ax1.set_xticks(range(len(methods_best)))
ax1.set_xticklabels(methods_best, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Объём, м³')
ax1.set_title('Лучшие воксельные методы (5-15мм)')
ax1.grid(True, alpha=0.3, axis='y')

# График 2: Trade-off точность vs производительность (все размеры)
vs_list = sorted(voxel_results.keys())
n_voxels_list = [voxel_results[v]['n_voxels'] for v in vs_list]
vs_mm_list = [v * 1000 for v in vs_list]

if has_gt:
    abs_errors_list = [abs(voxel_results[v]['error_pct']) for v in vs_list]

    # Цвет по размеру вокселя
    colors = plt.cm.viridis(np.linspace(0, 1, len(vs_list)))
    scatter = ax2.scatter(n_voxels_list, abs_errors_list, s=100, c=colors,
                         alpha=0.7, edgecolor='black', linewidth=1)

    # Аннотации для всех точек
    for i, (vs, n_vox, err) in enumerate(zip(vs_list, n_voxels_list, abs_errors_list)):
        ax2.annotate(f'{vs*1000:.1f}мм',
                    xy=(n_vox, err), xytext=(5, 5),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax2.set_xlabel('Количество вокселей')
    ax2.set_ylabel('|Ошибка|, %')
    ax2.set_title('Точность vs Вычислительная нагрузка')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axhline(y=20, color='green', linestyle=':', alpha=0.5, label='20% порог')
    ax2.legend(fontsize=9)
else:
    # Без GT показываем просто объём vs количество вокселей
    volumes_list = [voxel_results[v]['volume'] for v in vs_list]
    colors = plt.cm.viridis(np.linspace(0, 1, len(vs_list)))
    scatter = ax2.scatter(n_voxels_list, volumes_list, s=100, c=colors,
                         alpha=0.7, edgecolor='black', linewidth=1)

    for i, (vs, n_vox, vol) in enumerate(zip(vs_list, n_voxels_list, volumes_list)):
        ax2.annotate(f'{vs*1000:.1f}мм',
                    xy=(n_vox, vol), xytext=(5, 5),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax2.set_xlabel('Количество вокселей')
    ax2.set_ylabel('Объём, м³')
    ax2.set_title('Объём vs Вычислительная нагрузка')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_volume_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранён: 02_volume_comparison.png")


# --- График 3: Зависимость от размера вокселя ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vs_list = sorted(voxel_results.keys())
vv_list = [voxel_results[v]['volume'] for v in vs_list]
vs_mm = [v * 1000 for v in vs_list]

# Объём
ax = axes[0]
ax.plot(vs_mm, vv_list, 'o-', color='#2196F3', linewidth=2, markersize=7)
if has_gt:
    ax.axhline(y=total_gt_volume, color='red', linestyle='--', linewidth=1.5,
               label=f'GT = {total_gt_volume:.5f}')
    ax.fill_between(vs_mm, total_gt_volume * 0.8, total_gt_volume * 1.2,
                    alpha=0.15, color='green', label='±20% от GT')
    ax.legend(fontsize=9)
ax.set_xlabel('Размер вокселя, мм')
ax.set_ylabel('Объём, м³')
ax.set_title('Оценка объёма')
ax.grid(True, alpha=0.3)

# Ошибка (или текст если нет GT)
ax = axes[1]
if has_gt:
    ve_list = [abs(voxel_results[v]['error_pct']) for v in vs_list]
    ax.plot(vs_mm, ve_list, 's-', color='#f44336', linewidth=2, markersize=7)
    ax.axhline(y=20, color='green', linestyle=':', alpha=0.7, label='20% ошибка')
    ax.set_xlabel('Размер вокселя, мм')
    ax.set_ylabel('|Ошибка|, %')
    ax.set_title('Абсолютная ошибка')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
else:
    ax.text(0.5, 0.5, 'Объём не задан', ha='center', va='center',
            fontsize=16, transform=ax.transAxes)
    ax.set_title('Абсолютная ошибка')
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_voxel_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранён: 03_voxel_analysis.png")


# --- График 4: Почему Convex Hull / Alpha Shape не работают ---
if not args.skip_hull_methods:
    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = []
    method_volumes = []
    method_colors = []

    # Лучший воксельный
    if has_gt:
        best_vs = min(voxel_results, key=lambda v: abs(voxel_results[v]['error_pct']))
    else:
        best_vs = DEFAULT_VOXEL_SIZE_NO_GT

    method_names.append(f'Voxel {best_vs*1000:.1f}мм')
    method_volumes.append(voxel_results[best_vs]['volume'])
    method_colors.append('#2196F3')

    # Convex Hull
    method_names.append('Convex Hull')
    method_volumes.append(ch_volume)
    method_colors.append('#FF9800')

    # Alpha shapes
    for av in sorted(alpha_results.keys()):
        method_names.append(f'α-shape α={av}')
        method_volumes.append(alpha_results[av]['volume'])
        method_colors.append('#4CAF50')

    if has_gt:
        method_names.append('Ground Truth')
        method_volumes.append(total_gt_volume)
        method_colors.append('#f44336')

    bars = ax.barh(range(len(method_names)), method_volumes, color=method_colors,
                   alpha=0.8, edgecolor='black', linewidth=0.5)

    if has_gt:
        ax.axvline(x=total_gt_volume, color='red', linestyle='--', linewidth=2)

        for i, (bar, vol) in enumerate(zip(bars, method_volumes)):
            if vol > 0.1:
                ratio = vol / total_gt_volume
                ax.text(vol + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'×{ratio:.0f}', va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Объём, м³')
        ax.set_title('Convex Hull и Alpha Shape завышают объём\n'
                     '(считают оболочку, а не тонкие структуры растений)')
        ax.set_xscale('log')
    else:
        # Без GT просто показываем сравнение объёмов
        for i, (bar, vol) in enumerate(zip(bars, method_volumes)):
            ax.text(vol * 1.05, bar.get_y() + bar.get_height() / 2,
                    f'{vol:.4f} м³', va='center', fontsize=9)

        ax.set_xlabel('Объём, м³')
        ax.set_title('Сравнение методов оценки объёма')
        ax.set_xscale('log')

    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_hull_vs_voxel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Сохранён: 04_hull_vs_voxel.png")
else:
    print("График 04 пропущен (--skip-hull-methods)")


# ============================================================
# 7. 3D ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ (Plotly)
# ============================================================

print("\nГенерация 3D визуализации...")

# Подвыборка для производительности
max_3d_pts = 20000

def subsample(pts, n):
    if len(pts) > n:
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]
    return pts

# --- 3D: Исходное облако с шумом ---
pts_sub = subsample(all_pts_noisy, max_3d_pts)
fig3d_raw = go.Figure(data=[go.Scatter3d(
    x=pts_sub[:, 0], y=pts_sub[:, 1], z=pts_sub[:, 2],
    mode='markers',
    marker=dict(size=1.2, color=pts_sub[:, 2], colorscale='Viridis',
                opacity=0.6, colorbar=dict(title='Z, м')),
    name='Точки'
)])
fig3d_raw.update_layout(
    title='Исходное облако точек (с шумом, выбросами и окклюзией)',
    scene=dict(
        xaxis_title='X, м', yaxis_title='Y, м', zaxis_title='Z, м',
        aspectmode='data',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0))
    ),
    width=900, height=650
)
fig3d_raw.write_html(f'{OUTPUT_DIR}/3d_raw.html')
print("Сохранён: 3d_raw.html")

# --- 3D: Отфильтрованное облако (после SOR) ---
pts_f_sub = subsample(pts_filtered, max_3d_pts)
fig3d_filt = go.Figure(data=[go.Scatter3d(
    x=pts_f_sub[:, 0], y=pts_f_sub[:, 1], z=pts_f_sub[:, 2],
    mode='markers',
    marker=dict(size=1.2, color=pts_f_sub[:, 2], colorscale='Viridis',
                opacity=0.6, colorbar=dict(title='Z, м')),
    name='Точки'
)])
fig3d_filt.update_layout(
    title=f'Отфильтрованное облако (после SOR, {len(pts_filtered):,} точек)',
    scene=dict(
        xaxis_title='X, м', yaxis_title='Y, м', zaxis_title='Z, м',
        aspectmode='data',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0))
    ),
    width=900, height=650
)
fig3d_filt.write_html(f'{OUTPUT_DIR}/3d_filtered.html')
print("Сохранён: 3d_filtered.html")

# --- 3D: Классифицированное облако ---
g_sub = subsample(ground_classified, max_3d_pts // 2)
v_sub = subsample(vegetation_classified, max_3d_pts)

fig3d_class = go.Figure()
fig3d_class.add_trace(go.Scatter3d(
    x=g_sub[:, 0], y=g_sub[:, 1], z=g_sub[:, 2],
    mode='markers',
    marker=dict(size=1.2, color='#8B4513', opacity=0.4),
    name='Земля'
))
fig3d_class.add_trace(go.Scatter3d(
    x=v_sub[:, 0], y=v_sub[:, 1], z=v_sub[:, 2],
    mode='markers',
    marker=dict(size=1.5, color=v_sub[:, 2], colorscale='YlGn',
                opacity=0.7, colorbar=dict(title='Z, м')),
    name='Растительность'
))
# Позиция сканера
# fig3d_class.add_trace(go.Scatter3d(
#     x=[scanner_pos[0]], y=[scanner_pos[1]], z=[scanner_pos[2]],
#     mode='markers+text',
#     marker=dict(size=8, color='red', symbol='diamond'),
#     text=['TLS сканер'], textposition='top center',
#     name='Сканер'
# ))
fig3d_class.update_layout(
    title='Классифицированное облако (земля / растительность)',
    scene=dict(
        xaxis_title='X, м', yaxis_title='Y, м', zaxis_title='Z, м',
        aspectmode='data',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0))
    ),
    width=900, height=650
)
fig3d_class.write_html(f'{OUTPUT_DIR}/3d_classified.html')
print("Сохранён: 3d_classified.html")


# ============================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

results_data = {
    'ground_truth_volume_m3': total_gt_volume if has_gt else None,
    'n_plants': len(plant_params),
    'scanner_position': scanner_pos.tolist(),
    'points': {
        'raw_total': len(all_pts_clean),
        'after_occlusion': len(all_pts_occluded),
        'occlusion_removed_pct': round(100 * n_occluded / len(all_pts_clean), 1) if len(all_pts_clean) > 0 else 0,
        'with_noise': len(all_pts_noisy),
        'after_sor': len(pts_after_sor),
        'after_radius': len(pts_filtered),
        'ground': len(ground_classified),
        'vegetation': len(vegetation_classified)
    },
    'voxel_results': {},
    'convex_hull': {
        'volume': ch_volume,
        'error_pct': round(ch_error, 1) if ch_error is not None else None
    },
    'alpha_shape': {},
}

# Добавляем mean_height только если есть plant_params
if len(plant_params) > 0 and has_gt:
    mean_h = np.mean([p['height'] for p in plant_params])
    results_data['mean_height_m'] = float(mean_h)

# Воксельные результаты
for k, v in voxel_results.items():
    results_data['voxel_results'][f'{k*1000:.0f}mm'] = {
        'volume': v['volume'],
        'n_voxels': v['n_voxels'],
        'error_pct': round(v['error_pct'], 1) if v['error_pct'] is not None else None
    }

# Alpha shape результаты
for k, v in alpha_results.items():
    results_data['alpha_shape'][f'alpha_{k}'] = {
        'volume': v['volume'],
        'error_pct': round(v['error_pct'], 1) if v['error_pct'] is not None else None
    }

# Лучший метод
if has_gt:
    best_vs = min(voxel_results, key=lambda v: abs(voxel_results[v]['error_pct']))
    results_data['best_method'] = f'Voxel {best_vs*1000:.0f}mm'
    results_data['best_error_pct'] = round(abs(voxel_results[best_vs]['error_pct']), 1)
else:
    # Без GT используем дефолтный оптимальный размер вокселя
    best_vs = DEFAULT_VOXEL_SIZE_NO_GT
    results_data['best_method'] = f'Voxel {best_vs*1000:.0f}mm (default optimal)'
    results_data['best_error_pct'] = None

with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print("\nСохранён: results.json")
print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
print("=" * 60)

if has_gt:
    print(f"\nЛучший метод: Voxel {best_vs*1000:.1f}мм, ошибка {abs(voxel_results[best_vs]['error_pct']):.1f}%")
else:
    print(f"\nРекомендуемый метод: Voxel {best_vs*1000:.1f}мм")
    print(f"Оценка объёма: {voxel_results[best_vs]['volume']:.6f} м³")