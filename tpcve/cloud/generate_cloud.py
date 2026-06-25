"""
Генерация синтетического облака точек TLS для пшеничного поля.
Может использоваться как модуль или запускаться напрямую для сохранения облака.
"""

import numpy as np
import argparse
from pathlib import Path


def generate_ground(x_range=(-2, 2), y_range=(-2, 2), density=5000, noise_std=0.008):
    """Генерация плоскости земли с микрорельефом."""
    n = density
    x = np.random.uniform(x_range[0], x_range[1], n)
    y = np.random.uniform(y_range[0], y_range[1], n)
    z = 0.015 * np.sin(2 * x) * np.cos(3 * y) + np.random.normal(0, noise_std, n)
    return np.column_stack([x, y, z])


def generate_wheat_plant(cx, cy, height=0.6, stem_radius=0.008,
                         ear_length=0.08, ear_radius=0.012,
                         points_stem=200, points_ear=150, points_leaves=120):
    """
    Генерация одного растения пшеницы:
    - стебель: тонкий цилиндр с лёгким изгибом
    - колос: эллипсоид на вершине
    - листья: 2-4 изогнутые полоски
    """
    pts = []

    # Стебель
    t_stem = np.random.uniform(0, 1, points_stem)
    z_stem = t_stem * height
    bend_dir = np.random.uniform(0, 2 * np.pi)
    bend_amount = np.random.uniform(0.005, 0.02)
    theta = np.random.uniform(0, 2 * np.pi, points_stem)
    x_stem = cx + stem_radius * np.cos(theta) + bend_amount * t_stem ** 2 * np.cos(bend_dir)
    y_stem = cy + stem_radius * np.sin(theta) + bend_amount * t_stem ** 2 * np.sin(bend_dir)
    pts.append(np.column_stack([x_stem, y_stem, z_stem]))

    # Колос
    top_x = cx + bend_amount * np.cos(bend_dir)
    top_y = cy + bend_amount * np.sin(bend_dir)
    theta_ear = np.random.uniform(0, 2 * np.pi, points_ear)
    phi_ear = np.random.uniform(0, np.pi, points_ear)
    x_ear = top_x + ear_radius * np.sin(phi_ear) * np.cos(theta_ear)
    y_ear = top_y + ear_radius * np.sin(phi_ear) * np.sin(theta_ear)
    z_ear = height + ear_length * np.cos(phi_ear)
    pts.append(np.column_stack([x_ear, y_ear, z_ear]))

    # Листья
    n_leaves = np.random.randint(2, 5)
    for _ in range(n_leaves):
        leaf_angle = np.random.uniform(0, 2 * np.pi)
        leaf_start_z = np.random.uniform(0.1 * height, 0.65 * height)
        leaf_length = np.random.uniform(0.08, 0.18)
        n_lpts = points_leaves
        t = np.sort(np.random.uniform(0, 1, n_lpts))
        x_leaf = cx + t * leaf_length * np.cos(leaf_angle)
        y_leaf = cy + t * leaf_length * np.sin(leaf_angle)
        z_leaf = leaf_start_z + t * 0.04 - t ** 2 * 0.08
        width = 0.005 * (1 - 0.5 * t)
        x_leaf += np.random.normal(0, width, n_lpts)
        y_leaf += np.random.normal(0, width, n_lpts)
        pts.append(np.column_stack([x_leaf, y_leaf, z_leaf]))

    return np.vstack(pts)


def generate_wheat_field(n_rows=5, plants_per_row=8, row_spacing=0.25,
                         plant_spacing=0.12, height_mean=0.55, height_std=0.08):
    """Генерация поля пшеницы: ряды растений."""
    all_plants = []
    plant_params = []

    for row in range(n_rows):
        cy = -((n_rows - 1) * row_spacing / 2) + row * row_spacing
        for col in range(plants_per_row):
            cx = -((plants_per_row - 1) * plant_spacing / 2) + col * plant_spacing
            cx += np.random.normal(0, 0.012)
            cy += np.random.normal(0, 0.012)
            h = max(0.3, np.random.normal(height_mean, height_std))

            plant = generate_wheat_plant(cx, cy, height=h)
            all_plants.append(plant)

            stem_vol = np.pi * 0.008 ** 2 * h
            ear_vol = (4 / 3) * np.pi * 0.012 * 0.012 * 0.08
            leaf_vol = 3 * 0.15 * 0.008 * 0.002
            plant_params.append({
                'cx': cx, 'cy': cy, 'height': h,
                'volume': stem_vol + ear_vol + leaf_vol
            })

    return np.vstack(all_plants), plant_params


def simulate_occlusion(points, scanner_pos=np.array([-2.5, 0, 0.5]),
                       occlusion_strength=0.2):
    """Моделирование окклюзии TLS."""
    dists = np.linalg.norm(points - scanner_pos, axis=1)
    max_dist = dists.max()
    p_remove = occlusion_strength * (dists / max_dist) ** 1.5
    above_ground = points[:, 2] > 0.03
    p_remove[above_ground] *= 1.5
    p_remove = np.clip(p_remove, 0, 0.7)
    keep_mask = np.random.random(len(points)) > p_remove
    return points[keep_mask], keep_mask


def add_realistic_noise(points, noise_std=0.008, outlier_fraction=0.06,
                        phantom_fraction=0.03):
    """Реалистичный шум TLS."""
    n = len(points)
    points = points + np.random.normal(0, noise_std, points.shape)

    n_outliers = int(n * outlier_fraction)
    outliers = np.random.uniform(
        [points[:, 0].min() - 1.0, points[:, 1].min() - 1.0, -0.2],
        [points[:, 0].max() + 1.0, points[:, 1].max() + 1.0, 1.2],
        (n_outliers, 3)
    )

    n_phantoms = int(n * phantom_fraction)
    phantom_idx = np.random.choice(n, n_phantoms)
    phantoms = points[phantom_idx].copy()
    phantoms[:, 2] *= np.random.uniform(0.2, 0.8, n_phantoms)
    phantoms[:, 0] += np.random.normal(0, 0.03, n_phantoms)
    phantoms[:, 1] += np.random.normal(0, 0.03, n_phantoms)

    return np.vstack([points, outliers, phantoms])


def generate_full_cloud(seed=42, occlusion_strength=0.1, noise_std=0.005,
                       outlier_fraction=0.03, phantom_fraction=0.015):
    """
    Генерирует полное облако точек с землёй, растительностью, окклюзией и шумом.

    Returns:
        dict: {
            'all_pts_noisy': облако с шумом,
            'ground_pts': исходные точки земли,
            'vegetation_pts': исходные точки растительности,
            'plant_params': параметры растений,
            'scanner_pos': позиция сканера,
            'total_gt_volume': ground truth объём
        }
    """
    np.random.seed(seed)

    ground_pts = generate_ground()
    vegetation_pts, plant_params = generate_wheat_field()
    total_gt_volume = sum(p['volume'] for p in plant_params)

    all_pts_clean = np.vstack([ground_pts, vegetation_pts])
    scanner_pos = np.array([-2.5, 0, 0.5])

    all_pts_occluded, _ = simulate_occlusion(all_pts_clean, scanner_pos, occlusion_strength)
    all_pts_noisy = add_realistic_noise(all_pts_occluded, noise_std,
                                        outlier_fraction, phantom_fraction)

    return {
        'all_pts_noisy': all_pts_noisy,
        'ground_pts': ground_pts,
        'vegetation_pts': vegetation_pts,
        'plant_params': plant_params,
        'scanner_pos': scanner_pos,
        'total_gt_volume': total_gt_volume
    }


def save_cloud(data, filepath):
    """Сохраняет облако точек в .npz файл."""
    np.savez_compressed(
        filepath,
        all_pts_noisy=data['all_pts_noisy'],
        ground_pts=data['ground_pts'],
        vegetation_pts=data['vegetation_pts'],
        scanner_pos=data['scanner_pos'],
        total_gt_volume=data['total_gt_volume'],
        plant_params=np.array([
            [p['cx'], p['cy'], p['height'], p['volume']]
            for p in data['plant_params']
        ])
    )


def load_cloud(filepath):
    """
    Загружает облако точек из .npz файла.

    Returns:
        dict: тот же формат что и generate_full_cloud()
    """
    npz = np.load(filepath)
    plant_params_arr = npz['plant_params']
    plant_params = [
        {'cx': row[0], 'cy': row[1], 'height': row[2], 'volume': row[3]}
        for row in plant_params_arr
    ]

    return {
        'all_pts_noisy': npz['all_pts_noisy'],
        'ground_pts': npz['ground_pts'],
        'vegetation_pts': npz['vegetation_pts'],
        'plant_params': plant_params,
        'scanner_pos': npz['scanner_pos'],
        'total_gt_volume': float(npz['total_gt_volume'])
    }


def detect_units(points):
    """
    Автоопределение единиц измерения облака точек.

    Returns:
        str: 'm', 'cm', или 'mm'
    """
    if len(points) == 0:
        return 'm'

    ranges = points.max(axis=0) - points.min(axis=0)
    max_range = ranges.max()

    # Вычисляем среднее расстояние до ближайшего соседа (выборка)
    sample_size = min(1000, len(points))
    sample_idx = np.random.choice(len(points), sample_size, replace=False)
    sample_points = points[sample_idx]

    # Простая эвристика: расстояние до ближайшего из 10 случайных точек
    avg_nearest = []
    for i in range(min(100, len(sample_points))):
        dists = np.linalg.norm(sample_points - sample_points[i], axis=1)
        dists = dists[dists > 0]  # исключаем саму точку
        if len(dists) > 0:
            avg_nearest.append(dists.min())

    avg_dist = np.mean(avg_nearest) if avg_nearest else max_range / 100

    # Эвристика определения единиц
    if max_range < 10:
        # Поле меньше 10 единиц - скорее всего метры
        return 'm'
    elif max_range > 100 and avg_dist > 10:
        # Большой масштаб и большие расстояния между точками - миллиметры
        return 'mm'
    elif max_range > 100:
        # Большой масштаб, но малые расстояния - сантиметры
        return 'cm'
    else:
        # Промежуточный случай - сантиметры
        return 'cm'


def load_db3_cloud(filepath):
    """
    Загрузка облака точек из ROS 2 bag-файла (.db3).

    Читает .db3 напрямую через sqlite3 (не требует metadata.yaml),
    CDR-десериализацию выполняет через rosbags.

    Args:
        filepath: путь к .db3 файлу

    Returns:
        np.ndarray: массив точек (N, 3) в исходных единицах (до конвертации)

    Raises:
        ImportError: если rosbags не установлен
        ValueError: если в bag нет топиков PointCloud2
    """
    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        raise ImportError("Для .db3 нужен rosbags: uv add rosbags")

    import sqlite3

    # Маппинг типов PointField datatype → numpy dtype
    FIELD_DTYPE = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        8: np.float64,
    }

    typestore = get_typestore(Stores.LATEST)
    msgtype = 'sensor_msgs/msg/PointCloud2'

    con = sqlite3.connect(str(filepath))
    try:
        # Найти топики с PointCloud2
        rows = con.execute(
            "SELECT id, name FROM topics WHERE type = ?", (msgtype,)
        ).fetchall()

        if not rows:
            available = [r[0] for r in con.execute("SELECT name FROM topics").fetchall()]
            raise ValueError(
                f"В bag не найдено топиков PointCloud2. "
                f"Доступные топики: {available}"
            )

        topic_id, topic_name = rows[0]
        print(f"  ROS 2 bag: топик '{topic_name}' ({msgtype})")
        if len(rows) > 1:
            print(f"  Найдено несколько PointCloud2 топиков: {[r[1] for r in rows]}, используется первый")

        messages = con.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,)
        ).fetchall()
    finally:
        con.close()

    if not messages:
        raise ValueError(f"В топике '{topic_name}' нет сообщений")

    all_points = []
    for (rawdata,) in messages:
        msg = typestore.deserialize_cdr(bytes(rawdata), msgtype)

        field_map = {f.name: f for f in msg.fields}
        if not all(k in field_map for k in ('x', 'y', 'z')):
            continue

        point_step = msg.point_step
        n_points = msg.width * msg.height
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        points_xyz = np.empty((n_points, 3), dtype=np.float64)
        for i, axis in enumerate(['x', 'y', 'z']):
            f = field_map[axis]
            dt = FIELD_DTYPE.get(f.datatype, np.float32)
            itemsize = np.dtype(dt).itemsize
            # Извлечь байты каждого поля для всех точек одним срезом
            field_raw = np.lib.stride_tricks.as_strided(
                raw[f.offset:],
                shape=(n_points, itemsize),
                strides=(point_step, 1)
            ).copy()
            vals = np.frombuffer(field_raw.tobytes(), dtype=dt)
            points_xyz[:, i] = vals.astype(np.float64)

        # Отфильтровать NaN/Inf (незаполненные точки организованных облаков)
        valid = np.isfinite(points_xyz).all(axis=1)
        points_xyz = points_xyz[valid]
        if len(points_xyz) > 0:
            all_points.append(points_xyz)

    if not all_points:
        raise ValueError("В bag нет валидных сообщений PointCloud2 с xyz данными")

    return np.vstack(all_points)


def _read_pcd_np(filepath: str) -> np.ndarray:
    """Прочитать PCD файл через numpy (ASCII и binary)."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    header_end = raw.find(b'DATA ')
    if header_end == -1:
        raise ValueError("Invalid PCD: missing DATA header")

    eol = raw.find(b'\n', header_end)
    data_fmt = raw[header_end:eol].decode('ascii').strip()
    data_off = eol + 1

    header = raw[:header_end].decode('ascii')
    fields = []
    types = {}
    sizes = {}
    width = 0
    height = 1

    for line in header.split('\n'):
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'FIELDS':
            fields = parts[1:]
        elif parts[0] == 'TYPE':
            for i, t in enumerate(parts[1:]):
                if i < len(fields):
                    types[fields[i]] = t
        elif parts[0] == 'SIZE':
            for i, s in enumerate(parts[1:]):
                if i < len(fields):
                    sizes[fields[i]] = int(s)
        elif parts[0] == 'WIDTH':
            width = int(parts[1])
        elif parts[0] == 'HEIGHT':
            height = int(parts[1])

    x_idx = fields.index('x')
    y_idx = fields.index('y')
    z_idx = fields.index('z')

    if data_fmt == 'DATA ascii':
        data = np.loadtxt(raw[data_off:].decode('ascii').split('\n'))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, [x_idx, y_idx, z_idx]].astype(np.float64)

    if data_fmt != 'DATA binary':
        raise ValueError(f"Unsupported PCD format: {data_fmt}")

    n_points = width * height
    DTYPE_MAP = {
        ('F', 4): np.float32, ('F', 8): np.float64,
        ('I', 1): np.int8, ('I', 2): np.int16, ('I', 4): np.int32,
        ('U', 1): np.uint8, ('U', 2): np.uint16, ('U', 4): np.uint32,
    }
    dt_list = []
    for f in fields:
        key = (types.get(f, 'F'), sizes.get(f, 4))
        dt_list.append(DTYPE_MAP.get(key, np.float32))
    dt = np.dtype(list(zip(fields, dt_list)))
    parsed = np.frombuffer(raw[data_off:], dtype=dt, count=n_points)
    return np.column_stack([parsed['x'], parsed['y'], parsed['z']]).astype(np.float64)


def _read_ply_np(filepath: str) -> np.ndarray:
    """Прочитать ASCII PLY через numpy."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    text = raw.decode('ascii')
    lines = text.split('\n')

    vertex_count = 0
    props = []
    in_vertex = False
    header_end = 0
    is_ascii = False

    for i, line in enumerate(lines):
        s = line.strip()
        if s == 'format ascii 1.0':
            is_ascii = True
        elif s.startswith('element vertex'):
            in_vertex = True
            vertex_count = int(s.split()[-1])
        elif s.startswith('property') and in_vertex:
            props.append(s.split()[-1])
        elif s == 'end_header':
            header_end = i + 1
            break
        else:
            in_vertex = False

    if not is_ascii:
        raise ValueError("Only ASCII PLY supported without open3d")

    data = np.loadtxt(lines[header_end:header_end + vertex_count])
    if data.ndim == 1:
        data = data.reshape(1, -1)

    x_idx = props.index('x')
    y_idx = props.index('y')
    z_idx = props.index('z')
    return data[:, [x_idx, y_idx, z_idx]].astype(np.float64)


def load_real_cloud(filepath, units='auto', verbose=True):
    """
    Загрузка реального облака точек из различных форматов.

    Поддерживаемые форматы:
    - PCD (ASCII + binary), PLY (ASCII), XYZ, PTS, TXT (через numpy)
    - LAS, LAZ (через laspy, если установлен)
    - DB3, ROS 2 bag (через rosbags, если установлен)

    Args:
        filepath: путь к файлу
        units: 'auto', 'm', 'cm', или 'mm' - единицы измерения координат

    Returns:
        dict: {
            'all_pts_noisy': облако точек (N, 3) в метрах,
            'ground_pts': пустой массив,
            'vegetation_pts': пустой массив,
            'plant_params': [],
            'scanner_pos': np.array([0, 0, 0]),
            'total_gt_volume': 0.0,
            'units_detected': str (если units='auto')
        }
    """
    ext = Path(filepath).suffix.lower()

    # LAS/LAZ через laspy
    if ext in ['.las', '.laz']:
        try:
            import laspy
            las = laspy.read(filepath)
            points = np.vstack([las.x, las.y, las.z]).T
        except ImportError:
            raise ImportError("Для LAS/LAZ нужен laspy: pip install laspy")

    # PCD через numpy
    elif ext == '.pcd':
        points = _read_pcd_np(filepath)

    # PLY (ASCII) через numpy
    elif ext == '.ply':
        points = _read_ply_np(filepath)

    # XYZ/PTS/TXT — простые ASCII с колонками
    elif ext in ['.xyz', '.pts', '.txt']:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        points = data[:, :3].astype(np.float64)

    # ROS 2 bag (.db3)
    elif ext == '.db3':
        points = load_db3_cloud(filepath)

    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    # Определение и конвертация единиц
    if units == 'auto':
        detected_units = detect_units(points)
        if verbose:
            print(f"  Автоопределение единиц: {detected_units}")
    else:
        detected_units = units
        if verbose:
            print(f"  Используются указанные единицы: {units}")

    # Конвертация в метры
    if detected_units == 'mm':
        points = points / 1000.0
        if verbose:
            print(f"  Координаты конвертированы: мм → м (÷1000)")
    elif detected_units == 'cm':
        points = points / 100.0
        if verbose:
            print(f"  Координаты конвертированы: см → м (÷100)")
    elif detected_units == 'm':
        if verbose:
            print(f"  Координаты уже в метрах")

    result = {
        'all_pts_noisy': points,
        'ground_pts': np.array([]),
        'vegetation_pts': np.array([]),
        'plant_params': [],
        'scanner_pos': np.array([0.0, 0.0, 0.0]),
        'total_gt_volume': 0.0
    }

    if units == 'auto':
        result['units_detected'] = detected_units

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Генерация синтетического облака точек TLS')
    parser.add_argument('-o', '--output', default='cloud.npz',
                       help='Путь для сохранения (default: cloud.npz)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    print(f"Генерация облака точек (seed={args.seed})...")
    data = generate_full_cloud(seed=args.seed)

    print(f"  Точек земли: {len(data['ground_pts'])}")
    print(f"  Точек растительности: {len(data['vegetation_pts'])}")
    print(f"  Точек итого (с шумом): {len(data['all_pts_noisy'])}")
    print(f"  Растений: {len(data['plant_params'])}")
    print(f"  Ground truth объём: {data['total_gt_volume']:.6f} м³")

    save_cloud(data, args.output)
    print(f"\nСохранено в {args.output}")
