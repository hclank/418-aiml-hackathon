from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


OPTIONAL_BAND_FILES = {
    "bathymetry": "bathymetry.tif",
    "owi_mask": "owiMask.tif",
    "wind_direction": "owiWindDirection.tif",
    "wind_quality": "owiWindQuality.tif",
    "wind_speed": "owiWindSpeed.tif",
}


@dataclass(frozen=True)
class ScenePaths:
    scene_id: str
    vv_path: Path
    vh_path: Path
    optional_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneBundle:
    scene_id: str
    image: np.ndarray
    transform: object
    crs: object
    vv_path: Path
    vh_path: Path


@dataclass(frozen=True)
class SceneMetadata:
    scene_id: str
    transform: object
    crs: object
    height: int
    width: int
    vv_path: Path
    vh_path: Path


def _candidate_scene_roots(scene_root: Path, scene_id: str) -> list[Path]:
    roots: list[Path] = []
    direct = scene_root / scene_id
    if direct.exists():
        roots.append(direct)

    roots.extend(
        path
        for path in scene_root.glob(f"**/{scene_id}")
        if path.is_dir() and path not in roots
    )
    return roots


def _find_band_path(scene_root: Path, scene_id: str, keywords: tuple[str, ...]) -> Path:
    matches: list[Path] = []

    candidate_roots = _candidate_scene_roots(scene_root, scene_id)
    for candidate_root in candidate_roots:
        for suffix in ("*.tif", "*.tiff"):
            for path in candidate_root.rglob(suffix):
                upper_name = path.name.upper()
                if any(keyword.upper() in upper_name for keyword in keywords):
                    matches.append(path)

    if not matches:
        for suffix in ("*.tif", "*.tiff"):
            for path in scene_root.rglob(suffix):
                upper_name = path.name.upper()
                upper_path = str(path).upper()
                if scene_id.upper() in upper_path and any(
                    keyword.upper() in upper_name for keyword in keywords
                ):
                    matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"Could not find {keywords} raster for scene {scene_id} under {scene_root}."
        )

    matches.sort(key=lambda path: (len(str(path)), str(path)))
    return matches[0]


def _find_optional_band_path(scene_root: Path, scene_id: str, filename: str) -> Path | None:
    candidate_roots = _candidate_scene_roots(scene_root, scene_id)
    for candidate_root in candidate_roots:
        direct = candidate_root / filename
        if direct.exists():
            return direct
        matches = list(candidate_root.rglob(filename))
        if matches:
            matches.sort(key=lambda path: (len(str(path)), str(path)))
            return matches[0]

    for path in scene_root.rglob(filename):
        if scene_id.upper() in str(path).upper():
            return path
    return None


def find_scene_paths(scene_root: Path, scene_id: str) -> ScenePaths:
    vv_path = _find_band_path(scene_root, scene_id, ("VV_DB", "VV"))
    vh_path = _find_band_path(scene_root, scene_id, ("VH_DB", "VH"))
    optional_paths = {
        name: path
        for name, filename in OPTIONAL_BAND_FILES.items()
        if (path := _find_optional_band_path(scene_root, scene_id, filename)) is not None
    }
    return ScenePaths(
        scene_id=scene_id,
        vv_path=vv_path,
        vh_path=vh_path,
        optional_paths=optional_paths,
    )


def list_available_scene_ids(scene_root: Path) -> set[str]:
    scene_ids: set[str] = set()
    for vv_path in scene_root.rglob("VV_dB.tif"):
        scene_id = vv_path.parent.name
        vh_path = vv_path.parent / "VH_dB.tif"
        if vh_path.exists():
            scene_ids.add(scene_id)

    if not scene_ids:
        for vv_path in scene_root.rglob("VV.tif"):
            scene_id = vv_path.parent.name
            vh_path = vv_path.parent / "VH.tif"
            if vh_path.exists():
                scene_ids.add(scene_id)
    return scene_ids


def _probe_raster_read(path: Path, max_dim: int) -> None:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        if ds.width <= 0 or ds.height <= 0:
            raise RuntimeError(f"Raster has invalid dimensions: {ds.width}x{ds.height}")

        probe_dim = max(1, int(max_dim))
        scale = max(ds.height, ds.width) / float(probe_dim)
        scale = max(1.0, scale)
        out_height = max(1, int(round(ds.height / scale)))
        out_width = max(1, int(round(ds.width / scale)))
        ds.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
        )


def is_scene_readable(
    scene_root: Path,
    scene_id: str,
    *,
    max_dim: int = 512,
) -> tuple[bool, str | None]:
    try:
        paths = find_scene_paths(scene_root, scene_id)
        _probe_raster_read(paths.vv_path, max_dim)
        _probe_raster_read(paths.vh_path, max_dim)
    except Exception as exc:  # noqa: BLE001 - surface any GDAL/raster corruption reason.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def load_scene_metadata(paths: ScenePaths) -> SceneMetadata:
    import rasterio

    with rasterio.open(paths.vv_path) as vv_ds:
        return SceneMetadata(
            scene_id=paths.scene_id,
            transform=vv_ds.transform,
            crs=vv_ds.crs,
            height=vv_ds.height,
            width=vv_ds.width,
            vv_path=paths.vv_path,
            vh_path=paths.vh_path,
        )


def load_scene(paths: ScenePaths) -> SceneBundle:
    import rasterio

    metadata = load_scene_metadata(paths)
    with rasterio.open(paths.vv_path) as vv_ds:
        vv = vv_ds.read(1).astype(np.float32)

    with rasterio.open(paths.vh_path) as vh_ds:
        vh = vh_ds.read(1).astype(np.float32)

    image = np.stack([vv, vh], axis=0)
    image = np.nan_to_num(image, nan=-50.0, posinf=5.0, neginf=-50.0)

    return SceneBundle(
        scene_id=paths.scene_id,
        image=image,
        transform=metadata.transform,
        crs=metadata.crs,
        vv_path=paths.vv_path,
        vh_path=paths.vh_path,
    )


def normalize_sar(image: np.ndarray) -> np.ndarray:
    vv = np.clip(image[0], -40.0, 5.0)
    vh = np.clip(image[1], -45.0, 0.0)
    vv = (vv + 40.0) / 45.0
    vh = (vh + 45.0) / 45.0
    return np.stack([vv, vh], axis=0).astype(np.float32)


def normalize_channel(channel_name: str, raw_bands: dict[str, np.ndarray]) -> np.ndarray:
    if channel_name == "vv":
        vv = np.clip(raw_bands["vv"], -40.0, 5.0)
        return ((vv + 40.0) / 45.0).astype(np.float32)
    if channel_name == "vh":
        vh = np.clip(raw_bands["vh"], -45.0, 0.0)
        return ((vh + 45.0) / 45.0).astype(np.float32)
    if channel_name == "vv_minus_vh":
        diff = np.clip(raw_bands["vv"] - raw_bands["vh"], -5.0, 20.0)
        return ((diff + 5.0) / 25.0).astype(np.float32)
    if channel_name == "depth":
        bathymetry = raw_bands.get("bathymetry")
        if bathymetry is None:
            return np.zeros_like(raw_bands["vv"], dtype=np.float32)
        depth = np.clip(-bathymetry, 0.0, 5000.0)
        return (depth / 5000.0).astype(np.float32)
    if channel_name == "wind_speed":
        wind_speed = raw_bands.get("wind_speed")
        if wind_speed is None:
            return np.zeros_like(raw_bands["vv"], dtype=np.float32)
        wind_speed = np.clip(wind_speed, 0.0, 50.0)
        return (wind_speed / 50.0).astype(np.float32)
    if channel_name == "wind_quality":
        wind_quality = raw_bands.get("wind_quality")
        if wind_quality is None:
            return np.zeros_like(raw_bands["vv"], dtype=np.float32)
        wind_quality = np.clip(wind_quality, 0.0, 3.0)
        return (wind_quality / 3.0).astype(np.float32)
    if channel_name == "owi_mask":
        mask = raw_bands.get("owi_mask")
        if mask is None:
            return np.zeros_like(raw_bands["vv"], dtype=np.float32)
        mask = mask.astype(np.float32)
        finite = mask[np.isfinite(mask)]
        if finite.size == 0:
            return np.zeros_like(mask, dtype=np.float32)
        max_value = float(np.nanmax(finite))
        if max_value > 1.0:
            mask = mask / max_value
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported channel name: {channel_name}")


def build_feature_stack(
    raw_bands: dict[str, np.ndarray],
    channel_names: tuple[str, ...],
) -> np.ndarray:
    channels = [normalize_channel(channel_name, raw_bands) for channel_name in channel_names]
    return np.stack(channels, axis=0).astype(np.float32)


def extract_center_crop(image: np.ndarray, row: int, col: int, crop_size: int) -> np.ndarray:
    half = crop_size // 2

    row_start = row - half
    row_end = row_start + crop_size
    col_start = col - half
    col_end = col_start + crop_size

    pad_top = max(0, -row_start)
    pad_left = max(0, -col_start)
    pad_bottom = max(0, row_end - image.shape[1])
    pad_right = max(0, col_end - image.shape[2])

    row_start = max(0, row_start)
    row_end = min(image.shape[1], row_end)
    col_start = max(0, col_start)
    col_end = min(image.shape[2], col_end)

    crop = image[:, row_start:row_end, col_start:col_end]
    if any(value > 0 for value in (pad_top, pad_bottom, pad_left, pad_right)):
        crop = np.pad(
            crop,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=-50.0,
        )
    return crop.astype(np.float32)


def extract_center_crop_from_paths(
    paths: ScenePaths,
    row: int,
    col: int,
    crop_size: int,
    *,
    channel_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    import rasterio
    from rasterio.windows import Window

    half = crop_size // 2
    row_start = row - half
    col_start = col - half
    window = Window(col_start, row_start, crop_size, crop_size)

    with rasterio.open(paths.vv_path) as vv_ds:
        vv = vv_ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=-50.0,
        ).astype(np.float32)

    with rasterio.open(paths.vh_path) as vh_ds:
        vh = vh_ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=-50.0,
        ).astype(np.float32)

    raw_bands: dict[str, np.ndarray] = {
        "vv": np.nan_to_num(vv, nan=-50.0, posinf=5.0, neginf=-50.0).astype(np.float32),
        "vh": np.nan_to_num(vh, nan=-50.0, posinf=5.0, neginf=-50.0).astype(np.float32),
    }

    requested_optional = set()
    if channel_names is not None:
        for channel_name in channel_names:
            if channel_name == "depth":
                requested_optional.add("bathymetry")
            elif channel_name in {"owi_mask", "wind_speed", "wind_quality"}:
                requested_optional.add(channel_name)

    for optional_name in requested_optional:
        optional_path = paths.optional_paths.get(optional_name)
        if optional_path is None:
            continue
        with rasterio.open(optional_path) as ds:
            band = ds.read(
                1,
                window=window,
                boundless=True,
                fill_value=0.0,
            ).astype(np.float32)
        raw_bands[optional_name] = np.nan_to_num(
            band,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    if channel_names is None:
        crop = np.stack([raw_bands["vv"], raw_bands["vh"]], axis=0)
        return crop.astype(np.float32)
    return build_feature_stack(raw_bands, channel_names)


def read_scene_overview(
    paths: ScenePaths,
    max_dim: int = 2048,
) -> tuple[np.ndarray, SceneMetadata, float, float]:
    import rasterio
    from rasterio.enums import Resampling

    metadata = load_scene_metadata(paths)
    scale = max(metadata.height, metadata.width) / float(max_dim)
    scale = max(1.0, scale)
    out_height = max(1, int(round(metadata.height / scale)))
    out_width = max(1, int(round(metadata.width / scale)))

    with rasterio.open(paths.vv_path) as vv_ds:
        vv = vv_ds.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
        ).astype(np.float32)

    with rasterio.open(paths.vh_path) as vh_ds:
        vh = vh_ds.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
        ).astype(np.float32)

    image = np.stack([vv, vh], axis=0)
    image = np.nan_to_num(image, nan=-50.0, posinf=5.0, neginf=-50.0)

    row_scale = metadata.height / float(out_height)
    col_scale = metadata.width / float(out_width)
    return image.astype(np.float32), metadata, row_scale, col_scale


def pixel_to_latlon_from_metadata(
    metadata: SceneMetadata,
    row: int,
    col: int,
) -> tuple[float, float]:
    from pyproj import Transformer
    from rasterio.transform import xy

    x, y = xy(metadata.transform, row, col, offset="center")
    if metadata.crs is None:
        return float(y), float(x)

    crs_string = str(metadata.crs).upper()
    if "4326" in crs_string:
        return float(y), float(x)

    transformer = Transformer.from_crs(metadata.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def pixel_to_latlon(scene: SceneBundle, row: int, col: int) -> tuple[float, float]:
    metadata = SceneMetadata(
        scene_id=scene.scene_id,
        transform=scene.transform,
        crs=scene.crs,
        height=scene.image.shape[1],
        width=scene.image.shape[2],
        vv_path=scene.vv_path,
        vh_path=scene.vh_path,
    )
    return pixel_to_latlon_from_metadata(metadata, row, col)
