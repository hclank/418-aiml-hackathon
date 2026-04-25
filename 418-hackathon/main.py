import os
import numpy as np
import pandas as pd
import rasterio
import terratorch
import matplotlib.pyplot as plt
import matplotlib
import warnings

from rasterio.transform import rowcol
from pyproj import Transformer

try:
    from terratorch.models import TerraMind
except ImportError:
    TerraMind = None

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

TINY_DIR = "data/boats"
VAL_CSV = "data/validation.csv"
TRAIN_CSV = "data/train.csv"
OUTPUT_DIR = "./xview3_preprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chips"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chips", "VV"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chips", "VH"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

LOAD_TERRAMIND = os.environ.get("LOAD_TERRAMIND", "0") == "1"
model = None
if LOAD_TERRAMIND and TerraMind is not None:
    model = TerraMind.from_pretrained("ibm-nasa-geospatial/terramind-1.0-base")


def sample_raster_at_point(tif_path, lat, lon):
    if not os.path.exists(tif_path):
        return np.nan

    try:
        with rasterio.open(tif_path) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = rowcol(src.transform, x, y)

            height, width = src.height, src.width
            if row < 0 or col < 0 or row >= height or col >= width:
                return np.nan

            val = float(src.read(1)[row, col])

            if src.nodata is not None and val == src.nodata:
                return np.nan

            return val

    except Exception:
        return np.nan


# crops out a square chip of size (chip_size x chip_size)
def extract_chip(array, row, col, chip_size=64):
    # Clamp to valid pixel indices so slicing never produces oversized chips.
    row = int(np.clip(row, 0, array.shape[0] - 1))
    col = int(np.clip(col, 0, array.shape[1] - 1))
    half = chip_size // 2

    r0 = max(0, row - half)
    r1 = min(array.shape[0], row + half)
    c0 = max(0, col - half)
    c1 = min(array.shape[1], col + half)

    chip = array[r0:r1, c0:c1]  # actual crop from the array

    pad_r = chip_size - chip.shape[0]
    pad_c = chip_size - chip.shape[1]

    if pad_r > 0 or pad_c > 0:
        chip = np.pad(chip, ((0, pad_r), (0, pad_c)), mode="reflect")

    return chip


def normalize_sar_db(array):
    arr = np.nan_to_num(array, nan=-30.0, neginf=-30.0, posinf=0.0).astype(np.float32)
    arr = np.clip(arr, -30.0, 0.0)
    return (arr + 30.0) / 30.0


train_raw = pd.read_csv(TRAIN_CSV)
val_raw = pd.read_csv(VAL_CSV)

# list of every subfolder of the boats folder inside data
tiny_scene_ids = [
    folder
    for folder in os.listdir(TINY_DIR)
    if os.path.isdir(os.path.join(TINY_DIR, folder))
]

def clean_labels(df: pd.DataFrame, name="dataframe"):
    df = df[df["is_vessel"] == True]
    df = df.dropna(subset=["detect_lat", "detect_lon"])
    df["vessel_length_m"] = df["vessel_length_m"].fillna(0.0)

    df = df.reset_index(drop=True)
    return df


def split_scenes_with_both_classes(df, val_frac=0.3, seed=42, max_tries=5000):
    scenes = df["scene_id"].drop_duplicates().to_numpy()
    n_val = max(1, int(round(len(scenes) * val_frac)))
    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        perm = rng.permutation(scenes)
        val_scenes = set(perm[:n_val])
        train_scenes = set(perm[n_val:])

        train_df = df[df["scene_id"].isin(train_scenes)].copy()
        val_df = df[df["scene_id"].isin(val_scenes)].copy()

        # Enforce class presence in both splits and avoid empty splits.
        if (
            len(train_df) > 0
            and len(val_df) > 0
            and train_df["is_dark"].nunique() == 2
            and val_df["is_dark"].nunique() == 2
        ):
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    raise RuntimeError("Could not find a scene split with both classes in train and val")


# Build tiny split from all available labels, then re-split by scene.
all_raw = pd.concat([train_raw, val_raw], ignore_index=True)
tiny_all = all_raw[all_raw["scene_id"].isin(tiny_scene_ids)].copy()
tiny_all = clean_labels(tiny_all, name="tiny_all")
tiny_all["is_dark"] = tiny_all["source"].str.upper().str.strip() != "AIS"

train_tiny, val_tiny = split_scenes_with_both_classes(tiny_all, val_frac=0.3, seed=42)

split_dir = os.path.join(OUTPUT_DIR, "splits")
os.makedirs(split_dir, exist_ok=True)
train_tiny.to_csv(os.path.join(split_dir, "train_split.csv"), index=False)
val_tiny.to_csv(os.path.join(split_dir, "val_split.csv"), index=False)


AUXILIARY_RASTERS = {
    "bathymetry": "bathymetry",
    "wind_speed": "owiWindSpeed",
    "wind_direction": "owiWindDirection",
}


def extract_features_from_df(df: pd.DataFrame, tiny_dir, split_name="data"):
    feature_records = []
    for i, row in df.iterrows():
        scene_dir = os.path.join(tiny_dir, row["scene_id"])
        record = {
            "scene_id": row["scene_id"],
            "detect_lat": row["detect_lat"],
            "detect_lon": row["detect_lon"],
            "vessel_length_m": row["vessel_length_m"],
            "is_dark": row["is_dark"],
            "is_fishing": row.get("is_fishing", False),
        }

        for raster_name in AUXILIARY_RASTERS:
            tif_path = os.path.join(scene_dir, f"{AUXILIARY_RASTERS[raster_name]}.tif")
            record[raster_name] = sample_raster_at_point(
                tif_path, row["detect_lat"], row["detect_lon"]
            )

        bathy = record["bathymetry"]
        record["near_shore"] = int(bathy > -50) if not np.isnan(bathy) else 0
        record["shallowness"] = max(0, bathy + 200) if not np.isnan(bathy) else 0

        feature_records.append(record)
    return pd.DataFrame(feature_records)


train_features = extract_features_from_df(train_tiny, TINY_DIR, "train")
val_features = extract_features_from_df(val_tiny, TINY_DIR, "val")
train_features.to_csv(os.path.join(OUTPUT_DIR, "train_features.csv"), index=False)
val_features.to_csv(os.path.join(OUTPUT_DIR, "val_features.csv"), index=False)


print("Train class counts:\n", train_tiny["is_dark"].value_counts())
print("Val class counts:\n", val_tiny["is_dark"].value_counts())
print("Train scenes:", train_tiny["scene_id"].nunique())
print("Val scenes:", val_tiny["scene_id"].nunique())


CHIP_SIZE = 64


def extract_and_save_chips(df: pd.DataFrame, tiny_dir, output_chip_dir, split_name):
    chip_manifest = []
    raster_cache = {}
    transformer_cache = {}

    for i, row in df.iterrows():
        scene_id = row["scene_id"]
        scene_dir = os.path.join(tiny_dir, scene_id)

        chips = {}
        valid = True

        for band in ["VV_dB", "VH_dB"]:
            tif_path = os.path.join(scene_dir, f"{band}.tif")

            if not os.path.exists(tif_path):
                valid = False
                break

            cache_key = f"{scene_id}_{band}"
            if cache_key not in raster_cache:
                with rasterio.open(tif_path) as src:
                    raster_cache[cache_key] = {
                        "data": src.read(1).astype(np.float32),
                        "transform": src.transform,
                        "crs": src.crs,
                        "height": src.height,
                        "width": src.width,
                    }

            cached = raster_cache[cache_key]
            array = cached["data"]
            transform = cached["transform"]
            raster_crs = cached["crs"]
            crs_key = str(raster_crs)

            if crs_key not in transformer_cache:
                transformer_cache[crs_key] = Transformer.from_crs(
                    "EPSG:4326", raster_crs, always_xy=True
                )

            x, y = transformer_cache[crs_key].transform(
                row["detect_lon"], row["detect_lat"]
            )
            r, c = rowcol(transform, x, y)

            if r < 0 or c < 0 or r >= cached["height"] or c >= cached["width"]:
                valid = False
                break

            chip = extract_chip(array, r, c, chip_size=CHIP_SIZE)
            chip = normalize_sar_db(chip)
            chips[band] = chip

        if not valid:
            continue

        stacked_chip = np.stack([chips["VV_dB"], chips["VH_dB"]], axis=0)

        lat_str = f"{row['detect_lat']:.4f}".replace(".", "p").replace("-", "n")
        lon_str = f"{row['detect_lon']:.4f}".replace(".", "p").replace("-", "n")
        chip_filename = f"{scene_id}__{lat_str}_{lon_str}.npy"
        chip_path = os.path.join(output_chip_dir, chip_filename)

        np.save(chip_path, stacked_chip)

        chip_manifest.append(
            {
                "chip_path": chip_path,
                "scene_id": scene_id,
                "detect_lat": row["detect_lat"],
                "detect_lon": row["detect_lon"],
                "is_dark": row["is_dark"],
            }
        )

    return pd.DataFrame(chip_manifest)


chip_dir = os.path.join(OUTPUT_DIR, "chips")
train_manifest = extract_and_save_chips(train_tiny, TINY_DIR, chip_dir, "train")
val_manifest = extract_and_save_chips(val_tiny, TINY_DIR, chip_dir, "val")

print(f"\nChips saved: {len(train_manifest)} train, {len(val_manifest)} val")
print(
    f"Chip shape: {np.load(train_manifest.iloc[0]['chip_path']).shape}"
)  # should be (2, 64, 64)

train_manifest.to_csv(os.path.join(OUTPUT_DIR, "train_chip_manifest.csv"), index=False)
val_manifest.to_csv(os.path.join(OUTPUT_DIR, "val_chip_manifest.csv"), index=False)
print(f"Saved chip manifests → {OUTPUT_DIR}/")
