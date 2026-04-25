import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib
import warnings

from rasterio.transform import rowcol
from pyproj import Transformer

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


def normalize_sar(array):
    epsilon = 1e-10
    db = 10 * np.log10(array + epsilon)
    db_clipped = np.clip(db, -30, 0)
    normalized = (db_clipped - (-30)) / (0 - (-30))

    return normalized.astype(np.float32)


train_raw = pd.read_csv(TRAIN_CSV)
val_raw = pd.read_csv(VAL_CSV)

# list of every subfolder of the boats folder inside data
tiny_scene_ids = [
    folder
    for folder in os.listdir(TINY_DIR)
    if os.path.isdir(os.path.join(TINY_DIR, folder))
]

# get the folders directories aswell as the files in each folder
for sid in tiny_scene_ids:
    scene_path = os.path.join(TINY_DIR, sid)
    files = os.listdir(scene_path)

# since our csv files have 500+ rows of data which is intended for the larger datasets... we have to shrink them down to match our small dataset
train_tiny = train_raw[train_raw["scene_id"].isin(tiny_scene_ids)].copy()
val_tiny = val_raw[val_raw["scene_id"].isin(tiny_scene_ids)].copy()


def clean_labels(df: pd.DataFrame, name="dataframe"):
    df = df[df["is_vessel"] == True]
    df = df.dropna(subset=["detect_lat", "detect_lon"])
    df["vessel_length_m"] = df["vessel_length_m"].fillna(0.0)

    df = df.reset_index(drop=True)
    return df


train_tiny = clean_labels(train_tiny, name="train")
val_tiny = clean_labels(val_tiny, name="validation")

# check whether this ship is dark or not
train_tiny["is_dark"] = train_tiny["source"].str.upper().str.strip() != "AIS"
val_tiny["is_dark"] = val_tiny["source"].str.upper().str.strip() != "AIS"


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


print(train_tiny["source"].unique())
print(train_raw["source"].value_counts())


CHIP_SIZE = 64


def extract_and_save_chips(df: pd.DataFrame, tiny_dir, output_chip_dir, split_name):
    chip_manifest = []
    raster_cache = {}

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
                    }

            cached = raster_cache[cache_key]
            array = cached["data"]
            transform = cached["transform"]

            r, c = rowcol(transform, row["detect_lon"], row["detect_lat"])
            chip = extract_chip(array, r, c, chip_size=CHIP_SIZE)
            chip = normalize_sar(chip)
            chips[band] = chip

            if not valid:
                continue

            stacked_chip = np.stack([chips["VV_dB"], chips["VV_dB"]], axis=0)

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

            del raster_cache

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
