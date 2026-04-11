from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import FEATURE_ENGINEERING_DIR, TRAINING_DATASET_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_DIR = PROJECT_ROOT / "geo"

INPUT_PATH = TRAINING_DATASET_DIR / "05_training_dataset.csv"
OUTPUT_PATH = FEATURE_ENGINEERING_DIR / "06_feature_engineered_dataset.csv"
REPORT_PATH = FEATURE_ENGINEERING_DIR / "06_feature_engineering_report.json"

OUTPUT_COLUMNS = [
    "source_dataset",
    "source_row_id",
    "property_family",
    "surface_m2",
    "rooms",
    "price_tnd",
    "price_per_m2",
    "log_price_tnd",
    "log_price_per_m2",
    "normalized_locality",
    "geo_governorate",
    "geo_delegation",
    "lon",
    "lat",
]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_delegation_coords() -> dict[str, tuple[float, float]]:
    geojson_path = GEO_DIR / "delegations-full.geojson"
    with open(geojson_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    coords: dict[str, tuple[float, float]] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if props.get("engtype_2") != "Delegation" or not props.get("name_2"):
            continue

        geometry = feature.get("geometry", {})
        geometry_type = geometry.get("type")
        geometry_coords = geometry.get("coordinates") or []

        if geometry_type == "Polygon":
            rings = geometry_coords[:1]
        elif geometry_type == "MultiPolygon":
            rings = [polygon[0] for polygon in geometry_coords if polygon]
        else:
            rings = []

        all_lons: list[float] = []
        all_lats: list[float] = []
        for ring in rings:
            all_lons.extend(point[0] for point in ring)
            all_lats.extend(point[1] for point in ring)

        if all_lons and all_lats:
            coords[props["name_2"]] = (sum(all_lons) / len(all_lons), sum(all_lats) / len(all_lats))

    return coords


def prepare_feature_dataset(frame: pd.DataFrame, delegation_coords: dict[str, tuple[float, float]]) -> pd.DataFrame:
    feature_frame = frame.copy()
    feature_frame["surface_m2"] = pd.to_numeric(feature_frame["surface_m2"], errors="coerce")
    feature_frame["rooms"] = pd.to_numeric(feature_frame["rooms"], errors="coerce")
    feature_frame["price_tnd"] = pd.to_numeric(feature_frame["price_tnd"], errors="coerce")
    feature_frame["price_per_m2"] = pd.to_numeric(feature_frame["price_per_m2"], errors="coerce")
    feature_frame["log_price_tnd"] = pd.to_numeric(feature_frame["log_price_tnd"], errors="coerce")
    feature_frame["log_price_per_m2"] = pd.to_numeric(feature_frame["log_price_per_m2"], errors="coerce")

    feature_frame = feature_frame[
        feature_frame["surface_m2"].notna()
        & feature_frame["price_tnd"].notna()
        & feature_frame["property_family"].isin(["apartment", "house", "land"])
        & feature_frame["geo_governorate"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    feature_frame["lon"] = feature_frame["geo_delegation"].map(lambda value: delegation_coords.get(value, (None, None))[0])
    feature_frame["lat"] = feature_frame["geo_delegation"].map(lambda value: delegation_coords.get(value, (None, None))[1])

    return feature_frame[OUTPUT_COLUMNS].copy().reset_index(drop=True)


def add_split_training_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    frame_train: pd.DataFrame,
    frame_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_mean = frame_train["log_price_tnd"].mean()
    for column in ["geo_delegation", "geo_governorate"]:
        target_mean = frame_train.groupby(column)["log_price_tnd"].mean()
        x_train[f"{column}_target_enc"] = frame_train[column].map(target_mean).fillna(global_mean)
        x_test[f"{column}_target_enc"] = frame_test[column].map(target_mean).fillna(global_mean)

    delegation_price_median = frame_train.groupby("geo_delegation")["price_per_m2"].median()
    global_price_median = frame_train["price_per_m2"].median()

    x_train["price_vs_local_median"] = np.sqrt(
        frame_train["price_per_m2"]
        / frame_train["geo_delegation"].map(delegation_price_median).fillna(global_price_median)
    )
    x_test["price_vs_local_median"] = np.sqrt(
        frame_test["price_per_m2"]
        / frame_test["geo_delegation"].map(delegation_price_median).fillna(global_price_median)
    )

    x_train["price_vs_local_median"] = x_train["price_vs_local_median"].fillna(1.0).clip(0.5, 2.0)
    x_test["price_vs_local_median"] = x_test["price_vs_local_median"].fillna(1.0).clip(0.5, 2.0)
    return x_train, x_test


def build_report(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "feature_columns": ["surface_m2", "rooms", "lon", "lat"],
        "geo_delegation_count": int(frame["geo_delegation"].nunique()),
        "geo_governorate_count": int(frame["geo_governorate"].nunique()),
        "delegations_with_coords": int(frame["lon"].notna().sum()),
    }


def main() -> None:
    ensure_processed_dirs()

    frame = pd.read_csv(INPUT_PATH)
    delegation_coords = load_delegation_coords()
    feature_frame = prepare_feature_dataset(frame, delegation_coords)
    feature_frame.to_csv(OUTPUT_PATH, index=False)

    report = build_report(feature_frame)
    write_json(REPORT_PATH, report)

    summary = {
        "rows": report["rows"],
        "geo_delegation_count": report["geo_delegation_count"],
        "delegations_with_coords": report["delegations_with_coords"],
    }
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
