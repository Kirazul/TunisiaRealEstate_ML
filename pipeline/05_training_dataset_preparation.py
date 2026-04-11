from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import GEO_ALIGNMENT_DIR, TRAINING_DATASET_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = GEO_ALIGNMENT_DIR / "04_geo_aligned_dataset.csv"

TRAINING_COLUMNS = [
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


def prepare_training_dataset() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH)

    training = frame.copy()
    training["surface_m2"] = pd.to_numeric(training["surface_m2"], errors="coerce")
    training["rooms"] = pd.to_numeric(training["rooms"], errors="coerce")
    training["price_tnd"] = pd.to_numeric(training["price_tnd"], errors="coerce")

    training = training[
        training["property_family"].isin(["apartment", "house", "land"])
        & training["geo_governorate"].fillna("").astype(str).str.strip().ne("")
        & training["normalized_locality"].fillna("").astype(str).str.strip().ne("")
        & training["surface_m2"].notna()
        & training["price_tnd"].notna()
    ].copy()

    training = training[(training["surface_m2"] > 0) & (training["price_tnd"] > 0)].copy()

    training["price_per_m2"] = training["price_tnd"] / training["surface_m2"]
    training["log_price_tnd"] = np.log(training["price_tnd"])
    training["log_price_per_m2"] = np.log(training["price_per_m2"])

    training.loc[training["property_family"] == "land", "rooms"] = 0
    training = training[TRAINING_COLUMNS].copy()
    training = training.drop_duplicates(subset=["source_dataset", "source_row_id"]).reset_index(drop=True)
    return training


def validate_training_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "all_empty_columns": [
            column
            for column in frame.columns
            if frame[column].isna().all()
            or (frame[column].dtype == object and frame[column].fillna("").astype(str).str.strip().eq("").all())
        ],
        "duplicate_rows": int(frame.duplicated().sum()),
        "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "source_distribution": frame["source_dataset"].value_counts().to_dict(),
        "family_distribution": frame["property_family"].value_counts().to_dict(),
        "geo_governorate_count": int(frame["geo_governorate"].nunique()),
        "geo_delegation_count": int(frame["geo_delegation"].nunique()),
        "normalized_locality_count": int(frame["normalized_locality"].nunique()),
        "price_range": {
            "min": float(frame["price_tnd"].min()),
            "median": float(frame["price_tnd"].median()),
            "max": float(frame["price_tnd"].max()),
        },
        "surface_range": {
            "min": float(frame["surface_m2"].min()),
            "median": float(frame["surface_m2"].median()),
            "max": float(frame["surface_m2"].max()),
        },
        "price_per_m2_range": {
            "min": float(frame["price_per_m2"].min()),
            "median": float(frame["price_per_m2"].median()),
            "max": float(frame["price_per_m2"].max()),
        },
    }


def main() -> None:
    ensure_processed_dirs()

    training = prepare_training_dataset()
    training.to_csv(TRAINING_DATASET_DIR / "05_training_dataset.csv", index=False)

    report = validate_training_dataset(training)
    write_json(TRAINING_DATASET_DIR / "05_training_dataset_report.json", report)

    summary = {
        "rows": report["rows"],
        "duplicate_rows": report["duplicate_rows"],
        "all_empty_columns": report["all_empty_columns"],
        "geo_governorate_count": report["geo_governorate_count"],
        "geo_delegation_count": report["geo_delegation_count"],
        "normalized_locality_count": report["normalized_locality_count"],
    }
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
