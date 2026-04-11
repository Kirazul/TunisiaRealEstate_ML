from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import FEATURE_ENGINEERING_DIR, VISUAL_CHECK_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = FEATURE_ENGINEERING_DIR / "06_feature_engineered_dataset.csv"

SAMPLE_COLUMNS = [
    "source_dataset",
    "source_row_id",
    "property_family",
    "surface_m2",
    "rooms",
    "price_tnd",
    "price_per_m2",
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


def source_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "family_distribution": frame["property_family"].value_counts().to_dict(),
        "governorate_count": int(frame["geo_governorate"].nunique()),
        "delegation_count": int(frame["geo_delegation"].nunique()),
        "locality_count": int(frame["normalized_locality"].nunique()),
        "median_price_tnd": float(frame["price_tnd"].median()),
        "median_surface_m2": float(frame["surface_m2"].median()),
        "median_price_per_m2": float(frame["price_per_m2"].median()),
        "top_delegations": frame["geo_delegation"].value_counts().head(10).to_dict(),
    }


def build_source_samples(frame: pd.DataFrame, sample_size: int = 12) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for source_name, source_frame in frame.groupby("source_dataset", sort=True):
        ordered = source_frame.sort_values(["geo_governorate", "geo_delegation", "property_family", "price_tnd", "surface_m2"]).copy()
        midpoint = len(ordered) // 2
        picked = pd.concat(
            [
                ordered.head(sample_size // 3),
                ordered.iloc[max(0, midpoint - sample_size // 6): midpoint + sample_size // 6],
                ordered.tail(sample_size // 3),
            ]
        ).drop_duplicates().head(sample_size)
        samples.append(picked[SAMPLE_COLUMNS])
    return pd.concat(samples, ignore_index=True)


def main() -> None:
    ensure_processed_dirs()

    frame = pd.read_csv(INPUT_PATH)

    report = {
        "stage": "07_training_dataset_visual_check",
        "dataset_rows": int(len(frame)),
        "sources": {
            source_name: source_summary(source_frame)
            for source_name, source_frame in frame.groupby("source_dataset", sort=True)
        },
    }

    samples = build_source_samples(frame)
    samples.to_csv(VISUAL_CHECK_DIR / "07_training_dataset_samples.csv", index=False)
    write_json(VISUAL_CHECK_DIR / "07_training_dataset_visual_report.json", report)

    summary = {
        "dataset_rows": int(len(frame)),
        "sample_rows_written": int(len(samples)),
        "sources": {key: {"rows": value["rows"], "median_price_tnd": value["median_price_tnd"]} for key, value in report["sources"].items()},
    }
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
