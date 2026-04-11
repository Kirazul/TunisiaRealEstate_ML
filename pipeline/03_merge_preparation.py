from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import CLEANING_DIR, MERGE_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "tunisia_real_estate": CLEANING_DIR / "02_tunisia_real_estate_clean.csv",
    "property_prices_in_tunisia": CLEANING_DIR / "02_property_prices_in_tunisia_clean.csv",
    "data_prices_cleaned": CLEANING_DIR / "02_data_prices_cleaned_clean.csv",
}

SOURCE_PRIORITY = {
    "tunisia_real_estate": 1,
    "property_prices_in_tunisia": 2,
    "data_prices_cleaned": 3,
}

CORE_COLUMNS = [
    "source_dataset",
    "source_row_id",
    "property_family",
    "governorate",
    "delegation",
    "city",
    "location",
    "surface_m2",
    "rooms",
    "bathrooms",
    "price_tnd",
]

DEDUP_COLUMNS = [
    "property_family",
    "governorate",
    "city",
    "surface_m2",
    "price_tnd",
    "rooms",
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


def read_clean_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["source_dataset"] = dataset_name
    return frame[CORE_COLUMNS].copy()


def load_clean_datasets() -> pd.DataFrame:
    frames = [read_clean_dataset(path, dataset_name) for dataset_name, path in INPUT_FILES.items()]
    merged = pd.concat(frames, ignore_index=True)
    merged["source_priority"] = merged["source_dataset"].map(SOURCE_PRIORITY).fillna(999).astype(int)
    return merged


def build_duplicate_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    duplicate_mask = frame.duplicated(subset=DEDUP_COLUMNS, keep=False)
    duplicates = frame[duplicate_mask].copy().sort_values(DEDUP_COLUMNS + ["source_priority", "source_dataset"])

    groups: list[dict[str, Any]] = []
    if not duplicates.empty:
        grouped = duplicates.groupby(DEDUP_COLUMNS, dropna=False)
        for keys, group in grouped:
            key_map = {column: value for column, value in zip(DEDUP_COLUMNS, keys)}
            groups.append(
                {
                    "duplicate_key": key_map,
                    "count": int(len(group)),
                    "sources": sorted(group["source_dataset"].astype(str).unique().tolist()),
                    "rows": group[["source_dataset", "source_row_id", "delegation", "location", "bathrooms"]].to_dict("records"),
                }
            )

    report = {
        "duplicate_rows": int(len(duplicates)),
        "duplicate_groups": int(len(groups)),
        "sample_groups": groups[:50],
    }
    return duplicates, report


def deduplicate_merged_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(DEDUP_COLUMNS + ["source_priority", "source_dataset", "source_row_id"]).copy()
    deduped = ordered.drop_duplicates(subset=DEDUP_COLUMNS, keep="first").copy()
    return deduped.reset_index(drop=True)


def validate_merged_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    ppm = frame["price_tnd"] / frame["surface_m2"]
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "all_empty_columns": [
            column
            for column in frame.columns
            if frame[column].isna().all()
            or (frame[column].dtype == object and frame[column].fillna("").astype(str).str.strip().eq("").all())
        ],
        "duplicate_rows": int(frame.duplicated(subset=DEDUP_COLUMNS).sum()),
        "source_distribution": frame["source_dataset"].value_counts().to_dict(),
        "family_distribution": frame["property_family"].value_counts().to_dict(),
        "governorate_count": int(frame["governorate"].nunique()),
        "city_count": int(frame["city"].nunique()),
        "surface_range": {
            "min": float(frame["surface_m2"].min()),
            "max": float(frame["surface_m2"].max()),
        },
        "price_range": {
            "min": float(frame["price_tnd"].min()),
            "max": float(frame["price_tnd"].max()),
        },
        "price_per_m2_range": {
            "min": float(ppm.min()),
            "max": float(ppm.max()),
        },
    }


def main() -> None:
    ensure_processed_dirs()

    merged_raw = load_clean_datasets()
    merged_raw.to_csv(MERGE_DIR / "03_merged_before_dedup.csv", index=False)

    duplicate_rows, duplicate_report = build_duplicate_report(merged_raw)
    if not duplicate_rows.empty:
        duplicate_rows.to_csv(MERGE_DIR / "03_cross_source_duplicates.csv", index=False)

    merged_final = deduplicate_merged_dataset(merged_raw)
    merged_final = merged_final.drop(columns=["source_priority"])
    merged_final.to_csv(MERGE_DIR / "03_final_merge_ready.csv", index=False)

    merge_report = {
        "stage": "03_merge_preparation",
        "input_rows": int(len(merged_raw)),
        "rows_removed_as_cross_source_duplicates": int(len(merged_raw) - len(merged_final)),
        "final_rows": int(len(merged_final)),
        "source_priority": SOURCE_PRIORITY,
        "duplicate_report": duplicate_report,
    }
    validation_report = validate_merged_dataset(merged_final)

    write_json(MERGE_DIR / "03_merge_report.json", merge_report)
    write_json(MERGE_DIR / "03_final_merge_validation.json", validation_report)

    summary = {
        "input_rows": int(len(merged_raw)),
        "rows_removed_as_cross_source_duplicates": int(len(merged_raw) - len(merged_final)),
        "final_rows": int(len(merged_final)),
        "duplicate_groups": duplicate_report["duplicate_groups"],
        "source_distribution": validation_report["source_distribution"],
    }
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
