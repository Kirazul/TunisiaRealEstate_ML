from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from paths import DISCOVERY_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "tunisia_real_estate": RAW_DIR / "tunisia-real-estate.csv",
    "property_prices_in_tunisia": RAW_DIR / "Property Prices in Tunisia.csv",
    "data_prices_cleaned": RAW_DIR / "data_prices_cleaned.csv",
}

PRICE_COLUMN_CANDIDATES = ["price", "Price"]
SURFACE_COLUMN_CANDIDATES = ["size", "superficie", "Surface"]
LOCATION_COLUMN_CANDIDATES = ["city", "City", "region", "Region", "location", "Locality", "Delegation", "Governorate"]


def read_csv_any(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip", engine="python")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read {path}") from last_error


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def find_first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def object_empty_mask(series: pd.Series) -> bool:
    if series.isna().all():
        return True
    if series.dtype == object:
        return series.fillna("").map(normalize_text).eq("").all()
    return False


def top_values(frame: pd.DataFrame, column: str, limit: int = 10) -> list[dict[str, Any]]:
    series = frame[column].fillna("").map(normalize_text)
    counts = series[series.ne("")].value_counts().head(limit)
    return [{"value": idx, "count": int(val)} for idx, val in counts.items()]


def summarize_column(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    series = frame[column]
    non_null = int(series.notna().sum())
    empty_strings = int(series.fillna("").map(normalize_text).eq("").sum()) if series.dtype == object else 0
    summary: dict[str, Any] = {
        "dtype": str(series.dtype),
        "non_null": non_null,
        "null": int(series.isna().sum()),
        "empty_strings": empty_strings,
        "unique_non_null": int(series.dropna().nunique()),
        "all_empty": object_empty_mask(series),
        "sample_values": [normalize_text(value) for value in series.dropna().head(5).tolist()],
    }
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        summary["numeric_stats"] = {
            "min": float(numeric.min()),
            "median": float(numeric.median()),
            "max": float(numeric.max()),
            "non_numeric_rows": int(len(series) - numeric.notna().sum() - series.isna().sum()),
        }
    return summary


def dataset_specific_checks(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    if name == "tunisia_real_estate":
        nature = frame["Nature"].fillna("").map(normalize_text) if "Nature" in frame.columns else pd.Series(dtype=str)
        checks["nature_distribution"] = [{"value": idx, "count": int(val)} for idx, val in nature.value_counts().head(10).items()]
        checks["non_sale_rows"] = int(nature.str.lower().isin(["rental", "vacation rental", "offices & shops"]).sum())

    if name == "property_prices_in_tunisia":
        listing_type = frame["type"].fillna("").map(normalize_text) if "type" in frame.columns else pd.Series(dtype=str)
        category = frame["category"].fillna("").map(normalize_text) if "category" in frame.columns else pd.Series(dtype=str)
        checks["type_distribution"] = [{"value": idx, "count": int(val)} for idx, val in listing_type.value_counts().head(10).items()]
        checks["category_distribution"] = [{"value": idx, "count": int(val)} for idx, val in category.value_counts().head(10).items()]
        checks["autres_villes_rows"] = int(frame["region"].fillna("").map(normalize_text).str.lower().eq("autres villes").sum()) if "region" in frame.columns else 0
        checks["negative_size_rows"] = int(pd.to_numeric(frame.get("size"), errors="coerce").lt(0).sum()) if "size" in frame.columns else 0
        checks["negative_room_rows"] = int(pd.to_numeric(frame.get("room_count"), errors="coerce").lt(0).sum()) if "room_count" in frame.columns else 0

    if name == "data_prices_cleaned":
        transaction = frame["transaction"].fillna("").map(normalize_text) if "transaction" in frame.columns else pd.Series(dtype=str)
        category = frame["category"].fillna("").map(normalize_text) if "category" in frame.columns else pd.Series(dtype=str)
        checks["transaction_distribution"] = [{"value": idx, "count": int(val)} for idx, val in transaction.value_counts().head(10).items()]
        checks["category_distribution"] = [{"value": idx, "count": int(val)} for idx, val in category.value_counts().head(10).items()]
        checks["currency_distribution"] = [{"value": idx, "count": int(val)} for idx, val in frame["currency"].fillna("").map(normalize_text).value_counts().head(10).items()] if "currency" in frame.columns else []

    return checks


def profile_dataset(name: str, path: Path) -> dict[str, Any]:
    frame = read_csv_any(path)
    price_col = find_first_column(frame, PRICE_COLUMN_CANDIDATES)
    surface_col = find_first_column(frame, SURFACE_COLUMN_CANDIDATES)
    location_col = find_first_column(frame, LOCATION_COLUMN_CANDIDATES)

    price_series = numeric_series(frame, price_col)
    surface_series = numeric_series(frame, surface_col)

    all_empty_columns = [column for column in frame.columns if object_empty_mask(frame[column])]
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_rows_without_date = 0
    if "date" in frame.columns:
        subset = [column for column in frame.columns if column != "date"]
        duplicate_rows_without_date = int(frame.duplicated(subset=subset).sum())

    profile = {
        "dataset_name": name,
        "file_name": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "column_count": int(len(frame.columns)),
        "all_empty_columns": all_empty_columns,
        "duplicate_rows_exact": duplicate_rows,
        "duplicate_rows_without_date": duplicate_rows_without_date,
        "price_column": price_col,
        "surface_column": surface_col,
        "location_column": location_col,
        "price_missing": int(price_series.isna().sum()) if not price_series.empty else None,
        "surface_missing": int(surface_series.isna().sum()) if not surface_series.empty else None,
        "price_non_positive": int(price_series.le(0).sum()) if not price_series.empty else None,
        "surface_non_positive": int(surface_series.le(0).sum()) if not surface_series.empty else None,
        "top_location_values": top_values(frame, location_col) if location_col else [],
        "columns_summary": {column: summarize_column(frame, column) for column in frame.columns},
        "dataset_specific_checks": dataset_specific_checks(name, frame),
    }
    return profile


def build_merge_overview(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "01_dataset_discovery",
        "datasets": [
            {
                "dataset_name": profile["dataset_name"],
                "rows": profile["rows"],
                "column_count": profile["column_count"],
                "all_empty_columns": profile["all_empty_columns"],
                "price_column": profile["price_column"],
                "surface_column": profile["surface_column"],
                "location_column": profile["location_column"],
                "price_missing": profile["price_missing"],
                "surface_missing": profile["surface_missing"],
                "price_non_positive": profile["price_non_positive"],
                "surface_non_positive": profile["surface_non_positive"],
            }
            for profile in profiles
        ],
        "merge_risks": [
            "Column names and semantics differ across all three sources.",
            "Sale, rent, and non-residential records are mixed in some raw datasets.",
            "Some datasets use placeholder negatives like -1 for missing values.",
            "Location granularity differs between governorate, delegation, city, and free-text location.",
            "Empty-only columns should be dropped before dataset-specific cleaning outputs are saved.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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


def main() -> None:
    ensure_processed_dirs()

    profiles: list[dict[str, Any]] = []
    for dataset_name, path in DATASETS.items():
        profile = profile_dataset(dataset_name, path)
        profiles.append(profile)
        output_path = DISCOVERY_DIR / f"01_{dataset_name}_discovery.json"
        write_json(output_path, profile)

    merge_overview = build_merge_overview(profiles)
    write_json(DISCOVERY_DIR / "01_merge_overview.json", merge_overview)

    summary = {
        profile["dataset_name"]: {
            "rows": profile["rows"],
            "all_empty_columns": profile["all_empty_columns"],
            "price_missing": profile["price_missing"],
            "surface_missing": profile["surface_missing"],
        }
        for profile in profiles
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
