from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import CLEANING_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "tunisia_real_estate": RAW_DIR / "tunisia-real-estate.csv",
    "property_prices_in_tunisia": RAW_DIR / "Property Prices in Tunisia.csv",
    "data_prices_cleaned": RAW_DIR / "data_prices_cleaned.csv",
}

CANONICAL_COLUMNS = [
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

ALLOWED_FAMILIES = {"apartment", "house", "land"}

CANONICAL_GOVERNORATES = {
    "ariana": "Ariana",
    "beja": "Beja",
    "ben arous": "Ben Arous",
    "bizerte": "Bizerte",
    "gabes": "Gabes",
    "gafsa": "Gafsa",
    "jendouba": "Jendouba",
    "kairouan": "Kairouan",
    "kasserine": "Kasserine",
    "kebili": "Kebili",
    "kef": "Le Kef",
    "le kef": "Le Kef",
    "mahdia": "Mahdia",
    "manouba": "Manouba",
    "medenine": "Medenine",
    "monastir": "Monastir",
    "nabeul": "Nabeul",
    "sfax": "Sfax",
    "sidi bouzid": "Sidi Bouzid",
    "siliana": "Siliana",
    "sousse": "Sousse",
    "tataouine": "Tataouine",
    "tozeur": "Tozeur",
    "tunis": "Tunis",
    "zaghouan": "Zaghouan",
}

GOVERNORATE_ALIASES = {
    "ben arous": "Ben Arous",
    "ben arouss": "Ben Arous",
    "la manouba": "Manouba",
    "manouba": "Manouba",
    "mannouba": "Manouba",
    "mdenine": "Medenine",
    "medenine": "Medenine",
    "gabes": "Gabes",
    "gabs": "Gabes",
    "beja": "Beja",
    "bja": "Beja",
    "le kef": "Le Kef",
    "kef": "Le Kef",
    "sidi bouzid": "Sidi Bouzid",
    "sidi bouzid governorate": "Sidi Bouzid",
}

CITY_ALIASES = {
    "l aouina": "L'Aouina",
    "l'aouina": "L'Aouina",
    "laouina": "L'Aouina",
    "klibia": "Kelibia",
    "kelibia": "Kelibia",
    "chatt mariem": "Chott Mariem",
    "chot mariem": "Chott Mariem",
    "cit ennasr 2": "Cite Ennasr 2",
    "cite ennasr 2": "Cite Ennasr 2",
    "ennasr": "Ennasr",
    "khzema": "Khzema",
    "khzema est": "Khzema Est",
    "khzema ouest": "Khzema Ouest",
    "sahloul": "Sahloul",
    "mnihla": "Mnihla",
    "mannouba": "Manouba",
    "manouba": "Manouba",
    "bni khiar": "Beni Khiar",
    "rades": "Rades",
    "kala kebira": "Kalaa Kebira",
    "kalaa kebira": "Kalaa Kebira",
    "medenine": "Medenine",
    "sayada lamta bou hajar": "Sayada Lamta Bou Hajar",
    "djerba houmt souk": "Djerba Houmt Souk",
    "les jardins el menzah 1": "Les Jardins El Menzah 1",
    "les jardins el menzah 2": "Les Jardins El Menzah 2",
    "autres villes": "",
    "ain zaghouan": "Ain Zaghouan",
    "la marsa": "La Marsa",
    "la soukra": "La Soukra",
    "hammam sousse": "Hammam Sousse",
    "hammamet": "Hammamet",
    "raoued": "Raoued",
    "boumhel": "Boumhel",
    "ezzahra": "Ezzahra",
    "menzah": "El Menzah",
}


def read_csv_any(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip", engine="python")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read {path}") from last_error


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


def repair_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if any(token in text for token in ("Ã", "Â", "â", "�")):
        for source_encoding in ("latin-1", "cp1252"):
            try:
                repaired = text.encode(source_encoding).decode("utf-8")
            except Exception:
                continue
            if repaired:
                text = repaired
                break
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def normalize_key(value: Any) -> str:
    text = repair_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_case_text(value: Any) -> str:
    text = repair_text(value)
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def parse_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = repair_text(value)
    if not text:
        return None
    text = text.replace("TND", "").replace("DT", "").replace("€", "")
    text = text.replace(",", ".")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[0])


def parse_int(value: Any) -> int | None:
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def infer_property_family(*values: Any) -> str:
    blob = " ".join(normalize_key(value) for value in values if repair_text(value))
    if any(token in blob for token in ("terrain", "land", "ferme", "vacant land", "lot")):
        return "land"
    if any(token in blob for token in ("villa", "maison", "house", "duplex", "immeuble", "houses")):
        return "house"
    if any(token in blob for token in ("appart", "apartment", "studio", "flat", "residence", "s 1", "s 2", "s 3", "s 4", "room apartment")):
        return "apartment"
    return "other"


def clean_governorate(value: Any) -> str:
    raw = repair_text(value)
    key = normalize_key(raw)
    if not key:
        return ""
    if key in GOVERNORATE_ALIASES:
        return GOVERNORATE_ALIASES[key]
    if key in CANONICAL_GOVERNORATES:
        return CANONICAL_GOVERNORATES[key]
    return title_case_text(raw)


def clean_city(value: Any) -> str:
    raw = repair_text(value)
    key = normalize_key(raw)
    if not key:
        return ""
    if key in CITY_ALIASES:
        return CITY_ALIASES[key]
    return title_case_text(raw)


def ensure_positive_float(value: Any) -> float | None:
    numeric = parse_float(value)
    if numeric is None or numeric <= 0:
        return None
    return float(numeric)


def ensure_non_negative_int(value: Any) -> int | None:
    numeric = parse_int(value)
    if numeric is None or numeric < 0:
        return None
    return int(numeric)


def finalize_clean_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = None

    cleaned = cleaned[CANONICAL_COLUMNS].copy()

    for column in ["governorate", "delegation", "city", "location", "property_family"]:
        cleaned[column] = cleaned[column].map(repair_text)

    cleaned["surface_m2"] = pd.to_numeric(cleaned["surface_m2"], errors="coerce")
    cleaned["rooms"] = pd.to_numeric(cleaned["rooms"], errors="coerce")
    cleaned["bathrooms"] = pd.to_numeric(cleaned["bathrooms"], errors="coerce")
    cleaned["price_tnd"] = pd.to_numeric(cleaned["price_tnd"], errors="coerce")

    cleaned.loc[(cleaned["rooms"] < 0) | (cleaned["rooms"] > 20), "rooms"] = np.nan
    cleaned.loc[(cleaned["bathrooms"] < 0) | (cleaned["bathrooms"] > 10), "bathrooms"] = np.nan

    cleaned.loc[cleaned["property_family"] == "land", "rooms"] = 0
    cleaned.loc[cleaned["property_family"] == "land", "bathrooms"] = 0

    cleaned["price_per_m2"] = cleaned["price_tnd"] / cleaned["surface_m2"]

    apartment_mask = cleaned["property_family"] == "apartment"
    house_mask = cleaned["property_family"] == "house"
    land_mask = cleaned["property_family"] == "land"

    cleaned = cleaned[
        (~apartment_mask | (cleaned["surface_m2"].between(15, 500) & cleaned["price_tnd"].between(30000, 5000000)))
        & (~house_mask | (cleaned["surface_m2"].between(40, 2000) & cleaned["price_tnd"].between(50000, 30000000)))
        & (~land_mask | (cleaned["surface_m2"].between(50, 50000) & cleaned["price_tnd"].between(2000, 50000000)))
    ].copy()

    apartment_mask = cleaned["property_family"] == "apartment"
    house_mask = cleaned["property_family"] == "house"
    land_mask = cleaned["property_family"] == "land"

    cleaned = cleaned[
        (~apartment_mask | cleaned["price_per_m2"].between(300, 20000))
        & (~house_mask | cleaned["price_per_m2"].between(300, 20000))
        & (~land_mask | cleaned["price_per_m2"].between(10, 15000))
    ].copy()

    # Keep only training-usable rows.
    cleaned = cleaned[
        cleaned["property_family"].isin(ALLOWED_FAMILIES)
        & cleaned["governorate"].ne("")
        & cleaned["city"].ne("")
        & cleaned["surface_m2"].notna()
        & cleaned["price_tnd"].notna()
    ].copy()

    cleaned = cleaned[(cleaned["surface_m2"] > 0) & (cleaned["price_tnd"] > 0)].copy()

    cleaned = cleaned.drop_duplicates(subset=["property_family", "governorate", "city", "surface_m2", "price_tnd", "rooms"])
    cleaned = cleaned.drop(columns=["price_per_m2"])
    cleaned = cleaned.reset_index(drop=True)
    cleaned["source_dataset"] = name
    cleaned["source_row_id"] = cleaned["source_row_id"].astype(str)
    return cleaned


def clean_tunisia_real_estate(path: Path) -> pd.DataFrame:
    raw = read_csv_any(path).reset_index().rename(columns={"index": "source_row_id"})
    nature = raw["Nature"].map(repair_text)
    family = raw.apply(lambda row: infer_property_family(row.get("Type of Real Estate"), row.get("Nature")), axis=1)

    cleaned = pd.DataFrame(
        {
            "source_row_id": raw["source_row_id"],
            "property_family": family,
            "governorate": raw["Governorate"].map(clean_governorate),
            "delegation": raw["Delegation"].map(clean_city),
            "city": raw["Locality"].map(clean_city),
            "location": raw["Locality"].map(clean_city),
            "surface_m2": raw["Surface"].map(ensure_positive_float),
            "rooms": raw["Type of Real Estate"].map(lambda value: ensure_non_negative_int(repair_text(value).split("-room")[0]) if "-room" in repair_text(value).lower() else None),
            "bathrooms": None,
            "price_tnd": raw["Price"].map(ensure_positive_float),
        }
    )

    sale_like = nature.map(normalize_key).isin({"sale", "land"})
    cleaned = cleaned[sale_like].copy()
    return finalize_clean_frame("tunisia_real_estate", cleaned)


def clean_property_prices_in_tunisia(path: Path) -> pd.DataFrame:
    raw = read_csv_any(path).reset_index().rename(columns={"index": "source_row_id"})
    type_key = raw["type"].map(normalize_key)
    category_key = raw["category"].map(normalize_key)

    allowed_categories = {
        "appartements": "apartment",
        "maisons et villas": "house",
        "terrains et fermes": "land",
    }

    cleaned = pd.DataFrame(
        {
            "source_row_id": raw["source_row_id"],
            "property_family": category_key.map(allowed_categories).fillna("other"),
            "governorate": raw["city"].map(clean_governorate),
            "delegation": raw["region"].map(clean_city),
            "city": raw["region"].map(clean_city),
            "location": raw["region"].map(clean_city),
            "surface_m2": raw["size"].map(ensure_positive_float),
            "rooms": raw["room_count"].map(ensure_non_negative_int),
            "bathrooms": raw["bathroom_count"].map(ensure_non_negative_int),
            "price_tnd": raw["price"].map(ensure_positive_float),
        }
    )

    sale_like = type_key.eq("a vendre")
    wanted_family = cleaned["property_family"].isin(ALLOWED_FAMILIES)
    valid_region = raw["region"].map(normalize_key).ne("autres villes")
    cleaned = cleaned[sale_like & wanted_family & valid_region].copy()
    return finalize_clean_frame("property_prices_in_tunisia", cleaned)


def split_location_pair(value: Any) -> tuple[str, str]:
    text = repair_text(value)
    if not text:
        return "", ""
    if "," not in text:
        city = clean_city(text)
        return "", city
    left, right = [part.strip() for part in text.split(",", 1)]
    return clean_governorate(left), clean_city(right)


def clean_data_prices_cleaned(path: Path) -> pd.DataFrame:
    raw = read_csv_any(path).reset_index().rename(columns={"index": "source_row_id"})
    location_pairs = raw["location"].map(split_location_pair)
    location_state = location_pairs.map(lambda item: item[0])
    location_city = location_pairs.map(lambda item: item[1])

    category_key = raw["category"].map(normalize_key)
    allowed_categories = {
        "appartements": "apartment",
        "maisons et villas": "house",
        "terrains et fermes": "land",
    }

    cleaned = pd.DataFrame(
        {
            "source_row_id": raw["source_row_id"],
            "property_family": category_key.map(allowed_categories).fillna("other"),
            "governorate": raw["state"].map(clean_governorate),
            "delegation": raw["city"].map(clean_city),
            "city": raw["city"].map(clean_city),
            "location": raw["location"].map(repair_text),
            "surface_m2": raw["superficie"].map(ensure_positive_float),
            "rooms": raw["chambres"].map(ensure_non_negative_int),
            "bathrooms": raw["salles_de_bains"].map(ensure_non_negative_int),
            "price_tnd": raw["price"].map(ensure_positive_float),
        }
    )

    # Fill missing governorate/city from the combined location field when possible.
    cleaned.loc[cleaned["governorate"].eq(""), "governorate"] = location_state[cleaned["governorate"].eq("")]
    cleaned.loc[cleaned["city"].eq(""), "city"] = location_city[cleaned["city"].eq("")]
    cleaned.loc[cleaned["delegation"].eq(""), "delegation"] = location_city[cleaned["delegation"].eq("")]
    cleaned["location"] = cleaned["city"]

    sale_like = raw["transaction"].map(normalize_key).eq("sale")
    wanted_family = cleaned["property_family"].isin(ALLOWED_FAMILIES)
    cleaned = cleaned[sale_like & wanted_family].copy()
    return finalize_clean_frame("data_prices_cleaned", cleaned)


def validate_cleaned_dataset(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    all_empty_columns = []
    for column in frame.columns:
        series = frame[column]
        if series.isna().all():
            all_empty_columns.append(column)
        elif series.dtype == object and series.fillna("").map(repair_text).eq("").all():
            all_empty_columns.append(column)

    report = {
        "dataset_name": name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "all_empty_columns": all_empty_columns,
        "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "duplicate_rows": int(frame.duplicated().sum()),
        "invalid_property_family_rows": int((~frame["property_family"].isin(ALLOWED_FAMILIES)).sum()),
        "invalid_governorate_rows": int(frame["governorate"].map(normalize_key).map(lambda value: value not in CANONICAL_GOVERNORATES and value not in {normalize_key(v) for v in CANONICAL_GOVERNORATES.values()}).sum()),
        "surface_non_positive_rows": int(frame["surface_m2"].le(0).sum()),
        "price_non_positive_rows": int(frame["price_tnd"].le(0).sum()),
        "rows_missing_core_fields": int(
            (
                frame["property_family"].eq("")
                | frame["governorate"].eq("")
                | frame["city"].eq("")
                | frame["surface_m2"].isna()
                | frame["price_tnd"].isna()
            ).sum()
        ),
        "governorate_top": frame["governorate"].value_counts().head(15).to_dict(),
        "city_top": frame["city"].value_counts().head(20).to_dict(),
        "family_distribution": frame["property_family"].value_counts().to_dict(),
    }
    return report


def build_merge_readiness(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "02_dataset_cleaning",
        "datasets": [
            {
                "dataset_name": report["dataset_name"],
                "rows": report["rows"],
                "all_empty_columns": report["all_empty_columns"],
                "duplicate_rows": report["duplicate_rows"],
                "invalid_property_family_rows": report["invalid_property_family_rows"],
                "invalid_governorate_rows": report["invalid_governorate_rows"],
                "surface_non_positive_rows": report["surface_non_positive_rows"],
                "price_non_positive_rows": report["price_non_positive_rows"],
                "rows_missing_core_fields": report["rows_missing_core_fields"],
            }
            for report in reports
        ],
        "status": "ready_for_merge_if_all_issue_counts_are_zero",
    }


def main() -> None:
    ensure_processed_dirs()

    cleaned_frames = {
        "tunisia_real_estate": clean_tunisia_real_estate(DATASETS["tunisia_real_estate"]),
        "property_prices_in_tunisia": clean_property_prices_in_tunisia(DATASETS["property_prices_in_tunisia"]),
        "data_prices_cleaned": clean_data_prices_cleaned(DATASETS["data_prices_cleaned"]),
    }

    reports: list[dict[str, Any]] = []
    for name, frame in cleaned_frames.items():
        frame.to_csv(CLEANING_DIR / f"02_{name}_clean.csv", index=False)
        report = validate_cleaned_dataset(name, frame)
        reports.append(report)
        write_json(CLEANING_DIR / f"02_{name}_validation.json", report)

    merge_report = build_merge_readiness(reports)
    write_json(CLEANING_DIR / "02_merge_readiness.json", merge_report)

    summary = {
        report["dataset_name"]: {
            "rows": report["rows"],
            "all_empty_columns": report["all_empty_columns"],
            "duplicate_rows": report["duplicate_rows"],
            "invalid_governorate_rows": report["invalid_governorate_rows"],
            "rows_missing_core_fields": report["rows_missing_core_fields"],
        }
        for report in reports
    }
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
