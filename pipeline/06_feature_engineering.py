from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import unicodedata

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

BASE_FEATURE_COLUMNS = ["surface_m2", "rooms", "lon", "lat"]
SPLIT_DEPENDENT_FEATURE_COLUMNS = [
    "geo_delegation_target_enc",
    "geo_governorate_target_enc",
    "price_vs_local_median",
]
FINAL_MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + SPLIT_DEPENDENT_FEATURE_COLUMNS + [
    "property_family",
    "geo_governorate",
    "geo_delegation",
]

BASE_FEATURE_GROUPS = {
    "base": BASE_FEATURE_COLUMNS,
}

BASE_FEATURE_DESCRIPTIONS = {
    "surface_m2": "Property surface area in square meters.",
    "rooms": "Reported room count when available.",
    "lon": "Delegation centroid longitude from the canonical geo reference.",
    "lat": "Delegation centroid latitude from the canonical geo reference.",
}

FEATURE_GROUPS = {
    "base": BASE_FEATURE_COLUMNS,
    "split_dependent": SPLIT_DEPENDENT_FEATURE_COLUMNS,
    "categorical": ["property_family", "geo_governorate", "geo_delegation"],
    "final_model": FINAL_MODEL_FEATURE_COLUMNS,
}

FEATURE_DESCRIPTIONS = {
    **BASE_FEATURE_DESCRIPTIONS,
    "geo_delegation_target_enc": "Training-split mean log price for the matched delegation.",
    "geo_governorate_target_enc": "Training-split mean log price for the matched governorate.",
    "price_vs_local_median": "Relative price-per-m2 ratio versus the delegation median.",
    "property_family": "Property type bucket used as a categorical model feature.",
    "geo_governorate": "Canonical governorate name used as a categorical location feature.",
    "geo_delegation": "Canonical delegation name used as a categorical location feature.",
}

STAGE6_FEATURE_NOTE = "Stage 6 writes the base engineered dataset with canonical geographic coordinates."

GOVERNORATE_ALIASES = {
    "beja": "Beja",
    "baja": "Beja",
    "gabes": "Gabes",
    "kasserine": "Kasserine",
    "sidi bou zid": "Sidi Bouzid",
    "manubah": "Manouba",
    "medenine": "Medenine",
    "mednine": "Medenine",
    "kef": "Le Kef",
    "le kef": "Le Kef",
}

TEXT_CANONICAL_ALIASES = {
    "B�ja Nord": "Beja Nord",
    "M�grine": "Megrine",
    "Djerba - Midoun": "Djerba Midoun",
    "T�boulba": "Teboulba",
    "B�ni Khiar": "Beni Khiar",
    "Sakiet Edda�er": "Sakiet Eddaier",
    "Zaouit-ksibat Thrayett": "Zaouit Ksibat Thrayett",
    "Centre Ville - Lafayette": "Centre Ville Lafayette",
    "Cit� El Khadra": "Cite El Khadra",
    "El Omrane Sup�rieur": "El Omrane Superieur",
    "Omrane Sup�rieur": "Omrane Superieur",
    "Sidi El B�chir": "Sidi El Bechir",
    "Gab�s Ouest": "Gabes Ouest",
    "Gab�s Sud": "Gabes Sud",
    "Gab�s M�dina": "Gabes Medina",
    "Sfax M�dina": "Sfax Medina",
    "Sousse M�dina": "Sousse Medina",
    "M�denine Nord": "Medenine Nord",
    "M�denine Sud": "Medenine Sud",
    "Ksibet El-m�diouni": "Ksibet El Mediouni",
    "M�dina": "Medina",
}

TEXT_CANONICAL_BY_NORM = {
    "ain draham": "Ain Draham",
    "ariana medina": "Ariana Medina",
    "beja nord": "Beja Nord",
    "megrine": "Megrine",
    "djerba midoun": "Djerba Midoun",
    "nouvelle medina": "Nouvelle Medina",
    "rades": "Rades",
    "teboulba": "Teboulba",
    "teboursouk": "Teboursouk",
    "beni khiar": "Beni Khiar",
    "sakiet eddaier": "Sakiet Eddaier",
    "zaouit ksibat thrayett": "Zaouit Ksibat Thrayett",
    "centre ville lafayette": "Centre Ville Lafayette",
    "cite el khadra": "Cite El Khadra",
    "el omrane superieur": "El Omrane Superieur",
    "omrane superieur": "Omrane Superieur",
    "sidi el bechir": "Sidi El Bechir",
    "gabes ouest": "Gabes Ouest",
    "gabes sud": "Gabes Sud",
    "gabes medina": "Gabes Medina",
    "sfax medina": "Sfax Medina",
    "sousse medina": "Sousse Medina",
    "medenine nord": "Medenine Nord",
    "medenine sud": "Medenine Sud",
    "ksibet el mediouni": "Ksibet El Mediouni",
    "medina": "Medina",
}

DELEGATION_CANONICAL_ALIASES = {
    ("Sousse", "Sousse Jawhara"): "Sousse Jaouhara",
    ("Ariana", "Ariana Ville"): "Ariana Medina",
    ("Ariana", "La Soukra"): "Soukra",
    ("Ariana", "Kalaat Landlous"): "Kalaat El Andalous",
    ("Tunis", "Le Bardo"): "Bardo",
    ("Tunis", "El Ouerdia"): "El Ouardia",
    ("Tunis", "El Omrane Superieur"): "Omrane Superieur",
    ("Tunis", "El Kabaria"): "Kabaria",
    ("Tunis", "El Kabbaria"): "Kabaria",
    ("Tunis", "Essijoumi"): "Sijoumi",
    ("Tunis", "Ezzouhour (Tunis)"): "Ezzouhour",
    ("Tunis", "La Medina"): "Medina",
    ("Tunis", "El Tahrir"): "El Tahrir",
    ("Tunis", "Ettahrir"): "El Tahrir",
    ("Tunis", "El Omrane"): "Omrane",
    ("Tunis", "El Hrairia"): "Hrairia",
    ("Tunis", "El Kram"): "Le Kram",
    ("Tunis", "Hrairia"): "Hrairia",
    ("Tunis", "Cite El Khadra"): "Cite El Khadra",
    ("Tunis", "Sidi El Bechir"): "Sidi El Bechir",
    ("Ben Arous", "Bou Mhel El Bassatine"): "Boumhel",
    ("Ben Arous", "Hammam Chatt"): "Hammam Chott",
    ("Ben Arous", "Mohamadia"): "M'Hamdia",
    ("Ben Arous", "Rades"): "Rades",
    ("Ben Arous", "Megrine"): "Megrine",
    ("Nabeul", "Hammam El Ghezaz"): "Hammam Ghezaz",
    ("Nabeul", "El Haouaria"): "Haouaria",
    ("Nabeul", "Dar Chaabane Elfehri"): "Dar Chaabane El Fehri",
    ("Sousse", "Sousse Ville"): "Sousse Medina",
    ("Sousse", "Kalaa Essghira"): "Kalaa Sghira",
    ("Sousse", "Kalaa El Kebira"): "Kalaa Kebira",
    ("Sousse", "Bou Ficha"): "Bouficha",
    ("Sousse", "Msaken"): "M'Saken",
    ("Sousse", "Zaouit-ksibat Thrayett"): "Zaouia Ksiba Thraya",
    ("Sousse", "Zaouit Ksibat Thrayett"): "Zaouia Ksiba Thraya",
    ("Sfax", "Sfax Ville"): "Sfax Medina",
    ("Sfax", "Sfax Est"): "Sfax Sud",
    ("Sfax", "El Hencha"): "Hencha",
    ("Sfax", "Mahras"): "Mahres",
    ("Sfax", "Kerkenah"): "Kerkennah",
    ("Le Kef", "Le Kef Est"): "Kef Est",
    ("Le Kef", "Le Kef Ouest"): "Kef Ouest",
    ("Monastir", "Jemmal"): "Jammel",
    ("Monastir", "Ksar Helal"): "Ksar Hellal",
    ("Monastir", "Sayada Lamta Bou Hajar"): "Sayada-Lamta-Bou Hjar",
    ("Zaghouan", "Bir Mcherga"): "Bir Mchergua",
    ("Zaghouan", "El Fahs"): "Fahs",
    ("Zaghouan", "Hammam Zriba"): "Zriba",
    ("Zaghouan", "Ennadhour"): "Nadhour",
    ("Gabes", "El Hamma"): "Hamma",
    ("Gabes", "Ghannouche"): "Ghannouch",
    ("Gabes", "Nouvelle Matmata"): "Matmata Nouvelle",
    ("Gabes", "Gabes Medina"): "Gabes Medina",
    ("Gabes", "Gabes Ouest"): "Gabes Ouest",
    ("Gabes", "Gabes Sud"): "Gabes Sud",
    ("Jendouba", "Bou Salem"): "Bousalem",
    ("Siliana", "El Aroussa"): "Aroussa",
    ("Mahdia", "Ksour Essaf"): "Ksour Essef",
    ("Mahdia", "La Chebba"): "Chebba",
    ("Gafsa", "El Ksar"): "Ksar",
    ("Gafsa", "Sned"): "Sened",
    ("Le Kef", "El Ksour"): "Ksour",
    ("Sidi Bouzid", "Maknassy"): "Meknassi",
    ("Sidi Bouzid", "Mezzouna"): "Mazzouna",
}


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


def normalize_text(value: Any) -> str:
    text = repair_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_governorate_name(value: Any) -> str:
    normalized = normalize_text(value)
    canonical = GOVERNORATE_ALIASES.get(normalized)
    if canonical:
        return canonical
    return repair_text(value)


def clean_text_name(value: Any) -> str:
    cleaned = repair_text(value)
    if not cleaned:
        return ""
    exact = TEXT_CANONICAL_ALIASES.get(cleaned)
    if exact:
        return exact
    return TEXT_CANONICAL_BY_NORM.get(normalize_text(cleaned), cleaned)


def clean_delegation_name(governorate: str, delegation: Any) -> str:
    cleaned = clean_text_name(delegation)
    if not cleaned:
        return ""
    return DELEGATION_CANONICAL_ALIASES.get((governorate, cleaned), cleaned)


def build_delegation_key(governorate: str, delegation: str) -> str:
    return f"{governorate}::{delegation}"


def load_delegation_coords() -> dict[str, tuple[float, float]]:
    geojson_path = GEO_DIR / "delegations-full.geojson"
    with open(geojson_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    coords: dict[str, tuple[float, float]] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if props.get("engtype_2") != "Delegation" or not props.get("name_2"):
            continue

        governorate = clean_governorate_name(
            props.get("gov_name_f") or props.get("circo_na_1") or props.get("name_1")
        )
        delegation = clean_delegation_name(
            governorate,
            props.get("deleg_na_1") or props.get("name_2") or props.get("deleg_name"),
        )
        if not governorate or not delegation or normalize_text(delegation).startswith("unknown"):
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
            coords[build_delegation_key(governorate, delegation)] = (
                sum(all_lons) / len(all_lons),
                sum(all_lats) / len(all_lats),
            )

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

    feature_frame["delegation_key"] = feature_frame.apply(
        lambda row: build_delegation_key(
            str(row["geo_governorate"]).strip(),
            str(row["geo_delegation"]).strip(),
        )
        if str(row["geo_governorate"]).strip() and str(row["geo_delegation"]).strip()
        else "",
        axis=1,
    )
    feature_frame["lon"] = feature_frame["delegation_key"].map(lambda value: delegation_coords.get(value, (None, None))[0])
    feature_frame["lat"] = feature_frame["delegation_key"].map(lambda value: delegation_coords.get(value, (None, None))[1])

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
    rows_with_geo_delegation = frame["geo_delegation"].fillna("").astype(str).str.strip().ne("")
    rows_with_coords = frame["lon"].notna() & frame["lat"].notna()
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "feature_groups": BASE_FEATURE_GROUPS,
        "feature_descriptions": BASE_FEATURE_DESCRIPTIONS,
        "feature_columns": BASE_FEATURE_COLUMNS,
        "stage6_note": STAGE6_FEATURE_NOTE,
        "geo_delegation_count": int(frame["geo_delegation"].nunique()),
        "geo_governorate_count": int(frame["geo_governorate"].nunique()),
        "rows_with_geo_delegation": int(rows_with_geo_delegation.sum()),
        "rows_with_coords": int(rows_with_coords.sum()),
        "rows_missing_coords": int((~rows_with_coords).sum()),
        "rows_missing_coords_despite_geo_match": int((rows_with_geo_delegation & ~rows_with_coords).sum()),
        "delegations_with_coords": int(frame.loc[rows_with_coords, "geo_delegation"].nunique()),
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
