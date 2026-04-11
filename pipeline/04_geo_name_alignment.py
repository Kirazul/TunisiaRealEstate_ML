from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import GEO_ALIGNMENT_DIR, MERGE_DIR, ensure_processed_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_DIR = PROJECT_ROOT / "geo"

INPUT_DATASET_PATH = MERGE_DIR / "03_final_merge_ready.csv"
GEOJSON_PATH = GEO_DIR / "delegations-full.geojson"
LOCALITY_JSON_PATH = GEO_DIR / "tunisia.json"

LOCALITY_NORMALIZATION = {
    ("Tunis", "Ain Zaghouan"): "Ain Zaghouan",
    ("Tunis", "Ain Zaghouen"): "Ain Zaghouan",
    ("Tunis", "Ain Zaghouan Nord"): "Ain Zaghouan",
    ("Tunis", "L'Aouina"): "El Aouina",
    ("Tunis", "Aouina"): "El Aouina",
    ("Tunis", "Jardins De Carthage"): "Carthage",
    ("Tunis", "Gammarth"): "La Marsa",
    ("Tunis", "Mutuelleville"): "El Menzah",
    ("Tunis", "Manar"): "El Menzah",
    ("Tunis", "El Manar 1"): "El Menzah",
    ("Tunis", "El Manar 2"): "El Menzah",
    ("Tunis", "Tunis"): "Bab Bhar",
    ("Ariana", "Ennasr"): "Ariana Ville",
    ("Ariana", "Cite Ennasr 2"): "Ariana Ville",
    ("Ariana", "Ghazela"): "Raoued",
    ("Ariana", "Jardins D'el Menzah"): "Ariana Ville",
    ("Ariana", "Les Jardins El Menzah 1"): "Ariana Ville",
    ("Ariana", "Les Jardins El Menzah 2"): "Ariana Ville",
    ("Ariana", "Chotrana"): "La Soukra",
    ("Ariana", "Chotrana 1"): "La Soukra",
    ("Ariana", "Chotrana 2"): "La Soukra",
    ("Ariana", "Chotrana 3"): "La Soukra",
    ("Ariana", "Riadh Andalous"): "Ariana Ville",
    ("Ben Arous", "Medina Jedida"): "Nouvelle Medina",
    ("Ben Arous", "Mohamedia"): "Mohamadia",
    ("Ben Arous", "Rads"): "Rades",
    ("Manouba", "La Manouba"): "Manouba",
    ("Manouba", "Manouba Ville"): "Manouba",
    ("Manouba", "Manouba"): "Manouba",
    ("Manouba", "Denden"): "Manouba",
    ("Sousse", "Kantaoui"): "Hammam Sousse",
    ("Sousse", "El Kantaoui"): "Hammam Sousse",
    ("Sousse", "Khzema"): "Sousse Jaouhara",
    ("Sousse", "Khzema Est"): "Sousse Jaouhara",
    ("Sousse", "Khzema Ouest"): "Sousse Jaouhara",
    ("Sousse", "Chott Mariem"): "Akouda",
    ("Sousse", "Sousse Jawhara"): "Sousse Jaouhara",
    ("Sousse", "Sousse Jaouhara"): "Sousse Jaouhara",
    ("Nabeul", "Mrezga"): "Hammamet",
    ("Nabeul", "Hammamet Nord"): "Hammamet",
    ("Nabeul", "Hammamet Centre"): "Hammamet",
    ("Medenine", "Djerba - Houmet Essouk"): "Houmt Souk",
    ("Medenine", "Djerba Houmt Souk"): "Houmt Souk",
    ("Le Kef", "Le Kef Est"): "Le Kef Est",
}

MANUAL_LOCALITY_TO_DELEGATION = {
    ("Ariana", "Ariana"): "Ariana Ville",
    ("Ariana", "Ariana Ville"): "Ariana Ville",
    ("Ariana", "Ariana Essoughra"): "Ariana Ville",
    ("Ariana", "La Soukra"): "La Soukra",
    ("Ariana", "Borj Louzir"): "Mnihla",
    ("Tunis", "Ain Zaghouan"): "La Marsa",
    ("Tunis", "Ain Zaghouen"): "La Marsa",
    ("Tunis", "Ain Zaghouan Nord"): "La Marsa",
    ("Tunis", "L'Aouina"): "La Marsa",
    ("Tunis", "Aouina"): "La Marsa",
    ("Tunis", "Jardins De Carthage"): "Carthage",
    ("Tunis", "Gammarth"): "La Marsa",
    ("Tunis", "Mutuelleville"): "El Menzah",
    ("Tunis", "Manar"): "El Menzah",
    ("Tunis", "El Manar 1"): "El Menzah",
    ("Tunis", "El Manar 2"): "El Menzah",
    ("Tunis", "Tunis"): "Bab Bhar",
    ("Tunis", "Agba"): "Sidi Hassine",
    ("Tunis", "Ain Zaghouan Sud"): "La Marsa",
    ("Tunis", "Centre Urbain Nord"): "Cite El Khadra",
    ("Tunis", "Centre Ville Lafayette"): "Bab Bhar",
    ("Tunis", "Centre Ville - Lafayette"): "Bab Bhar",
    ("Tunis", "Cit Olympique"): "El Menzah",
    ("Tunis", "El Kabaria"): "El Kabaria",
    ("Tunis", "Lac 1"): "La Marsa",
    ("Tunis", "Lac 2"): "La Marsa",
    ("Tunis", "Sidi Bou Said"): "Carthage",
    ("Tunis", "Sejoumi"): "Sijoumi",
    ("Tunis", "Sijoumi"): "Sijoumi",
    ("Tunis", "Séjoumi"): "Sijoumi",
    ("Tunis", "Le Bardo"): "Le Bardo",
    ("Tunis", "Bardo"): "Le Bardo",
    ("Tunis", "La Medina"): "La Medina",
    ("Tunis", "Menzah"): "El Menzah",
    ("Tunis", "El Kram"): "El Kram",
    ("Ariana", "Ennasr"): "Ariana Ville",
    ("Ariana", "Cite Ennasr 2"): "Ariana Ville",
    ("Ariana", "Cit Hedi Nouira"): "Ariana Ville",
    ("Ariana", "Ghazela"): "Raoued",
    ("Ariana", "Jardins D'el Menzah"): "Ariana Ville",
    ("Ariana", "Jardins El Menzah"): "Ariana Ville",
    ("Ariana", "Les Jardins El Menzah 1"): "Ariana Ville",
    ("Ariana", "Les Jardins El Menzah 2"): "Ariana Ville",
    ("Ariana", "Chotrana"): "La Soukra",
    ("Ariana", "Chotrana 1"): "La Soukra",
    ("Ariana", "Chotrana 2"): "La Soukra",
    ("Ariana", "Chotrana 3"): "La Soukra",
    ("Ariana", "Riadh Andalous"): "Ariana Ville",
    ("Ben Arous", "Medina Jedida"): "Nouvelle Medina",
    ("Ben Arous", "Mohamedia"): "Mohamadia",
    ("Ben Arous", "Rads"): "Rades",
    ("Gabes", "Gabs"): "Gabes Medina",
    ("Manouba", "La Manouba"): "Manouba",
    ("Manouba", "Manouba Ville"): "Manouba",
    ("Manouba", "Manouba"): "Manouba",
    ("Manouba", "Denden"): "Manouba",
    ("Manouba", "Djedeida"): "Jedaida",
    ("Sousse", "Kantaoui"): "Hammam Sousse",
    ("Sousse", "El Kantaoui"): "Hammam Sousse",
    ("Sousse", "Khzema"): "Sousse Ville",
    ("Sousse", "Khzema Est"): "Sousse Ville",
    ("Sousse", "Khzema Ouest"): "Sousse Ville",
    ("Sousse", "Chott Mariem"): "Akouda",
    ("Sousse", "Sousse Jawhara"): "Sousse Jaouhara",
    ("Sousse", "Sousse Jaouhara"): "Sousse Jaouhara",
    ("Sousse", "Zaouit-ksibat Thrayett"): "Sousse Jaouhara",
    ("Sousse", "Zaouit Ksibat Thrayett"): "Sousse Jaouhara",
    ("Sousse", "Sousse"): "Sousse Ville",
    ("Sousse", "Sousse Ville"): "Sousse Ville",
    ("Sousse", "Sahloul"): "Sousse Jaouhara",
    ("Nabeul", "Mrezga"): "Hammamet",
    ("Nabeul", "Hammamet Nord"): "Hammamet",
    ("Nabeul", "Hammamet Centre"): "Hammamet",
    ("Nabeul", "El Haouaria"): "El Haouaria",
    ("Medenine", "Djerba - Houmet Essouk"): "Houmt Souk",
    ("Medenine", "Djerba Houmt Souk"): "Houmt Souk",
    ("Medenine", "Ben Gardane"): "Ben Guerdane",
    ("Kairouan", "El Ouslatia"): "Oueslatia",
    ("Sidi Bouzid", "Bir El Hafey"): "Bir El Hafey",
    ("Bizerte", "Zarzouna"): "Bizerte Nord",
    ("Zaghouan", "Ez-zeriba"): "Zriba",
    ("Le Kef", "Le Kef Est"): "Le Kef Est",
    ("Ben Arous", "Bou Mhel El Bassatine"): "Bou Mhel El Bassatine",
    ("Sfax", "Route Soukra"): "Sfax Ville",
    ("Sfax", "Route Sokra"): "Sfax Ville",
    ("Sfax", "Route El Afrane"): "Sfax Ville",
    ("Sfax", "Route El Ain"): "Sfax Ville",
    ("Sfax", "Route Tunis"): "Sfax Ville",
    ("Sfax", "Route Menzel Chaker"): "Sfax Ville",
    ("Sfax", "Route Manzel Chaker"): "Sfax Ville",
    ("Sfax", "Route Mehdia"): "Sfax Ville",
    ("Sfax", "Route Gremda"): "Sfax Ville",
    ("Sfax", "Route Taniour"): "Sfax Ville",
    ("Sfax", "Route Mharza"): "Sfax Ville",
    ("Sfax", "Route Saltania"): "Sfax Ville",
    ("Sfax", "Sfax"): "Sfax Ville",
    ("Sfax", "Sfax Ville"): "Sfax Ville",
}

GOVERNORATE_CANONICAL = {
    "Ariana": "Ariana",
    "Beja": "Beja",
    "Ben Arous": "Ben Arous",
    "Bizerte": "Bizerte",
    "Gabes": "Gabes",
    "Gafsa": "Gafsa",
    "Jendouba": "Jendouba",
    "Kairouan": "Kairouan",
    "Kasserine": "Kasserine",
    "Kebili": "Kebili",
    "Le Kef": "Le Kef",
    "Mahdia": "Mahdia",
    "Manouba": "Manouba",
    "Medenine": "Medenine",
    "Monastir": "Monastir",
    "Nabeul": "Nabeul",
    "Sfax": "Sfax",
    "Sidi Bouzid": "Sidi Bouzid",
    "Siliana": "Siliana",
    "Sousse": "Sousse",
    "Tataouine": "Tataouine",
    "Tozeur": "Tozeur",
    "Tunis": "Tunis",
    "Zaghouan": "Zaghouan",
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
    "beja nord": "Beja Nord",
    "megrine": "Megrine",
    "djerba midoun": "Djerba Midoun",
    "teboulba": "Teboulba",
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


def clean_governorate_name(value: Any) -> str:
    normalized = normalize_text(value)
    aliases = {
        "beja": "Beja",
        "baja": "Beja",
        "gabes": "Gabes",
        "gabes": "Gabes",
        "kasserine": "Kasserine",
        "sidi bou zid": "Sidi Bouzid",
        "manubah": "Manouba",
        "medenine": "Medenine",
        "mednine": "Medenine",
        "kef": "Le Kef",
        "le kef": "Le Kef",
    }
    canonical = aliases.get(normalized)
    if canonical:
        return canonical
    for name in GOVERNORATE_CANONICAL:
        if normalize_text(name) == normalized:
            return name
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


def build_geo_reference() -> tuple[pd.DataFrame, dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    payload = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    locality_payload = json.loads(LOCALITY_JSON_PATH.read_text(encoding="utf-8"))

    rows: list[dict[str, str]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        feature_type = normalize_text(props.get("type_2") or props.get("engtype_2"))
        if feature_type != "delegation":
            continue
        governorate = clean_governorate_name(props.get("gov_name_f") or props.get("circo_na_1") or props.get("name_1"))
        delegation = clean_delegation_name(
            governorate,
            props.get("deleg_na_1") or props.get("name_2") or props.get("deleg_name"),
        )
        if not governorate or not delegation or normalize_text(delegation).startswith("unknown"):
            continue
        rows.append(
            {
                "governorate": governorate,
                "delegation": delegation,
                "governorate_norm": normalize_text(governorate),
                "delegation_norm": normalize_text(delegation),
            }
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["governorate_norm", "delegation_norm"]).reset_index(drop=True)

    exact_delegations: dict[str, list[dict[str, str]]] = {}
    exact_localities: dict[str, list[dict[str, str]]] = {}
    substring_localities: dict[str, list[dict[str, str]]] = {}

    for row in frame.to_dict("records"):
        exact_delegations.setdefault(row["delegation_norm"], []).append(row)

    for governorate, records in locality_payload.items():
        canonical_governorate = clean_governorate_name(governorate)
        for record in records:
            delegation = clean_delegation_name(canonical_governorate, record.get("delegation"))
            locality = repair_text(record.get("localite"))
            if not delegation:
                continue
            payload_row = {
                "governorate": canonical_governorate,
                "delegation": delegation,
                "locality": locality,
                "delegation_norm": normalize_text(delegation),
                "locality_norm": normalize_text(locality),
            }
            if payload_row["delegation_norm"]:
                exact_delegations.setdefault(payload_row["delegation_norm"], []).append(
                    {
                        "governorate": canonical_governorate,
                        "delegation": delegation,
                        "governorate_norm": normalize_text(canonical_governorate),
                        "delegation_norm": payload_row["delegation_norm"],
                    }
                )
            if payload_row["locality_norm"]:
                exact_localities.setdefault(payload_row["locality_norm"], []).append(
                    {
                        "governorate": canonical_governorate,
                        "delegation": delegation,
                        "governorate_norm": normalize_text(canonical_governorate),
                        "delegation_norm": payload_row["delegation_norm"],
                    }
                )
                tokens = payload_row["locality_norm"].split()
                for start in range(len(tokens)):
                    for end in range(start + 1, len(tokens) + 1):
                        piece = " ".join(tokens[start:end])
                        if len(piece) < 5:
                            continue
                        substring_localities.setdefault(piece, []).append(
                            {
                                "governorate": canonical_governorate,
                                "delegation": delegation,
                                "governorate_norm": normalize_text(canonical_governorate),
                                "delegation_norm": payload_row["delegation_norm"],
                            }
                        )

    return frame, exact_delegations, exact_localities, substring_localities


def unique_governorate_candidate(candidates: list[dict[str, str]], governorate_norm: str) -> dict[str, str] | None:
    filtered = [candidate for candidate in candidates if candidate["governorate_norm"] == governorate_norm]
    unique = {(item["governorate"], item["delegation"], item["governorate_norm"], item["delegation_norm"]) for item in filtered}
    if len(unique) == 1:
        governorate, delegation, gov_norm, del_norm = next(iter(unique))
        return {
            "governorate": governorate,
            "delegation": delegation,
            "governorate_norm": gov_norm,
            "delegation_norm": del_norm,
        }
    return None


def best_fuzzy_delegation(term_norm: str, governorate_norm: str, geo_frame: pd.DataFrame) -> dict[str, str] | None:
    candidates = geo_frame[geo_frame["governorate_norm"] == governorate_norm].copy()
    if candidates.empty or not term_norm:
        return None
    candidates["score"] = candidates["delegation_norm"].map(lambda value: SequenceMatcher(None, term_norm, value).ratio())
    best = candidates.sort_values("score", ascending=False).iloc[0]
    if float(best["score"]) < 0.88:
        return None
    return {
        "governorate": str(best["governorate"]),
        "delegation": str(best["delegation"]),
        "governorate_norm": str(best["governorate_norm"]),
        "delegation_norm": str(best["delegation_norm"]),
    }


def is_valid_geo_pair(governorate: str, delegation: str, geo_frame: pd.DataFrame) -> bool:
    if not governorate or not delegation:
        return False
    gov_norm = normalize_text(governorate)
    del_norm = normalize_text(delegation)
    return not geo_frame[(geo_frame["governorate_norm"] == gov_norm) & (geo_frame["delegation_norm"] == del_norm)].empty


def align_row(
    record: dict[str, Any],
    geo_frame: pd.DataFrame,
    exact_delegations: dict[str, list[dict[str, str]]],
    exact_localities: dict[str, list[dict[str, str]]],
    substring_localities: dict[str, list[dict[str, str]]],
) -> tuple[str, str, str]:
    governorate = clean_governorate_name(record.get("governorate"))
    governorate_norm = normalize_text(governorate)
    delegation = repair_text(record.get("delegation"))
    city = repair_text(record.get("city"))
    location = repair_text(record.get("location"))

    normalized_locality = clean_text_name(city or delegation or location)
    for key in [(governorate, delegation), (governorate, city), (governorate, location)]:
        override = LOCALITY_NORMALIZATION.get(key)
        if override:
            normalized_locality = clean_text_name(override)
            break

    manual_keys = [
        (governorate, normalized_locality),
        (governorate, delegation),
        (governorate, city),
        (governorate, location),
    ]
    for key in manual_keys:
        override = MANUAL_LOCALITY_TO_DELEGATION.get(key)
        if not override:
            continue
        cleaned_override = clean_delegation_name(governorate, override)
        if is_valid_geo_pair(governorate, cleaned_override, geo_frame):
            return governorate, cleaned_override, "manual_override"

    terms = [normalized_locality, delegation, city, location]
    for term in terms:
        term_norm = normalize_text(term)
        if not term_norm:
            continue
        exact = unique_governorate_candidate(exact_delegations.get(term_norm, []), governorate_norm)
        if exact is not None:
            return exact["governorate"], clean_delegation_name(exact["governorate"], exact["delegation"]), "exact_delegation"
        locality = unique_governorate_candidate(exact_localities.get(term_norm, []), governorate_norm)
        if locality is not None:
            cleaned_locality_delegation = clean_delegation_name(locality["governorate"], locality["delegation"])
            if is_valid_geo_pair(locality["governorate"], cleaned_locality_delegation, geo_frame):
                return locality["governorate"], cleaned_locality_delegation, "exact_locality"
        substring = unique_governorate_candidate(substring_localities.get(term_norm, []), governorate_norm)
        if substring is not None:
            cleaned_substring_delegation = clean_delegation_name(substring["governorate"], substring["delegation"])
            if is_valid_geo_pair(substring["governorate"], cleaned_substring_delegation, geo_frame):
                return substring["governorate"], cleaned_substring_delegation, "substring_locality"

    for term in terms:
        term_norm = normalize_text(term)
        if not term_norm:
            continue
        fuzzy = best_fuzzy_delegation(term_norm, governorate_norm, geo_frame)
        if fuzzy is not None:
            return fuzzy["governorate"], clean_delegation_name(fuzzy["governorate"], fuzzy["delegation"]), "fuzzy_delegation"

    return governorate, "", "unmatched"


def main() -> None:
    ensure_processed_dirs()

    geo_frame, exact_delegations, exact_localities, substring_localities = build_geo_reference()
    dataset = pd.read_csv(INPUT_DATASET_PATH)

    aligned_rows: list[dict[str, Any]] = []
    for record in dataset.to_dict("records"):
        normalized_locality = clean_text_name(repair_text(record.get("city")) or repair_text(record.get("delegation")) or repair_text(record.get("location")))
        governorate_for_norm = clean_governorate_name(record.get("governorate"))
        for key in [
            (governorate_for_norm, repair_text(record.get("delegation"))),
            (governorate_for_norm, repair_text(record.get("city"))),
            (governorate_for_norm, repair_text(record.get("location"))),
        ]:
            override = LOCALITY_NORMALIZATION.get(key)
            if override:
                normalized_locality = clean_text_name(override)
                break
        canonical_governorate, canonical_delegation, match_status = align_row(
            record,
            geo_frame,
            exact_delegations,
            exact_localities,
            substring_localities,
        )
        aligned = dict(record)
        aligned["normalized_locality"] = normalized_locality
        aligned["geo_governorate"] = canonical_governorate
        aligned["geo_delegation"] = clean_delegation_name(canonical_governorate, canonical_delegation)
        aligned["geo_match_status"] = match_status
        if aligned["geo_delegation"] and not is_valid_geo_pair(aligned["geo_governorate"], aligned["geo_delegation"], geo_frame):
            aligned["geo_delegation"] = ""
            aligned["geo_match_status"] = "unmatched"
        aligned_rows.append(aligned)

    aligned_frame = pd.DataFrame(aligned_rows)
    aligned_frame.to_csv(GEO_ALIGNMENT_DIR / "04_geo_aligned_dataset.csv", index=False)

    unmatched = aligned_frame[aligned_frame["geo_match_status"] == "unmatched"].copy()
    unmatched.to_csv(GEO_ALIGNMENT_DIR / "04_geo_unmatched_rows.csv", index=False)

    counts_by_governorate = geo_frame.groupby("governorate").size().sort_index().to_dict()
    report = {
        "stage": "04_geo_name_alignment",
        "geo_canonical_delegations": int(len(geo_frame)),
        "geo_counts_by_governorate": counts_by_governorate,
        "dataset_rows": int(len(aligned_frame)),
        "match_status": aligned_frame["geo_match_status"].value_counts().to_dict(),
        "matched_rows": int(aligned_frame["geo_delegation"].fillna("").astype(str).str.strip().ne("").sum()),
        "unmatched_rows": int(len(unmatched)),
        "unique_geo_delegations_covered": int(aligned_frame["geo_delegation"].fillna("").astype(str).str.strip().nunique() - (1 if "" in set(aligned_frame["geo_delegation"].fillna("").astype(str)) else 0)),
        "top_unmatched_pairs": [
            {"governorate": governorate, "city": city, "count": int(count)}
            for (governorate, city), count in unmatched.groupby(["geo_governorate", "city"]).size().sort_values(ascending=False).head(50).items()
        ],
    }
    write_json(GEO_ALIGNMENT_DIR / "04_geo_alignment_report.json", report)

    summary = {
        "geo_canonical_delegations": int(len(geo_frame)),
        "dataset_rows": int(len(aligned_frame)),
        "match_status": report["match_status"],
        "unmatched_rows": int(len(unmatched)),
        "unique_geo_delegations_covered": report["unique_geo_delegations_covered"],
    }
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
