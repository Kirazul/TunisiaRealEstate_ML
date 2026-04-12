from __future__ import annotations

import importlib.util
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paths import FEATURE_ENGINEERING_DIR, MODEL_TRAINING_DIR, ensure_processed_dirs


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DATA_DIR = FRONTEND_DIR / "assets" / "data"
GEOJSON_PATH = PROJECT_ROOT / "geo" / "delegations-full.geojson"

INPUT_PATH = FEATURE_ENGINEERING_DIR / "06_feature_engineered_dataset.csv"
TRAINING_DATASET_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "05_training_dataset" / "05_training_dataset_report.json"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "08_best_model.joblib"
METRICS_PATH = MODEL_TRAINING_DIR / "08_model_training_report.json"
PREDICTIONS_PATH = MODEL_TRAINING_DIR / "08_holdout_predictions.csv"
MODEL_COMPARISON_PLOT_PATH = MODEL_TRAINING_DIR / "08_model_comparison.png"
HOLDOUT_SCATTER_PLOT_PATH = MODEL_TRAINING_DIR / "08_holdout_scatter.png"
FRONTEND_MODEL_SUMMARY_PATH = FRONTEND_DIR / "model_summary.json"
FRONTEND_ATLAS_PATH = FRONTEND_DATA_DIR / "atlas.geojson"
FRONTEND_ZONE_COVERAGE_PATH = FRONTEND_DATA_DIR / "zone_coverage.json"
FRONTEND_PROFILES_PATH = FRONTEND_DATA_DIR / "delegation_profiles.json"
FRONTEND_PIPELINE_MANIFEST_PATH = FRONTEND_DATA_DIR / "pipeline_assets_manifest.json"

RANDOM_STATE = 42
FAMILIES = ["apartment", "house", "land"]
MIN_DIRECT_SUPPORT = 5

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
    "sidi bou zid": "Sidi Bouzid",
    "sidi bouzid": "Sidi Bouzid",
    "siliana": "Siliana",
    "sousse": "Sousse",
    "tataouine": "Tataouine",
    "tozeur": "Tozeur",
    "tunis": "Tunis",
    "zaghouan": "Zaghouan",
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def canonical_governorate(value: Any) -> str:
    repaired = repair_text(value)
    normalized = normalize_key(repaired)
    return CANONICAL_GOVERNORATES.get(normalized, repaired)


def build_delegation_key(governorate: Any, delegation: Any) -> str:
    return f"{normalize_key(governorate)}::{normalize_key(delegation)}"


def polygon_area_km2(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    mean_lat = sum(lat for _, lat in coords) / len(coords)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(mean_lat))
    projected = [(lon * km_per_deg_lon, lat * km_per_deg_lat) for lon, lat in coords]
    area = 0.0
    for i in range(len(projected)):
        x1, y1 = projected[i]
        x2, y2 = projected[(i + 1) % len(projected)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def extract_rings(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry_type = geometry.get("type")
    geometry_coords = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        rings = geometry_coords[:1]
    elif geometry_type == "MultiPolygon":
        rings = [polygon[0] for polygon in geometry_coords if polygon]
    else:
        rings = []
    return [[(float(point[0]), float(point[1])) for point in ring] for ring in rings if ring]


def compute_centroid_and_area(geometry: dict[str, Any]) -> tuple[float | None, float | None, float]:
    rings = extract_rings(geometry)
    if not rings:
        return None, None, 0.0
    all_points = [point for ring in rings for point in ring]
    centroid_lon = sum(lon for lon, _ in all_points) / len(all_points)
    centroid_lat = sum(lat for _, lat in all_points) / len(all_points)
    area_km2 = sum(polygon_area_km2(ring) for ring in rings)
    return centroid_lon, centroid_lat, area_km2


def load_feature_engineering_module():
    module_path = PROJECT_ROOT / "pipeline" / "06_feature_engineering.py"
    spec = importlib.util.spec_from_file_location("pipeline_feature_engineering", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load feature engineering module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_geography_artifacts() -> tuple[dict[str, Any], pd.DataFrame]:
    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    geo_rows: list[dict[str, Any]] = []
    kept_features: list[dict[str, Any]] = []

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        feature_type = normalize_key(props.get("engtype_2") or props.get("type_2"))
        deleg_id = props.get("deleg_id")
        if feature_type != "delegation" or deleg_id is None or float(deleg_id) <= 0:
            continue

        governorate = canonical_governorate(props.get("gov_name_f") or props.get("name_1"))
        delegation = repair_text(props.get("deleg_na_1") or props.get("name_2"))
        if not governorate or not delegation or delegation.lower().startswith("unknown"):
            continue

        centroid_lon, centroid_lat, area_km2 = compute_centroid_and_area(feature.get("geometry", {}))
        delegation_key = build_delegation_key(governorate, delegation)

        updated_feature = {
            **feature,
            "properties": {
                **props,
                "delegation_key": delegation_key,
                "governorate": governorate,
                "delegation": delegation,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "area_km2": round(area_km2, 3),
            },
        }
        kept_features.append(updated_feature)
        geo_rows.append(
            {
                "delegation_key": delegation_key,
                "governorate": governorate,
                "delegation": delegation,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "area_km2": round(area_km2, 3),
            }
        )

    geo_frame = pd.DataFrame(geo_rows).drop_duplicates(subset=["delegation_key"]).reset_index(drop=True)
    return {"type": "FeatureCollection", "features": kept_features}, geo_frame


def mode_or_fallback(series: pd.Series, fallback: str = "") -> str:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return fallback
    mode = cleaned.mode()
    return str(mode.iat[0] if not mode.empty else cleaned.iat[0])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    return default if math.isnan(numeric) else numeric


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def confidence_for(level: str, support_count: int) -> str:
    if level == "direct":
        if support_count >= 20:
            return "high"
        if support_count >= MIN_DIRECT_SUPPORT:
            return "medium"
        return "low"
    if level == "governorate_fallback":
        return "medium"
    return "low"


def build_frontend_profiles(support_frame: pd.DataFrame, geo_frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    direct_source = support_frame.copy()
    direct_source["delegation_key"] = direct_source.apply(
        lambda row: build_delegation_key(row["geo_governorate"], row["geo_delegation"]),
        axis=1,
    )

    direct = direct_source.groupby(["delegation_key", "property_family"], as_index=False).agg(
        support_count=("price_tnd", "size"),
        surface_m2=("surface_m2", "median"),
        rooms=("rooms", "median"),
        benchmark_ppm2=("price_per_m2", "median"),
        governorate=("geo_governorate", mode_or_fallback),
        delegation=("geo_delegation", mode_or_fallback),
        centroid_lon=("lon", "median"),
        centroid_lat=("lat", "median"),
    )

    governorate = support_frame.groupby(["geo_governorate", "property_family"], as_index=False).agg(
        support_count=("price_tnd", "size"),
        surface_m2=("surface_m2", "median"),
        rooms=("rooms", "median"),
        benchmark_ppm2=("price_per_m2", "median"),
        centroid_lon=("lon", "median"),
        centroid_lat=("lat", "median"),
    )

    national = support_frame.groupby(["property_family"], as_index=False).agg(
        support_count=("price_tnd", "size"),
        surface_m2=("surface_m2", "median"),
        rooms=("rooms", "median"),
        benchmark_ppm2=("price_per_m2", "median"),
        centroid_lon=("lon", "median"),
        centroid_lat=("lat", "median"),
    )

    direct_lookup = {(row.delegation_key, row.property_family): row._asdict() for row in direct.itertuples(index=False)}
    governorate_lookup = {(row.geo_governorate, row.property_family): row._asdict() for row in governorate.itertuples(index=False)}
    national_lookup = {row.property_family: row._asdict() for row in national.itertuples(index=False)}

    profiles: dict[str, Any] = {}
    zone_coverage: list[dict[str, Any]] = []
    tier_counts = {"exact_sector": 0, "governorate_fallback": 0, "national_fallback": 0}

    for geo_row in geo_frame.itertuples(index=False):
        family_profiles: dict[str, Any] = {}

        for family in FAMILIES:
            direct_profile = direct_lookup.get((geo_row.delegation_key, family))
            if direct_profile and int(direct_profile["support_count"]) >= MIN_DIRECT_SUPPORT:
                chosen = direct_profile
                level = "direct"
            else:
                chosen = governorate_lookup.get((geo_row.governorate, family))
                level = "governorate_fallback"
                if chosen is None:
                    chosen = national_lookup.get(family)
                    level = "national_fallback"

            if chosen is None:
                continue

            support_count = safe_int(chosen.get("support_count"), 0)
            surface_m2 = safe_float(chosen.get("surface_m2"), 0.0)
            rooms = safe_int(chosen.get("rooms"), 0)
            benchmark_ppm2 = safe_float(chosen.get("benchmark_ppm2"), 0.0)
            benchmark_price_tnd = benchmark_ppm2 * max(surface_m2, 1.0)
            exported_level = "exact_sector" if level == "direct" else level

            family_profiles[family] = {
                "coverage_level": level,
                "support_count": support_count,
                "confidence": confidence_for(level, support_count),
                "surface_m2": round(surface_m2, 2),
                "rooms": 0 if family == "land" else rooms,
                "benchmark_ppm2": round(benchmark_ppm2, 2),
                "benchmark_price_tnd": round(benchmark_price_tnd, 2),
                "frontend_coverage_level": exported_level,
            }

        if not family_profiles:
            continue

        default_family = max(family_profiles.items(), key=lambda item: item[1]["support_count"])[0]
        default_profile = family_profiles[default_family]
        tier_counts[default_profile["frontend_coverage_level"]] = tier_counts.get(default_profile["frontend_coverage_level"], 0) + 1

        profiles[geo_row.delegation_key] = {
            "delegation_key": geo_row.delegation_key,
            "governorate": geo_row.governorate,
            "delegation": geo_row.delegation,
            "default_family": default_family,
            "profiles": {
                family: {
                    key: value
                    for key, value in profile.items()
                    if key != "frontend_coverage_level"
                }
                for family, profile in family_profiles.items()
            },
            "area_km2": round(float(geo_row.area_km2 or 0.0), 3),
            "centroid_lon": None if pd.isna(geo_row.centroid_lon) else float(geo_row.centroid_lon),
            "centroid_lat": None if pd.isna(geo_row.centroid_lat) else float(geo_row.centroid_lat),
        }

        zone_coverage.append(
            {
                "region_code": geo_row.delegation_key,
                "delegation_key": geo_row.delegation_key,
                "governorate": geo_row.governorate,
                "delegation": geo_row.delegation,
                "has_enough_data": True,
                "support_count": int(default_profile["support_count"]),
                "prediction": float(default_profile["benchmark_ppm2"]),
                "coverage_level": default_profile["frontend_coverage_level"],
                "default_family": default_family,
                "profiles": {
                    family: {
                        "family": family,
                        "property_type": family,
                        "nature": "sale",
                        "surface": float(profile["surface_m2"]),
                        "price_per_m2": float(profile["benchmark_ppm2"]),
                        "prediction": float(profile["benchmark_price_tnd"]),
                        "coverage_level": profile["frontend_coverage_level"],
                        "support_count": int(profile["support_count"]),
                    }
                    for family, profile in family_profiles.items()
                },
            }
        )

    return profiles, zone_coverage, tier_counts


def update_atlas_geojson(geojson: dict[str, Any], profiles: dict[str, Any]) -> dict[str, Any]:
    updated_features: list[dict[str, Any]] = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        delegation_key = props.get("delegation_key")
        profile = profiles.get(delegation_key)
        if not profile:
            continue
        default_family = profile["default_family"]
        family_profile = profile["profiles"][default_family]
        updated_features.append(
            {
                **feature,
                "properties": {
                    **props,
                    "region_code": delegation_key,
                    "name_fr": profile["delegation"],
                    "NomDelegat": profile["delegation"],
                    "default_family": default_family,
                    "coverage_level": "exact_sector" if family_profile["coverage_level"] == "direct" else family_profile["coverage_level"],
                    "support_count": int(family_profile["support_count"]),
                    "benchmark_ppm2": float(family_profile["benchmark_ppm2"]),
                    "benchmark_price_tnd": float(family_profile["benchmark_price_tnd"]),
                },
            }
        )
    return {"type": "FeatureCollection", "features": updated_features}


def save_plots(results: list[dict[str, Any]], holdout: pd.DataFrame) -> dict[str, str]:
    report_paths: dict[str, str] = {}

    comparison = pd.DataFrame(results).sort_values("r2", ascending=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(comparison["model"], comparison["r2"], color=["#ef4444", "#f97316"])
    plt.title("Model Comparison")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=180)
    plt.close()
    report_paths["model_comparison"] = str(MODEL_COMPARISON_PLOT_PATH.relative_to(PROJECT_ROOT))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(holdout["price_tnd"], holdout["predicted_price_tnd"], s=14, alpha=0.35, color="#ef4444")
    min_price = float(min(holdout["price_tnd"].min(), holdout["predicted_price_tnd"].min()))
    max_price = float(max(holdout["price_tnd"].max(), holdout["predicted_price_tnd"].max()))
    plt.plot([min_price, max_price], [min_price, max_price], linestyle="--", color="#cbd5e1")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual Price (TND)")
    plt.ylabel("Predicted Price (TND)")
    plt.title("Holdout Predictions")
    plt.tight_layout()
    plt.savefig(HOLDOUT_SCATTER_PLOT_PATH, dpi=180)
    plt.close()
    report_paths["holdout_scatter"] = str(HOLDOUT_SCATTER_PLOT_PATH.relative_to(PROJECT_ROOT))

    return report_paths


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_processed_dirs()

    feature_module = load_feature_engineering_module()
    training_dataset_report = json.loads(TRAINING_DATASET_REPORT_PATH.read_text(encoding="utf-8")) if TRAINING_DATASET_REPORT_PATH.exists() else {}
    support_frame = pd.read_csv(INPUT_PATH)
    support_frame["surface_m2"] = pd.to_numeric(support_frame["surface_m2"], errors="coerce")
    support_frame["rooms"] = pd.to_numeric(support_frame["rooms"], errors="coerce")
    support_frame["price_tnd"] = pd.to_numeric(support_frame["price_tnd"], errors="coerce")
    support_frame["price_per_m2"] = pd.to_numeric(support_frame["price_per_m2"], errors="coerce")
    support_frame["log_price_tnd"] = pd.to_numeric(support_frame["log_price_tnd"], errors="coerce")

    support_frame = support_frame[
        support_frame["surface_m2"].notna()
        & support_frame["price_tnd"].notna()
        & support_frame["property_family"].isin(FAMILIES)
        & support_frame["geo_governorate"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    coord_rows_before_imputation = int((support_frame["lon"].notna() & support_frame["lat"].notna()).sum())
    geo_rows_before_imputation = int(
        support_frame["geo_delegation"].fillna("").astype(str).str.strip().ne("").sum()
    )

    frame = support_frame.copy()
    print(f"Loaded {len(frame)} valid rows")

    lower = frame["price_tnd"].quantile(0.005)
    upper = frame["price_tnd"].quantile(0.995)
    frame = frame[(frame["price_tnd"] >= lower) & (frame["price_tnd"] <= upper)].copy()
    print(f"After outlier removal: {len(frame)} rows")

    coord_rows_after_outlier_filter = int((frame["lon"].notna() & frame["lat"].notna()).sum())
    geo_rows_after_outlier_filter = int(frame["geo_delegation"].fillna("").astype(str).str.strip().ne("").sum())

    numeric_features = ["surface_m2", "rooms"]
    categorical_features = ["property_family", "geo_governorate", "geo_delegation"]
    all_features = numeric_features + categorical_features

    x = frame[all_features].copy()
    x["lon"] = frame["lon"].fillna(frame["lon"].median())
    x["lat"] = frame["lat"].fillna(frame["lat"].median())
    y = frame["log_price_tnd"]

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, frame.index, test_size=0.2, random_state=RANDOM_STATE
    )

    frame_train = frame.loc[idx_train]
    frame_test = frame.loc[idx_test]
    x_train, x_test = feature_module.add_split_training_features(x_train, x_test, frame_train, frame_test)

    numeric_final = numeric_features + ["lon", "lat", "geo_delegation_target_enc", "geo_governorate_target_enc", "price_vs_local_median"]
    final_features = numeric_final + categorical_features
    x_train = x_train[final_features]
    x_test = x_test[final_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_final),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            ),
        ]
    )

    gb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=5,
                    min_samples_leaf=5,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    gb_pipeline.fit(x_train, y_train)
    gb_pred = np.expm1(gb_pipeline.predict(x_test))
    gb_true = np.expm1(y_test)
    gb_r2 = float(r2_score(gb_true, gb_pred))
    gb_mae = float(mean_absolute_error(gb_true, gb_pred))

    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features=0.5,
                    max_depth=18,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(x_train, y_train)
    rf_pred = np.expm1(rf_pipeline.predict(x_test))
    rf_r2 = float(r2_score(gb_true, rf_pred))
    rf_mae = float(mean_absolute_error(gb_true, rf_pred))

    results = [
        {"model": "GradientBoosting", "r2": gb_r2, "mae": gb_mae, "rmse": None},
        {"model": "RandomForest", "r2": rf_r2, "mae": rf_mae, "rmse": None},
    ]

    if gb_r2 >= rf_r2:
        best_name = "GradientBoosting"
        best_pipeline = gb_pipeline
        best_r2 = gb_r2
        best_mae = gb_mae
    else:
        best_name = "RandomForest"
        best_pipeline = rf_pipeline
        best_r2 = rf_r2
        best_mae = rf_mae

    cv_scores = cross_val_score(best_pipeline, x_train, y_train, cv=5, scoring="r2", n_jobs=-1)

    holdout = frame.loc[idx_test, [
        "source_dataset",
        "source_row_id",
        "property_family",
        "normalized_locality",
        "geo_governorate",
        "geo_delegation",
        "surface_m2",
        "price_tnd",
    ]].copy()
    holdout["predicted_price_tnd"] = np.expm1(best_pipeline.predict(x_test))
    holdout["error_tnd"] = holdout["predicted_price_tnd"] - holdout["price_tnd"]
    holdout["error_pct"] = (holdout["error_tnd"] / holdout["price_tnd"] * 100).round(2)
    holdout.to_csv(PREDICTIONS_PATH, index=False)

    geojson, geo_frame = load_geography_artifacts()
    profiles, zone_coverage, tier_counts = build_frontend_profiles(support_frame, geo_frame)
    atlas_geojson = update_atlas_geojson(geojson, profiles)
    write_json(FRONTEND_PROFILES_PATH, profiles)
    write_json(FRONTEND_ZONE_COVERAGE_PATH, zone_coverage)
    FRONTEND_ATLAS_PATH.write_text(json.dumps(atlas_geojson, ensure_ascii=False), encoding="utf-8")

    plot_reports = save_plots(results, holdout)

    delegations_with_direct_support = sum(
        1 for profile in profiles.values()
        if any(family_profile["coverage_level"] == "direct" for family_profile in profile["profiles"].values())
    )
    total_delegations = int(len(geo_frame))
    covered_delegations = int(len(profiles))
    tier_percentages = {
        key: round((value / max(covered_delegations, 1)) * 100, 2)
        for key, value in sorted(tier_counts.items())
    }
    benchmark_values = [record["prediction"] for record in zone_coverage if record.get("prediction") is not None]

    summary = {
        "best_model": best_name,
        "training_rows": int(training_dataset_report.get("rows") or len(support_frame)),
        "modeling_rows": int(len(frame)),
        "test_r2": best_r2,
        "validation_r2": best_r2,
        "test_mae": best_mae,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "accuracy_pct": round(best_r2 * 100, 2),
        "total_delegations": total_delegations,
        "covered_delegations": covered_delegations,
        "delegations_with_direct_support": int(delegations_with_direct_support),
        "direct_coverage_pct": round((delegations_with_direct_support / max(total_delegations, 1)) * 100, 2),
        "atlas_reach_pct": round((covered_delegations / max(total_delegations, 1)) * 100, 2),
        "fallback_support_pct": round(100 - ((delegations_with_direct_support / max(total_delegations, 1)) * 100), 2),
        "tier_percentages": tier_percentages,
        "benchmark_ppm2_range": {
            "min": round(float(min(benchmark_values)) if benchmark_values else 0.0, 2),
            "max": round(float(max(benchmark_values)) if benchmark_values else 0.0, 2),
        },
        "model_results": results,
        "base_feature_columns": list(getattr(feature_module, "BASE_FEATURE_COLUMNS", ["surface_m2", "rooms", "lon", "lat"])),
        "split_dependent_feature_columns": list(
            getattr(
                feature_module,
                "SPLIT_DEPENDENT_FEATURE_COLUMNS",
                ["geo_delegation_target_enc", "geo_governorate_target_enc", "price_vs_local_median"],
            )
        ),
        "feature_groups": dict(
            getattr(
                feature_module,
                "FEATURE_GROUPS",
                {
                    "base": ["surface_m2", "rooms", "lon", "lat"],
                    "split_dependent": ["geo_delegation_target_enc", "geo_governorate_target_enc", "price_vs_local_median"],
                    "categorical": ["property_family", "geo_governorate", "geo_delegation"],
                    "final_model": final_features,
                },
            )
        ),
        "feature_descriptions": dict(getattr(feature_module, "FEATURE_DESCRIPTIONS", {})),
        "stage6_note": getattr(
            feature_module,
            "STAGE6_FEATURE_NOTE",
            "Stage 6 writes the base engineered dataset with canonical geographic coordinates.",
        ),
        "stage8_note": "These are the final model features after Stage 8 augmentation and train/test-split-safe feature creation.",
        "features": final_features,
        "feature_columns": final_features,
        "coord_coverage": {
            "rows_with_geo_delegation_before_imputation": geo_rows_before_imputation,
            "rows_with_coords_before_imputation": coord_rows_before_imputation,
            "rows_missing_coords_despite_geo_match_before_imputation": geo_rows_before_imputation - coord_rows_before_imputation,
            "rows_with_geo_delegation_after_outlier_filter": geo_rows_after_outlier_filter,
            "rows_with_coords_after_outlier_filter": coord_rows_after_outlier_filter,
            "rows_missing_coords_despite_geo_match_after_outlier_filter": geo_rows_after_outlier_filter - coord_rows_after_outlier_filter,
            "numeric_coord_imputation_strategy": "median",
        },
        "reports": plot_reports,
        "frontend_exports": {
            "atlas_geojson": str(FRONTEND_ATLAS_PATH.relative_to(PROJECT_ROOT)),
            "zone_coverage": str(FRONTEND_ZONE_COVERAGE_PATH.relative_to(PROJECT_ROOT)),
            "delegation_profiles": str(FRONTEND_PROFILES_PATH.relative_to(PROJECT_ROOT)),
            "model_summary": str(FRONTEND_MODEL_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    pipeline_manifest = {
        "generated_at": "pipeline/08_model_training.py",
        "core_assets": summary["frontend_exports"],
        "reports": {
            "discovery": str((PROJECT_ROOT / "data" / "processed" / "01_discovery" / "01_merge_overview.json").relative_to(PROJECT_ROOT)),
            "cleaning": str((PROJECT_ROOT / "data" / "processed" / "02_cleaning" / "02_merge_readiness.json").relative_to(PROJECT_ROOT)),
            "merge": str((PROJECT_ROOT / "data" / "processed" / "03_merge" / "03_merge_report.json").relative_to(PROJECT_ROOT)),
            "geo_alignment": str((PROJECT_ROOT / "data" / "processed" / "04_geo_alignment" / "04_geo_alignment_report.json").relative_to(PROJECT_ROOT)),
            "training_dataset": str((PROJECT_ROOT / "data" / "processed" / "05_training_dataset" / "05_training_dataset_report.json").relative_to(PROJECT_ROOT)),
            "feature_engineering": str((PROJECT_ROOT / "data" / "processed" / "06_feature_engineering" / "06_feature_engineering_report.json").relative_to(PROJECT_ROOT)),
            "visual_check": str((PROJECT_ROOT / "data" / "processed" / "07_visual_check" / "07_training_dataset_visual_report.json").relative_to(PROJECT_ROOT)),
            "model_training": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "holdout_predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "model_comparison_plot": str(MODEL_COMPARISON_PLOT_PATH.relative_to(PROJECT_ROOT)),
            "holdout_scatter_plot": str(HOLDOUT_SCATTER_PLOT_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    write_json(METRICS_PATH, summary)
    write_json(FRONTEND_MODEL_SUMMARY_PATH, summary)
    write_json(FRONTEND_PIPELINE_MANIFEST_PATH, pipeline_manifest)

    artifact = {
        "pipeline": best_pipeline,
        "features": final_features,
        "feature_columns": final_features,
        "profiles": profiles,
        "summary": summary,
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
