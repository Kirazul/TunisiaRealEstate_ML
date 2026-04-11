from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DISCOVERY_DIR = PROCESSED_DIR / "01_discovery"
CLEANING_DIR = PROCESSED_DIR / "02_cleaning"
MERGE_DIR = PROCESSED_DIR / "03_merge"
GEO_ALIGNMENT_DIR = PROCESSED_DIR / "04_geo_alignment"
TRAINING_DATASET_DIR = PROCESSED_DIR / "05_training_dataset"
FEATURE_ENGINEERING_DIR = PROCESSED_DIR / "06_feature_engineering"
VISUAL_CHECK_DIR = PROCESSED_DIR / "07_visual_check"
MODEL_TRAINING_DIR = PROCESSED_DIR / "08_model_training"


def ensure_processed_dirs() -> None:
    for path in [
        PROCESSED_DIR,
        DISCOVERY_DIR,
        CLEANING_DIR,
        MERGE_DIR,
        GEO_ALIGNMENT_DIR,
        TRAINING_DATASET_DIR,
        FEATURE_ENGINEERING_DIR,
        VISUAL_CHECK_DIR,
        MODEL_TRAINING_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
