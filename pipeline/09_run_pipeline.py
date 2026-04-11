from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_dataset_discovery.py",
    "02_dataset_cleaning.py",
    "03_merge_preparation.py",
    "04_geo_name_alignment.py",
    "05_training_dataset_preparation.py",
    "06_feature_engineering.py",
    "07_training_dataset_visual_check.py",
    "08_model_training.py",
]


def main() -> None:
    for step in STEPS:
        step_path = PIPELINE_DIR / step
        print(f"\n=== Running {step} ===")
        subprocess.run([sys.executable, str(step_path)], check=True)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
