# Pipeline

This folder contains the active V2 numbered workflow.

## Execution order

1. `01_dataset_discovery.py`
2. `02_dataset_cleaning.py`
3. `03_merge_preparation.py`
4. `04_geo_name_alignment.py`
5. `05_training_dataset_preparation.py`
6. `06_feature_engineering.py`
7. `07_training_dataset_visual_check.py`
8. `08_model_training.py`
9. `09_run_pipeline.py` to execute stages 1-8 in order

## Inputs

- `data/raw/tunisia-real-estate.csv`
- `data/raw/Property Prices in Tunisia.csv`
- `data/raw/data_prices_cleaned.csv`
- geography assets under `geo/`

## Outputs

- stage-organized outputs under `data/processed/01_discovery/` through `data/processed/08_model_training/`
- final training dataset in `data/processed/05_training_dataset/05_training_dataset.csv`
- engineered feature dataset in `data/processed/06_feature_engineering/06_feature_engineered_dataset.csv`
- model report in `data/processed/08_model_training/08_model_training_report.json`
- trained model in `artifacts/08_best_model.joblib`

## Notes

- The API reads trained artifacts and processed outputs from V2 only.
- The live frontend summary is derived from V2 coverage data and V2 training outputs.
- `frontend/workflow_notebook.html` is a static exported notebook artifact, not the source of truth for the runtime workflow.
