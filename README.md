![Tunisia Real Estate AI](./assets/github_header.png)

# Tunisia Real Estate Price Prediction System

A complete end-to-end machine learning pipeline for predicting property prices across Tunisia. The system combines data processing, geographic alignment, feature engineering, and gradient boosting to deliver accurate price predictions with full national coverage through intelligent fallback mechanisms.

**Live Demo**: https://tunisiarealestate-ml.onrender.com/

---

## Problem Statement

Real estate pricing in Tunisia varies dramatically by location. A 100 m2 apartment in Tunis costs fundamentally differently than the same property in Sfax or Kebili. This system addresses the challenge of predicting property prices across 264 delegations, where many regions lack sufficient training data, by combining direct modeling with hierarchical fallback benchmarks.

---

## Architecture Overview

The system consists of three interconnected layers:

| Layer | Description |
|-------|------------|
| **Data Pipeline** | Eight-stage transformation from raw listings to trained model |
| **Prediction API** | FastAPI serving real-time predictions and model diagnostics |
| **Geographic Atlas** | Interactive frontend displaying coverage, predictions, and fallback tiers |

All components share common data artifacts, ensuring consistency between training and serving.

---

## Pipeline Stages

The pipeline transforms raw property listings through eight sequential stages:

### Stage 1: Dataset Discovery

Location: `pipeline/01_dataset_discovery.py`

> Loads and profiles raw data sources. Identifies column mappings for price, surface, location, and property type across three source datasets. Produces discovery profiles documenting data quality, value distributions, and schema assessments.

### Stage 2: Dataset Cleaning

Location: `pipeline/02_dataset_cleaning.py`

> Applies standardization rules across datasets. Handles encoding issues, trims whitespace, normalizes text fields, and filters obvious bad data. Removes duplicates based on key attribute combinations. Produces cleaned datasets ready for merging.

### Stage 3: Merge Preparation

Location: `pipeline/03_merge_preparation.py`

> Unifies multiple source datasets into a single training corpus. Resolves schema differences between sources, maps property types to consistent categories, and prepares join keys for geographic alignment. Produces a merged dataset with unified schema.

### Stage 4: Geographic Name Alignment

Location: `pipeline/04_geo_name_alignment.py`

> Maps textual location names to official Tunisia geography. Uses fuzzy matching against delegation and governorate names, with alias mappings for common variations (e.g., "Ariana Ville" -> "Ariana", "La Soukra" -> "La Soukra"). Produces geo-aligned dataset with standardized delegation keys.

### Stage 5: Training Dataset Preparation

Location: `pipeline/05_training_dataset_preparation.py`

> Prepares the final training dataset. Creates binary property family indicators (apartment, house, land), filters invalid rows, and structures data for modeling. Produces training-ready CSV with all required columns.

### Stage 6: Feature Engineering

Location: `pipeline/06_feature_engineering.py`

> Transforms raw features into model-ready representations. Creates log-transformed price and surface, generates target-encoded geographic features (delegation and governorate mean prices), computes price-vs-local-median ratios, and prepares final feature matrices.

### Stage 7: Visual Check

Location: `pipeline/07_training_dataset_visual_check.py`

> Performs quality assurance on the training dataset. Samples records across regions, validates geographic distributions, and generates diagnostic visualizations. Ensures data quality before model training.

### Stage 8: Model Training

Location: `pipeline/08_model_training.py`

> Trains and evaluates prediction models. Compares GradientBoosting and RandomForest regressors, performs cross-validation, selects the best model, and generates performance reports. Exports the trained model artifact and frontend-ready coverage data.

---

## Fallback System

The geographic fallback system ensures predictions for all 264 delegations, even those without direct training data:

| Tier | Description |
|------|-------------|
| **Exact Delegation** | Sufficient data in the delegation (direct support) |
| **Locality Fallback** | Data from broader locality |
| **Delegation Fallback** | Delegation-level aggregated data |
| **Governorate Fallback** | Governorate-wide aggregated data |
| **National Fallback** | Country-wide average |

Each delegation receives a coverage level and benchmark prediction based on the best available data in its hierarchy. This approach guarantees 100% national coverage while distinguishing areas with direct evidence from those using borrowed benchmarks.

---

## Model Performance

Current production model metrics:

- **Algorithm**: Gradient Boosting Regressor
- **Cross-Validation R2**: 0.928 +/- 0.003
- **Test Set R2**: 0.890
- **Test MAE**: 69,941 TND
- **Direct Coverage**: 103 delegations (39.02%)
- **Fallback Coverage**: 161 delegations (60.98%)
- **Atlas Reach**: 100% of Tunisia delegations

**Features used**: surface_m2, rooms, lon, lat, geo_delegation_target_enc, geo_governorate_target_enc, price_vs_local_median, property_family, geo_governorate, geo_delegation

## API Endpoints

The FastAPI application provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict property price for given attributes |
| `/model_summary` | GET | Retrieve model performance and coverage metrics |
| `/delegation/{key}` | GET | Get coverage profile for a delegation |
| `/coverage` | GET | Full coverage data for all delegations |
| `/search?q={query}` | GET | Search delegations by name |
| `/execute` | POST | Run notebook cells via API |

## Frontend Components

### Geographic Atlas

Location: `frontend/index.html`

Interactive map displaying all 264 Tunisia delegations. Supports two view modes:

- **Support mode**: Shows direct support (red) vs fallback (dark red) delegations
- **Exact mode**: Shows only delegations with direct training data

Features delegation search, selection detail panels, surface adjustment slider, and hover tooltips.

### Pipeline Notebook

Location: `frontend/notebook.html`

Static notebook artifact displaying the complete pipeline with explanations. Eight chapters corresponding to pipeline stages, with code cells, markdown documentation, and execution status.

## Data Sources

Input datasets located in `data/raw/`:

- `tunisia-real-estate.csv` - Primary listings source
- `Property Prices in Tunisia.csv` - Secondary price data
- `data_prices_cleaned.csv` - Pre-cleaned subset

Geography data in `geo/`:

- Delegation boundaries and centroids
- Governorate mappings
- Region codes

## Output Artifacts

The pipeline generates artifacts in several locations:

### Model Artifacts

- `artifacts/08_best_model.joblib` - Trained GradientBoosting model

### Processed Data

```
data/processed/
├── 01_discovery/              # Dataset profiles
├── 02_cleaning/               # Cleaned datasets
├── 03_merge/                 # Merged dataset + report
├── 04_geo_alignment/         # Geo-aligned data + report
├── 05_training_dataset/       # Training-ready CSV
├── 06_feature_engineering/  # Feature matrices
├── 07_visual_check/         # Diagnostic samples
└── 08_model_training/       # Model report + plots
```

### Frontend Exports

```
frontend/assets/data/
├── atlas.geojson             # Delegation boundaries
├── zone_coverage.json        # Coverage profiles per delegation
├── delegation_profiles.json  # Detailed profiles by region
frontend/model_summary.json  # Model metrics for UI
```

### Visualization Reports

- `data/processed/08_model_training/08_model_comparison.png` - Model performance comparison
- `data/processed/08_model_training/08_holdout_scatter.png` - Predicted vs actual scatter plot

## Quick Start

### Running the Pipeline

Execute all pipeline stages:

```bash
cd RealEstate/pipeline
python 09_run_pipeline.py
```

Execute individual stages:

```bash
python 01_dataset_discovery.py
python 02_dataset_cleaning.py
# ... through 08_model_training.py
```

### Running the API

```bash
cd RealEstate_/api
uvicorn app:app --reload --port 8000
```

Access API documentation at `http://localhost:8000/docs`

### Running the Frontend

Serve the frontend:

```bash
cd RealEstate/frontend
python -m http.server 8080
```

Open `http://localhost:8080` in browser

### Jupyter Notebook

Open and execute the complete pipeline in notebook:

```bash
jupyter notebook RealEstate_Complete_Pipeline.ipynb
```

## Project Structure

```
RealEstate/
├── api/                      # FastAPI application
├── artifacts/               # Trained model files
├── data/
│   ├── raw/                 # Source datasets
│   └── processed/           # Pipeline outputs (stages 1-8)
├── frontend/
│   ├── assets/data/         # Frontend data exports
│   ├── css/               # Stylesheets
│   ├── js/                # Frontend JavaScript
│   ├── index.html         # Geographic atlas
│   └── notebook.html      # Pipeline notebook UI
├── geo/                    # Tunisia geography data
├── pipeline/              # Pipeline scripts (stages 1-8)
├── README.md              # This file
└── QUICK_START.md       # Quick start guide
```

## Key Design Decisions

### Why Gradient Boosting

Gradient Boosting outperformed RandomForest in cross-validation (0.928 vs 0.859 R2) while maintaining reasonable inference speed. The algorithm handles the mixed numeric/categorical feature space effectively.

### Why Target Encoding

Target encoding (mean price per delegation/governorate) provides compact geographic signal without one-hot encoding explosion. Combined with coordinate features, it enables smooth location-based predictions while maintaining interpretability.

### Why Hierarchical Fallbacks

Some delegations have zero listings. Rather than refusing predictions, the fallback system borrows evidence from broader regions, guaranteeing predictions everywhere while transparently documenting data quality.

### Why Log Transform

Real estate prices follow approximately log-normal distributions. Log-transforming price during training makes the target more Gaussian, improving model stability and performance.

## Dependencies

Key Python packages:

- pandas, numpy - Data processing
- scikit-learn - Machine learning
- joblib - Model serialization
- matplotlib - Visualization
- fastapi, uvicorn - API server
- d3 - Frontend visualizations

See `requirements.txt` or environment-specific dependency files for complete lists.

## Limitations and Considerations
- **Temporal Scope**: Training data reflects market conditions at collection time
- **Coverage Quality**: Direct support delegations have better prediction accuracy than fallback areas
- **Surface Range**: Model trained on properties 20-1000 m2

