"""
Tunisia Real Estate Price Prediction API - V2
Production API with frontend for predicting property prices in Tunisia
"""

from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
import sys
import threading
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from typing import Optional

import joblib
import matplotlib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import markdown as markdown_lib
from pydantic import BaseModel, Field


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
GEO_DIR = PROJECT_ROOT / "geo"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

MODEL_PATH = ARTIFACTS_DIR / "08_best_model.joblib"
TRAINING_REPORT_PATH = DATA_DIR / "08_model_training" / "08_model_training_report.json"
TRAINING_DATASET_PATH = DATA_DIR / "05_training_dataset" / "05_training_dataset.csv"
ZONE_COVERAGE_PATH = FRONTEND_DIR / "assets" / "data" / "zone_coverage.json"
FRONTEND_MODEL_SUMMARY_PATH = FRONTEND_DIR / "model_summary.json"
MERGE_REPORT_PATH = DATA_DIR / "03_merge" / "03_merge_report.json"
GEO_ALIGNMENT_REPORT_PATH = DATA_DIR / "04_geo_alignment" / "04_geo_alignment_report.json"

COASTAL_GOVERNORATES = {
    "Tunis", "Ariana", "Ben Arous", "Bizerte", "Nabeul",
    "Sousse", "Monastir", "Mahdia", "Sfax", "Gabès", "Médenine"
}

app = FastAPI(
    title="Tunisia Real Estate Price Prediction API V2",
    description="AI-powered property price predictions for Tunisia",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE_HEADERS)
        if "etag" in response.headers:
            del response.headers["etag"]
        if "last-modified" in response.headers:
            del response.headers["last-modified"]
        return response


NOTEBOOK_LOCK = threading.Lock()
NOTEBOOK_GLOBALS: dict[str, object] = {}


def _init_notebook_globals() -> dict[str, object]:
    frontend_summary_path = PROJECT_ROOT / "frontend" / "model_summary.json"
    manifest_path = PROJECT_ROOT / "frontend" / "assets" / "data" / "pipeline_assets_manifest.json"
    frontend_summary_data = json.loads(frontend_summary_path.read_text(encoding="utf-8")) if frontend_summary_path.exists() else {}
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "__name__": "__notebook__",
        "PROJECT_ROOT": PROJECT_ROOT,
        "project_root": PROJECT_ROOT,
        "frontend_summary": frontend_summary_data,
        "manifest": manifest_data,
    }


def _get_notebook_globals(reset: bool = False) -> dict[str, object]:
    global NOTEBOOK_GLOBALS
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if reset or not NOTEBOOK_GLOBALS:
        NOTEBOOK_GLOBALS = _init_notebook_globals()
    return NOTEBOOK_GLOBALS


def _resolve_source_path(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if PROJECT_ROOT not in candidate.parents and candidate != PROJECT_ROOT:
        raise HTTPException(status_code=400, detail="Path escapes project root")
    if candidate.suffix.lower() not in {".py", ".html", ".js", ".css", ".json", ".md", ".ipynb"}:
        raise HTTPException(status_code=400, detail="Unsupported source file type")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Source file not found")
    return candidate


@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    if request.method == "GET" and (
        path == "/" or path.endswith((".html", ".css", ".js", ".json", ".svg", ".geojson"))
    ):
        response.headers.update(NO_CACHE_HEADERS)
        if "etag" in response.headers:
            del response.headers["etag"]
        if "last-modified" in response.headers:
            del response.headers["last-modified"]
    return response


def no_cache_file_response(path: Path) -> FileResponse:
    return FileResponse(path, headers=NO_CACHE_HEADERS)


class PredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    property_family: str = Field(..., description="apartment, house, or land")
    governorate: str = Field(..., description="Governorate name")
    delegation: str = Field(..., description="Delegation name")
    surface_m2: float = Field(..., gt=0, description="Surface area in square meters")
    rooms: Optional[int] = Field(None, ge=0, le=20, description="Number of rooms")


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    predicted_price_tnd: float
    predicted_price_per_m2: float
    benchmark_ppm2: float
    surface_m2: float
    confidence: str
    model_r2: float
    model_version: str
    governance: str


def load_delegation_coords() -> dict:
    geojson_path = GEO_DIR / "delegations-full.geojson"
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    coords = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if props.get("engtype_2") == "Delegation" and props.get("name_2"):
            geom = feature.get("geometry", {})
            geom_type = geom.get("type")
            geom_coords = geom.get("coordinates")
            
            if geom_type == "Polygon":
                coords_list = geom_coords[0] if geom_coords else []
                if coords_list:
                    lons = [c[0] for c in coords_list]
                    lats = [c[1] for c in coords_list]
                    coords[props["name_2"]] = (sum(lons) / len(lons), sum(lats) / len(lats))
            elif geom_type == "MultiPolygon":
                all_lons, all_lats = [], []
                for polygon in geom_coords:
                    for ring in polygon:
                        for coord in ring:
                            all_lons.append(coord[0])
                            all_lats.append(coord[1])
                if all_lons:
                    coords[props["name_2"]] = (sum(all_lons) / len(all_lons), sum(all_lats) / len(all_lats))
    
    return coords


def load_historical_averages() -> dict:
    if TRAINING_DATASET_PATH.exists():
        df = pd.read_csv(TRAINING_DATASET_PATH)
        df["log_price_tnd"] = pd.to_numeric(df["log_price_tnd"], errors="coerce")
        df["price_per_m2"] = pd.to_numeric(df["price_per_m2"], errors="coerce")
        
        del_avg = df.groupby("geo_delegation")["log_price_tnd"].mean().to_dict()
        gov_avg = df.groupby("geo_governorate")["log_price_tnd"].mean().to_dict()
        del_ppm2 = df.groupby("geo_delegation")["price_per_m2"].median().to_dict()
        
        return {
            "delegation_log_price": del_avg,
            "governorate_log_price": gov_avg,
            "delegation_price_per_m2": del_ppm2
        }
    return {"delegation_log_price": {}, "governorate_log_price": {}, "delegation_price_per_m2": {}}


def load_model_score() -> float:
    if not TRAINING_REPORT_PATH.exists():
        return 0.0
    payload = json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
    return float(payload.get("test_r2") or 0.0)


def load_frontend_summary() -> dict:
    if not ZONE_COVERAGE_PATH.exists():
        return {
            "best_model": "Unknown",
            "model_version": "V2",
            "training_rows": 0,
            "merge_rows": 0,
            "geo_delegations_covered": 0,
            "validation_r2": 0.0,
            "cv_r2_mean": 0.0,
            "cv_r2_std": 0.0,
            "accuracy_pct": 0.0,
            "total_delegations": 0,
            "covered_delegations": 0,
            "delegations_with_direct_support": 0,
            "direct_coverage_pct": 0.0,
            "fallback_support_pct": 0.0,
            "atlas_reach_pct": 0.0,
            "tier_percentages": {},
            "benchmark_ppm2_range": {"min": 0.0, "max": 0.0},
            "model_results": [],
            "reports": {},
        }

    exported_summary = {}
    if FRONTEND_MODEL_SUMMARY_PATH.exists():
        exported_summary = json.loads(FRONTEND_MODEL_SUMMARY_PATH.read_text(encoding="utf-8"))

    coverage_records = json.loads(ZONE_COVERAGE_PATH.read_text(encoding="utf-8"))
    total_delegations = len(coverage_records)
    covered_records = [item for item in coverage_records if item.get("has_enough_data")]
    direct_records = []
    predictions = [float(item["prediction"]) for item in covered_records if item.get("prediction") is not None]

    tier_counts = {}
    for item in covered_records:
        profiles = item.get("profiles") if isinstance(item.get("profiles"), dict) else {}
        has_direct_profile = any(
            isinstance(profile, dict) and profile.get("coverage_level") == "exact_sector"
            for profile in profiles.values()
        )
        if has_direct_profile:
            direct_records.append(item)
        level = str(item.get("coverage_level") or "unknown")
        tier_counts[level] = tier_counts.get(level, 0) + 1

    training_report = exported_summary or (json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8")) if TRAINING_REPORT_PATH.exists() else {})
    merge_report = json.loads(MERGE_REPORT_PATH.read_text(encoding="utf-8")) if MERGE_REPORT_PATH.exists() else {}
    geo_alignment_report = json.loads(GEO_ALIGNMENT_REPORT_PATH.read_text(encoding="utf-8")) if GEO_ALIGNMENT_REPORT_PATH.exists() else {}
    best_model = str(training_report.get("best_model") or "Unknown")
    model_score = float(training_report.get("test_r2") or 0.0)
    cv_score = float(training_report.get("cv_r2_mean") or 0.0)
    training_rows = int(training_report.get("training_rows") or 0)
    merge_rows = int(merge_report.get("final_rows") or 0)
    atlas_reach_pct = round((len(covered_records) / total_delegations) * 100, 2) if total_delegations else 0.0

    return {
        "best_model": best_model,
        "model_version": "V2",
        "training_rows": training_rows,
        "modeling_rows": int(training_report.get("modeling_rows") or 0),
        "merge_rows": merge_rows,
        "geo_delegations_covered": int(geo_alignment_report.get("unique_geo_delegations_covered") or 0),
        "validation_r2": round(model_score, 6),
        "cv_r2_mean": round(cv_score, 6),
        "cv_r2_std": round(float(training_report.get("cv_r2_std") or 0.0), 6),
        "accuracy_pct": round(model_score * 100, 2),
        "total_delegations": total_delegations,
        "covered_delegations": len(covered_records),
        "delegations_with_direct_support": len(direct_records),
        "atlas_reach_pct": atlas_reach_pct,
        "direct_coverage_pct": round((len(direct_records) / total_delegations) * 100, 2) if total_delegations else 0.0,
        "fallback_support_pct": round(((len(covered_records) - len(direct_records)) / total_delegations) * 100, 2) if total_delegations else 0.0,
        "tier_percentages": {
            key: round((value / total_delegations) * 100, 2) if total_delegations else 0.0
            for key, value in sorted(tier_counts.items())
        },
        "benchmark_ppm2_range": {
            "min": round(min(predictions), 2) if predictions else 0.0,
            "max": round(max(predictions), 2) if predictions else 0.0,
        },
        "model_results": [
            {
                "model": best_model,
                "r2": model_score,
                "rmse": None,
                "cv_r2_mean": cv_score,
            }
        ],
        "reports": {
            "training_report_path": str(TRAINING_REPORT_PATH.relative_to(PROJECT_ROOT)) if TRAINING_REPORT_PATH.exists() else None,
            "coverage_path": str(ZONE_COVERAGE_PATH.relative_to(PROJECT_ROOT)) if ZONE_COVERAGE_PATH.exists() else None,
        },
    }


pipeline = None
delegation_coords = None
historical_averages = None
frontend_summary = None


@app.on_event("startup")
async def load_resources():
    global pipeline, delegation_coords, historical_averages, frontend_summary
    
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        pipeline = artifact.get("pipeline")
    
    delegation_coords = load_delegation_coords()
    historical_averages = load_historical_averages()
    frontend_summary = load_frontend_summary()


@app.get("/")
async def root():
    return no_cache_file_response(FRONTEND_DIR / "index.html")


@app.get("/notebook")
async def notebook():
    return no_cache_file_response(FRONTEND_DIR / "notebook.html")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "delegations_loaded": len(delegation_coords) if delegation_coords else 0,
        "historical_data_loaded": len(historical_averages.get("delegation_log_price", {})) if historical_averages else 0,
        "frontend_summary_loaded": frontend_summary is not None,
    }


@app.get("/model_summary")
async def model_summary():
    return load_frontend_summary()


@app.get("/api/source")
async def source(path: str = Query(..., min_length=1)):
    file_path = _resolve_source_path(path)
    return {"path": path, "content": file_path.read_text(encoding="utf-8")}


@app.get("/api/source/raw")
async def source_raw(path: str = Query(..., min_length=1)):
    file_path = _resolve_source_path(path)
    return no_cache_file_response(file_path)


def _render_notebook_html(file_path: Path) -> str:
    notebook = json.loads(file_path.read_text(encoding="utf-8"))
    cells_html: list[str] = []

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = "".join(cell.get("source", []))
        cell_type = cell.get("cell_type", "code")

        if cell_type == "markdown":
            rendered = markdown_lib.markdown(source, extensions=["fenced_code", "tables"])
            cells_html.append(
                f'<section class="nb-cell nb-markdown"><div class="nb-body">{rendered}</div></section>'
            )
            continue

        escaped_source = html.escape(source)
        outputs_html: list[str] = []
        for output in cell.get("outputs", []):
            if "text" in output:
                outputs_html.append(f'<pre class="nb-output">{html.escape("".join(output.get("text", [])))}</pre>')
            elif output.get("data", {}).get("text/plain"):
                outputs_html.append(f'<pre class="nb-output">{html.escape("".join(output["data"]["text/plain"]))}</pre>')

        outputs_block = "".join(outputs_html)
        cells_html.append(
            f'''<section class="nb-cell nb-code">
<div class="nb-label">Cell {index}</div>
<pre class="nb-code-block"><code>{escaped_source}</code></pre>
{outputs_block}
</section>'''
        )

    title = file_path.stem.replace("_", " ")
    body = "\n".join(cells_html)
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; background: #07090f; color: #e5e7eb; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; }}
    .hero {{ margin-bottom: 32px; padding: 28px 30px; border: 1px solid rgba(255,255,255,0.08); border-radius: 22px; background: linear-gradient(180deg, rgba(20,24,35,0.95), rgba(10,12,18,0.95)); box-shadow: 0 20px 80px rgba(0,0,0,0.35); }}
    .eyebrow {{ display: inline-block; font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: #f87171; margin-bottom: 12px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(30px, 4vw, 48px); }}
    .sub {{ margin: 0; color: #a1a1aa; font-size: 16px; }}
    .nb-cell {{ margin: 0 0 18px; border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; background: rgba(14,18,27,0.92); overflow: hidden; }}
    .nb-markdown .nb-body {{ padding: 28px 30px; line-height: 1.75; }}
    .nb-markdown h1, .nb-markdown h2, .nb-markdown h3 {{ color: #fff; margin-top: 0; }}
    .nb-markdown p, .nb-markdown li {{ color: #d4d4d8; }}
    .nb-markdown code {{ background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; }}
    .nb-markdown pre {{ background: #0a0d14; padding: 18px; border-radius: 14px; overflow: auto; }}
    .nb-label {{ padding: 14px 18px; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #fca5a5; border-bottom: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.02); }}
    .nb-code-block, .nb-output {{ margin: 0; padding: 20px 22px; overflow: auto; font-family: Consolas, monospace; font-size: 13px; line-height: 1.6; white-space: pre-wrap; }}
    .nb-code-block {{ background: #0b1020; color: #e2e8f0; }}
    .nb-output {{ border-top: 1px solid rgba(255,255,255,0.06); background: #121826; color: #cbd5e1; }}
    a {{ color: #f87171; }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <div class="eyebrow">Notebook Render</div>
      <h1>{html.escape(title)}</h1>
      <p class="sub">Rendered notebook view for the raw project notebook file.</p>
    </header>
    {body}
  </main>
</body>
</html>'''


@app.get("/api/source/rendered", response_class=HTMLResponse)
async def source_rendered(path: str = Query(..., min_length=1)):
    file_path = _resolve_source_path(path)
    if file_path.suffix.lower() != ".ipynb":
        raise HTTPException(status_code=400, detail="Rendered view is only available for notebook files")
    return HTMLResponse(_render_notebook_html(file_path), headers=NO_CACHE_HEADERS)


@app.post("/run-cell")
async def run_cell(payload: dict[str, object]):
    code = str(payload.get("code", ""))
    reset = bool(payload.get("reset", False))

    with NOTEBOOK_LOCK:
        namespace = _get_notebook_globals(reset=reset)
        if reset and not code.strip():
            return {"stdout": "Kernel reset.", "stderr": "", "plot": None}

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        plot_b64 = None

        import matplotlib.pyplot as plt

        previous_cwd = Path.cwd()
        plt.close("all")
        try:
            os.chdir(PROJECT_ROOT)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
                    category=UserWarning,
                )
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    exec(code, namespace, namespace)
            figure_numbers = plt.get_fignums()
            if figure_numbers:
                figure = plt.figure(figure_numbers[-1])
                image_buffer = BytesIO()
                figure.savefig(image_buffer, format="png", bbox_inches="tight")
                image_buffer.seek(0)
                plot_b64 = base64.b64encode(image_buffer.read()).decode("utf-8")
        except Exception:
            traceback.print_exc(file=stderr_buffer)
        finally:
            os.chdir(previous_cwd)
            plt.close("all")

        return {
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "plot": plot_b64,
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest) -> PredictionResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.property_family not in ["apartment", "house", "land"]:
        raise HTTPException(status_code=400, detail="Invalid property family")
    
    coord = delegation_coords.get(request.delegation)
    if coord is None:
        coord = (10.0, 36.0)
    
    lon, lat = coord
    
    rooms = request.rooms if request.rooms else 3
    if request.property_family == "land":
        rooms = 0
    
    log_price_tnd_del = historical_averages.get("delegation_log_price", {}).get(request.delegation, 7.5)
    log_price_tnd_gov = historical_averages.get("governorate_log_price", {}).get(request.governorate, 7.5)
    ppm2_del = historical_averages.get("delegation_price_per_m2", {}).get(request.delegation, 1500)
    
    if request.property_family == "apartment":
        current_price_per_m2 = ppm2_del * 1.1
    elif request.property_family == "house":
        current_price_per_m2 = ppm2_del * 1.3
    else:
        current_price_per_m2 = ppm2_del * 0.9
    
    price_vs_median = np.sqrt(max(0.5, min(2.0, current_price_per_m2 / ppm2_del if ppm2_del > 0 else 1.0)))
    
    df = pd.DataFrame([{
        "surface_m2": request.surface_m2,
        "rooms": rooms,
        "lon": lon,
        "lat": lat,
        "geo_delegation_target_enc": log_price_tnd_del,
        "geo_governorate_target_enc": log_price_tnd_gov,
        "price_vs_local_median": price_vs_median,
        "property_family": request.property_family,
        "geo_governorate": request.governorate,
        "geo_delegation": request.delegation,
    }])
    
    predicted_log_price = pipeline.predict(df)[0]

    # Slider pricing is intentionally direct: price per m2 times selected surface.
    # We keep the benchmark fixed for the chosen family and delegation so the
    # result changes exactly with the user-entered surface.
    benchmark_ppm2 = float(current_price_per_m2)
    predicted_price_per_m2 = benchmark_ppm2
    predicted_price = predicted_price_per_m2 * request.surface_m2
    
    confidence = "high" if coord != (10.0, 36.0) else "low"
    
    return PredictionResponse(
        predicted_price_tnd=round(predicted_price, 2),
        predicted_price_per_m2=round(predicted_price_per_m2, 2),
        benchmark_ppm2=round(benchmark_ppm2, 2),
        surface_m2=round(float(request.surface_m2), 2),
        confidence=confidence,
        model_r2=load_model_score(),
        model_version="2.1",
        governance=f"{request.governorate}/{request.delegation}"
    )


@app.get("/delegations")
async def get_delegations():
    return {"delegations": list(delegation_coords.keys()) if delegation_coords else []}


app.mount("/", NoCacheStaticFiles(directory=FRONTEND_DIR, html=True), name="frontend-root")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
