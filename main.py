from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "crop_stress_model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run the notebook save-model cell first.")
    bundle = joblib.load(MODEL_PATH)
    app.state.classifier = bundle["classifier"]
    app.state.regressor = bundle["regressor"]
    app.state.features = bundle["features"]
    app.state.levels = bundle.get("levels", ["Low", "Moderate", "Severe"])
    yield

app = FastAPI(title="Crop Stress Risk Prediction API", version="1.0.0", lifespan=lifespan,
               description="Machine-learning classification of crop stress risk and regression of stress index.")

class Features(BaseModel):
    crop_type: str = Field(..., examples=["Rice"])
    growth_stage: str = Field(..., examples=["Mid-season"])
    soil_type: str = Field(..., examples=["Clay Loam"])
    soil_moisture_pct: float = Field(..., ge=0, le=100)
    air_temperature_c: float
    relative_humidity_pct: float = Field(..., ge=0, le=100)
    rainfall_7d_mm: float = Field(..., ge=0)
    solar_radiation_mj_m2: float = Field(..., ge=0)
    wind_speed_ms: float = Field(..., ge=0)
    et0_ref_evapotranspiration_mm: float = Field(..., ge=0)
    etc_crop_evapotranspiration_mm: float = Field(..., ge=0)
    eta_actual_evapotranspiration_mm: float = Field(..., ge=0)
    evaporative_stress_index: float
    ndvi: float = Field(..., ge=-1, le=1)
    vapor_pressure_deficit_kpa: float = Field(..., ge=0)
    historical_stress_days_30d: int = Field(..., ge=0, le=30)

class PredictRequest(BaseModel):
    features: Features

@app.get("/")
def root():
    return {"message": "Crop Stress Risk Prediction API", "docs": "/docs", "endpoint": "/predict"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": hasattr(app.state, "classifier")}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        payload = request.features.model_dump()
        row = pd.DataFrame([payload], columns=app.state.features)
        classifier = app.state.classifier
        regressor = app.state.regressor
        stress_level = str(classifier.predict(row)[0])
        stress_index = float(np.clip(regressor.predict(row)[0], 0, 100))
        probs = classifier.predict_proba(row)[0]
        labels = classifier.classes_
        probability_map = {str(label): round(float(prob), 4) for label, prob in zip(labels, probs)}
        # A simple field priority interpretation based on model prediction only.
        if stress_level == "Severe" or stress_index >= 65:
            priority = "High - prioritize irrigation assessment"
        elif stress_level == "Moderate" or stress_index >= 35:
            priority = "Medium - monitor and schedule irrigation"
        else:
            priority = "Low - continue monitoring"
        return {
            "predicted_stress_level": stress_level,
            "predicted_stress_index_0_to_100": round(stress_index, 2),
            "class_probabilities": probability_map,
            "irrigation_priority_recommendation": priority,
            "note": "Prediction is model-based. Confirm with field observations before irrigation."
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
