"""
app.py — Backend FastAPI del Clasificador de Segmento de Precio de Vehículos
----------------------------------------------------------------------------
Expone:
  GET  /            -> sirve el frontend (index.html)
  POST /predict     -> predice el segmento + probabilidades + explicación
  GET  /metrics     -> métricas de monitoreo del servicio (uso, latencia, mezcla)
  GET  /health      -> estado del servicio

Ejecutar:  uvicorn app:app --reload  (desde la carpeta backend/)
"""
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# permitir importar el módulo de preprocesamiento
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import CONDITION_ORDER, SEG_LABELS  # noqa: E402

MODEL_PATH = ROOT / "models" / "final_model.joblib"
REF_YEAR = 2019  # año del aviso más reciente del dataset

app = FastAPI(title="Clasificador de Segmento de Precio de Vehículos",
              version="1.0")

model = joblib.load(MODEL_PATH)

# --- Estado de monitoreo (en memoria) ---
MONITOR = {
    "total": 0,
    "by_class": Counter(),
    "latencies_ms": deque(maxlen=500),
    "recent": deque(maxlen=20),
}


class Vehicle(BaseModel):
    model_year: int = Field(..., ge=1980, le=REF_YEAR, example=2015)
    odometer: float = Field(..., ge=0, le=400000, example=90000)
    cylinders: int = Field(..., ge=3, le=12, example=6)
    condition: str = Field(..., example="good")
    fuel: str = Field(..., example="gas")
    transmission: str = Field(..., example="automatic")
    type: str = Field(..., example="SUV")
    paint_color: str = Field("unknown", example="white")
    brand: str = Field(..., example="toyota")
    is_4wd: int = Field(0, ge=0, le=1, example=1)
    days_listed: int = Field(30, ge=0, le=300, example=30)


def to_features(v: Vehicle) -> pd.DataFrame:
    """Construye el vector de features que espera el pipeline."""
    age = max(REF_YEAR - v.model_year, 0)
    row = {
        "model_year": v.model_year,
        "age": age,
        "odometer": v.odometer,
        "odometer_per_year": v.odometer / (age if age else 1),
        "cylinders": v.cylinders,
        "condition_ord": CONDITION_ORDER.get(v.condition, 2),
        "is_4wd": v.is_4wd,
        "days_listed": v.days_listed,
        "brand": v.brand.lower(),
        "fuel": v.fuel,
        "transmission": v.transmission,
        "type": v.type,
        "paint_color": v.paint_color,
    }
    return pd.DataFrame([row])


def explain(v: Vehicle, pred: str) -> list[str]:
    """Explicación heurística por instancia, fundada en el análisis SHAP
    (año, odómetro, 4x4, cilindros y carrocería son los drivers clave)."""
    age = REF_YEAR - v.model_year
    notes = []
    if age <= 4:
        notes.append("Año reciente (poca antigüedad): empuja hacia Premium.")
    elif age >= 12:
        notes.append("Vehículo antiguo: empuja hacia Económico.")
    if v.odometer <= 60000:
        notes.append("Kilometraje bajo: sube el valor.")
    elif v.odometer >= 160000:
        notes.append("Kilometraje alto: baja el valor.")
    if v.is_4wd:
        notes.append("Tracción 4×4: factor que incrementa el precio.")
    if v.cylinders >= 8:
        notes.append("Motor de muchos cilindros: asociado a segmentos altos.")
    elif v.cylinders <= 4:
        notes.append("Motor de pocos cilindros: asociado a segmentos económicos.")
    if v.type in ("truck", "pickup"):
        notes.append("Carrocería truck/pickup: tiende a Premium en este mercado.")
    if not notes:
        notes.append("Combinación de atributos coherente con el segmento estimado.")
    return notes


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent.parent / "frontend" / "index.html").read_text(
        encoding="utf-8")


@app.post("/predict")
def predict(v: Vehicle):
    t0 = time.perf_counter()
    X = to_features(v)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    pred = classes[int(np.argmax(proba))]
    dt = (time.perf_counter() - t0) * 1000

    MONITOR["total"] += 1
    MONITOR["by_class"][pred] += 1
    MONITOR["latencies_ms"].append(dt)
    MONITOR["recent"].appendleft({"pred": pred, "brand": v.brand,
                                  "year": v.model_year})

    return {
        "segment": pred,
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "explanation": explain(v, pred),
        "latency_ms": round(dt, 2),
    }


@app.get("/metrics")
def metrics():
    lat = list(MONITOR["latencies_ms"])
    return JSONResponse({
        "total_predictions": MONITOR["total"],
        "class_distribution": dict(MONITOR["by_class"]),
        "avg_latency_ms": round(float(np.mean(lat)), 2) if lat else None,
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
        "recent": list(MONITOR["recent"]),
    })


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "classes": SEG_LABELS}
