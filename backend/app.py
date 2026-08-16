"""
app.py
------
Backend FastAPI del clasificador de segmento de precio de vehículos.

Endpoints:

    GET  /         -> sirve la interfaz web.
    POST /predict  -> predice segmento y probabilidades estimadas.
    GET  /metrics  -> entrega métricas básicas de monitoreo.
    GET  /health   -> informa el estado del servicio.

La explicación entregada por la aplicación es heurística. Utiliza reglas
descriptivas construidas a partir de variables relevantes identificadas
durante el EDA y el análisis SHAP global. No corresponde a una atribución
SHAP local de cada predicción.

Ejecución desde la raíz del proyecto:

    python -m uvicorn backend.app:app --reload
"""

import sys
import time

from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MODEL_PATH = ROOT / "models" / "final_model.joblib"
FRONTEND_PATH = ROOT / "frontend" / "index.html"

sys.path.insert(
    0,
    str(SRC_DIR),
)

from preprocessing import (  # noqa: E402
    CONDITION_ORDER,
    SEG_LABELS,
)


# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------

REF_YEAR = 2019

app = FastAPI(
    title="Clasificador de Segmento de Precio de Vehículos",
    description=(
        "API para clasificar vehículos en los segmentos "
        "Económico, Medio o Premium."
    ),
    version="1.1.0",
)


# ---------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "No se encontró el modelo final:\n"
        f"{MODEL_PATH}\n\n"
        "Ejecuta primero:\n"
        "python src/final_model.py"
    )

model = joblib.load(
    MODEL_PATH
)


# ---------------------------------------------------------
# Estado de monitoreo
# ---------------------------------------------------------

MONITOR = {
    "total": 0,
    "by_class": Counter(),
    "latencies_ms": deque(
        maxlen=500
    ),
    "recent": deque(
        maxlen=20
    ),
}


# ---------------------------------------------------------
# Modelo de entrada
# ---------------------------------------------------------

class Vehicle(BaseModel):
    """Datos requeridos para clasificar un vehículo."""

    model_year: int = Field(
        ...,
        ge=1980,
        le=REF_YEAR,
        examples=[2015],
    )

    odometer: float = Field(
        ...,
        ge=0,
        le=400000,
        examples=[90000],
    )

    cylinders: int = Field(
        ...,
        ge=3,
        le=12,
        examples=[6],
    )

    condition: str = Field(
        ...,
        examples=["good"],
    )

    fuel: str = Field(
        ...,
        examples=["gas"],
    )

    transmission: str = Field(
        ...,
        examples=["automatic"],
    )

    type: str = Field(
        ...,
        examples=["SUV"],
    )

    paint_color: str = Field(
        default="unknown",
        examples=["white"],
    )

    brand: str = Field(
        ...,
        examples=["toyota"],
    )

    is_4wd: int = Field(
        default=0,
        ge=0,
        le=1,
        examples=[1],
    )

    days_listed: int = Field(
        default=30,
        ge=0,
        le=300,
        examples=[30],
    )


# ---------------------------------------------------------
# Construcción de características
# ---------------------------------------------------------

def to_features(vehicle: Vehicle) -> pd.DataFrame:
    """
    Construye las características que espera el pipeline.

    La ingeniería de variables debe coincidir con la utilizada durante
    el entrenamiento del modelo.
    """

    age = max(
        REF_YEAR - vehicle.model_year,
        0,
    )

    odometer_per_year = (
        vehicle.odometer
        / (age if age else 1)
    )

    row = {
        "model_year": vehicle.model_year,
        "age": age,
        "odometer": vehicle.odometer,
        "odometer_per_year": odometer_per_year,
        "cylinders": vehicle.cylinders,
        "condition_ord": CONDITION_ORDER.get(
            vehicle.condition,
            2,
        ),
        "is_4wd": vehicle.is_4wd,
        "days_listed": vehicle.days_listed,
        "brand": vehicle.brand.lower(),
        "fuel": vehicle.fuel,
        "transmission": vehicle.transmission,
        "type": vehicle.type,
        "paint_color": vehicle.paint_color,
    }

    return pd.DataFrame([
        row
    ])


# ---------------------------------------------------------
# Orientación heurística
# ---------------------------------------------------------

def build_heuristic_explanation(
    vehicle: Vehicle,
) -> list[str]:
    """
    Genera una orientación descriptiva sobre los atributos ingresados.

    Las reglas se basan en variables relevantes observadas durante el
    análisis exploratorio y el análisis SHAP global. No corresponden a
    valores SHAP locales ni explican matemáticamente la predicción
    individual del clasificador.
    """

    age = REF_YEAR - vehicle.model_year
    notes = []

    if age <= 4:
        notes.append(
            "Año reciente: atributo generalmente asociado "
            "con segmentos de mayor precio."
        )
    elif age >= 12:
        notes.append(
            "Mayor antigüedad: atributo generalmente asociado "
            "con segmentos de menor precio."
        )

    if vehicle.odometer <= 60000:
        notes.append(
            "Kilometraje bajo: factor habitualmente asociado "
            "con un mayor valor."
        )
    elif vehicle.odometer >= 160000:
        notes.append(
            "Kilometraje alto: factor habitualmente asociado "
            "con un menor valor."
        )

    if vehicle.is_4wd:
        notes.append(
            "Tracción 4×4: característica asociada con un "
            "mayor valor en el conjunto analizado."
        )

    if vehicle.cylinders >= 8:
        notes.append(
            "Motor de ocho o más cilindros: característica "
            "frecuente en segmentos de mayor precio."
        )
    elif vehicle.cylinders <= 4:
        notes.append(
            "Motor de cuatro o menos cilindros: característica "
            "frecuente en segmentos económicos."
        )

    if vehicle.type in (
        "truck",
        "pickup",
    ):
        notes.append(
            "Carrocería truck/pickup: categoría asociada con "
            "segmentos de mayor precio en este mercado."
        )

    if not notes:
        notes.append(
            "La combinación ingresada no activa ninguna de las "
            "reglas descriptivas principales."
        )

    return notes


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    """Entrega la interfaz web del clasificador."""

    if not FRONTEND_PATH.exists():
        return HTMLResponse(
            content=(
                "<h1>Frontend no disponible</h1>"
                "<p>No se encontró frontend/index.html.</p>"
            ),
            status_code=404,
        )

    return FRONTEND_PATH.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------
# Predicción
# ---------------------------------------------------------

@app.post("/predict")
def predict(vehicle: Vehicle):
    """
    Clasifica un vehículo y entrega probabilidades estimadas.

    Las probabilidades no han sido sometidas a un procedimiento
    específico de calibración, por lo que no deben interpretarse
    automáticamente como niveles de confianza calibrados.
    """

    start_time = time.perf_counter()

    features = to_features(
        vehicle
    )

    probabilities = model.predict_proba(
        features
    )[0]

    classes = list(
        model.classes_
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_segment = classes[
        predicted_index
    ]

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    MONITOR["total"] += 1
    MONITOR["by_class"][
        predicted_segment
    ] += 1

    MONITOR["latencies_ms"].append(
        latency_ms
    )

    MONITOR["recent"].appendleft({
        "segment": predicted_segment,
        "brand": vehicle.brand,
        "model_year": vehicle.model_year,
    })

    probability_by_class = {
        class_name: round(
            float(probability),
            4,
        )
        for class_name, probability in zip(
            classes,
            probabilities,
        )
    }

    return {
        "segment": predicted_segment,
        "probabilities": probability_by_class,
        "probabilities_calibrated": False,
        "explanation": build_heuristic_explanation(
            vehicle
        ),
        "explanation_type": "heuristic",
        "latency_ms": round(
            latency_ms,
            2,
        ),
    }


# ---------------------------------------------------------
# Monitoreo
# ---------------------------------------------------------

@app.get("/metrics")
def metrics():
    """Entrega métricas básicas de uso y latencia."""

    latencies = list(
        MONITOR["latencies_ms"]
    )

    average_latency = (
        round(
            float(np.mean(latencies)),
            2,
        )
        if latencies
        else None
    )

    p95_latency = (
        round(
            float(
                np.percentile(
                    latencies,
                    95,
                )
            ),
            2,
        )
        if latencies
        else None
    )

    return JSONResponse({
        "total_predictions": MONITOR["total"],
        "class_distribution": dict(
            MONITOR["by_class"]
        ),
        "avg_latency_ms": average_latency,
        "p95_latency_ms": p95_latency,
        "recent": list(
            MONITOR["recent"]
        ),
    })


# ---------------------------------------------------------
# Estado del servicio
# ---------------------------------------------------------

@app.get("/health")
def health():
    """Informa el estado de la API y del modelo."""

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": SEG_LABELS,
        "probabilities_calibrated": False,
        "explanation_type": "heuristic",
    }