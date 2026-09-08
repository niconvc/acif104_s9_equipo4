"""
app.py
------
Backend FastAPI del clasificador de segmento de precio de vehículos.

Endpoints:

    GET  /         -> sirve la interfaz web.
    POST /predict  -> predice segmento y probabilidades estimadas.
    GET  /metrics  -> entrega métricas básicas de monitoreo.
    GET  /health   -> informa el estado del servicio.

La explicación entregada por la aplicación corresponde a una atribución
SHAP LOCAL calculada sobre el mismo Random Forest final que produce la
predicción (no un modelo sustituto). Cada respuesta de /predict incluye
las variables que más empujaron esa predicción específica, en la
dirección observada para la clase predicha.

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
import shap

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
    version="1.2.0",
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
# Explicador SHAP (construido una sola vez sobre el clasificador
# final, para no reconstruir el árbol de decisiones en cada
# solicitud de predicción).
# ---------------------------------------------------------

_PREPROCESSOR = model.named_steps["preprocessor"]
_CLASSIFIER = model.named_steps["classifier"]
_FEATURE_NAMES = (
    _PREPROCESSOR.named_transformers_["num"].feature_names_in_.tolist()
    + _PREPROCESSOR.named_transformers_["cat"].get_feature_names_out().tolist()
)
_EXPLAINER = shap.TreeExplainer(_CLASSIFIER)


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
# Explicación local SHAP
# ---------------------------------------------------------

# Nombres legibles para las variables de entrada más frecuentes en las
# explicaciones locales (el resto se muestra con su nombre técnico).
_READABLE_NAMES = {
    "model_year": "Año del modelo",
    "odometer": "Kilometraje",
    "odometer_per_year": "Kilometraje por año",
    "is_4wd": "Tracción 4×4",
    "cylinders": "Cantidad de cilindros",
    "condition_ord": "Condición del vehículo",
    "days_listed": "Días de publicación",
}


def build_shap_explanation(
    features: pd.DataFrame,
    predicted_segment: str,
    top_n: int = 5,
) -> list[dict]:
    """
    Genera una explicación LOCAL para la predicción actual, calculada
    directamente sobre el Random Forest final (mismo objeto que
    produce la predicción, no un modelo sustituto).

    Devuelve las variables con mayor contribución SHAP para la clase
    predicha, indicando si empujaron la predicción hacia esa clase
    (contribución positiva) o en contra (contribución negativa).
    """

    transformed = _PREPROCESSOR.transform(features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    class_index = list(_CLASSIFIER.classes_).index(predicted_segment)

    shap_values = _EXPLAINER.shap_values(
        transformed,
        check_additivity=False,
    )

    values_for_class = (
        shap_values[:, :, class_index]
        if not isinstance(shap_values, list)
        else shap_values[class_index]
    )

    contributions = pd.Series(
        values_for_class[0],
        index=_FEATURE_NAMES,
    )

    top = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    ).head(top_n)

    explanation = []

    for feature_name, shap_value in top.items():
        direction = "a favor" if shap_value > 0 else "en contra"
        explanation.append({
            "variable": _READABLE_NAMES.get(feature_name, feature_name),
            "shap_value": round(float(shap_value), 4),
            "direccion": direction,
        })

    return explanation


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

    explanation = build_shap_explanation(
        features,
        predicted_segment,
    )

    return {
        "segment": predicted_segment,
        "probabilities": probability_by_class,
        "probabilities_calibrated": False,
        "explanation": explanation,
        "explanation_type": "shap_local",
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
        "explanation_type": "shap_local",
    }