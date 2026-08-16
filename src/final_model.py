"""
final_model.py
--------------
Entrena el modelo final seleccionado, Random Forest combinado con SMOTE,
utilizando los conjuntos de entrenamiento y validación.

Posteriormente, evalúa el modelo sobre el conjunto de test, que no fue
utilizado durante el entrenamiento ni la selección de hiperparámetros.

Archivos generados:

    models/final_model.joblib
    reports/final_model_results.csv
    reports/confusion_matrix.csv
    reports/figures/confusion_matrix.png

El modelo se guarda comprimido para reducir su tamaño en disco.
"""

from pathlib import Path

import joblib
import matplotlib

# Permite generar la figura sin abrir una ventana gráfica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocessing import SEG_LABELS
from train_ml import (
    build_preprocessor,
    get_splits,
)


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vehicles_us.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "final_model.joblib"
RESULTS_PATH = REPORTS_DIR / "final_model_results.csv"
CONFUSION_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
CONFUSION_FIGURE_PATH = (
    FIGURES_DIR / "confusion_matrix.png"
)


# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------

RANDOM_STATE = 42
N_ESTIMATORS = 200
N_JOBS = -1


# ---------------------------------------------------------
# Construcción del modelo
# ---------------------------------------------------------

def build_final_pipeline():
    """
    Construye el pipeline final.

    Etapas:

        1. Preprocesamiento numérico y categórico.
        2. Balanceo de clases mediante SMOTE.
        3. Clasificación mediante Random Forest.
    """

    return ImbPipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=None,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------

def build_results_table(
    y_true,
    predictions,
):
    """
    Construye la tabla de precisión, recall, F1-score y soporte.

    La fila Global (macro) no presenta soporte, porque corresponde
    al promedio no ponderado de las métricas de las tres clases.
    """

    report = classification_report(
        y_true,
        predictions,
        labels=SEG_LABELS,
        output_dict=True,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    rows = []

    for segment in SEG_LABELS:
        segment_result = report[
            segment
        ]

        rows.append({
            "segmento": segment,
            "precision": round(
                segment_result["precision"],
                3,
            ),
            "recall": round(
                segment_result["recall"],
                3,
            ),
            "f1_score": round(
                segment_result["f1-score"],
                3,
            ),
            "soporte": int(
                segment_result["support"]
            ),
            "accuracy_global": "",
        })

    macro_result = report[
        "macro avg"
    ]

    rows.append({
        "segmento": "Global (macro)",
        "precision": round(
            macro_result["precision"],
            3,
        ),
        "recall": round(
            macro_result["recall"],
            3,
        ),
        "f1_score": round(
            macro_result["f1-score"],
            3,
        ),
        "soporte": "",
        "accuracy_global": round(
            accuracy,
            3,
        ),
    })

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# Matriz de confusión
# ---------------------------------------------------------

def build_confusion_table(
    y_true,
    predictions,
):
    """
    Construye la matriz de confusión con el orden oficial
    de los segmentos.
    """

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=SEG_LABELS,
    )

    return pd.DataFrame(
        matrix,
        index=SEG_LABELS,
        columns=SEG_LABELS,
    )


def save_confusion_figure(
    confusion_table,
):
    """
    Genera y guarda la figura de la matriz de confusión.
    """

    sns.set_theme(
        style="white",
        context="notebook",
        font_scale=1.05,
    )

    figure, axis = plt.subplots(
        figsize=(7.5, 6.0),
    )

    sns.heatmap(
        confusion_table,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.8,
        linecolor="white",
        square=True,
        ax=axis,
        annot_kws={
            "size": 12,
        },
        cbar_kws={
            "label": "Cantidad de vehículos",
            "shrink": 0.85,
        },
    )

    axis.set_title(
        "Matriz de confusión del modelo final",
        fontsize=14,
        pad=14,
    )

    axis.set_xlabel(
        "Segmento predicho",
        fontsize=12,
        labelpad=10,
    )

    axis.set_ylabel(
        "Segmento real",
        fontsize=12,
        labelpad=10,
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.tick_params(
        axis="y",
        rotation=0,
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------

def save_results(
    results_table,
    confusion_table,
):
    """
    Guarda las métricas, matriz de confusión y figura.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_table.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_table.to_csv(
        CONFUSION_CSV_PATH,
        index=True,
        index_label="segmento_real",
        encoding="utf-8-sig",
    )

    save_confusion_figure(
        confusion_table
    )


def save_model(
    pipeline,
):
    """
    Guarda el pipeline completo con compresión.

    La compresión reduce el tamaño del archivo sin modificar
    predicciones ni métricas.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
        compress=3,
    )


# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------

def format_file_size_mb(
    path,
):
    """Devuelve el tamaño de un archivo en megabytes."""

    size_bytes = path.stat().st_size

    return size_bytes / (
        1024 ** 2
    )


def validate_outputs():
    """
    Comprueba que todos los archivos esperados hayan sido creados.
    """

    expected_files = [
        MODEL_PATH,
        RESULTS_PATH,
        CONFUSION_CSV_PATH,
        CONFUSION_FIGURE_PATH,
    ]

    missing_files = [
        path
        for path in expected_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            str(path)
            for path in missing_files
        )

        raise RuntimeError(
            "No se generaron los siguientes archivos:\n"
            f"{missing_text}"
        )


# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------

def main():
    """
    Entrena, evalúa y guarda el modelo final y sus evidencias.
    """

    print("=" * 65)
    print("ENTRENAMIENTO DEL MODELO FINAL")
    print("=" * 65)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = get_splits(
        path=str(DATA_PATH)
    )

    # El conjunto de test se mantiene completamente separado.
    X_full = pd.concat(
        [
            X_train,
            X_validation,
        ],
        ignore_index=True,
    )

    y_full = pd.concat(
        [
            y_train,
            y_validation,
        ],
        ignore_index=True,
    )

    print(
        f"Entrenamiento + validación: "
        f"{len(X_full):,}"
    )

    print(
        f"Test:                      "
        f"{len(X_test):,}"
    )

    pipeline = build_final_pipeline()

    print()
    print("Entrenando Random Forest + SMOTE...")

    pipeline.fit(
        X_full,
        y_full,
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    results_table = build_results_table(
        y_test,
        predictions,
    )

    confusion_table = build_confusion_table(
        y_test,
        predictions,
    )

    print()
    print("=" * 65)
    print("EVALUACIÓN FINAL SOBRE TEST")
    print("=" * 65)

    print(
        f"Accuracy:  {accuracy:.3f}"
    )

    print(
        f"F1-macro:  {f1_macro:.3f}"
    )

    print()
    print("Resultados por segmento:")
    print(
        results_table.to_string(
            index=False
        )
    )

    print()
    print(
        "Matriz de confusión "
        "(filas=real, columnas=predicho):"
    )

    print(
        confusion_table.to_string()
    )

    save_results(
        results_table,
        confusion_table,
    )

    save_model(
        pipeline
    )

    validate_outputs()

    model_size_mb = format_file_size_mb(
        MODEL_PATH
    )

    print()
    print("=" * 65)
    print("ARCHIVOS GENERADOS")
    print("=" * 65)

    print(
        f"Modelo:\n{MODEL_PATH}"
    )

    print(
        f"\nTamaño comprimido: "
        f"{model_size_mb:.1f} MB"
    )

    print(
        f"\nResultados:\n{RESULTS_PATH}"
    )

    print(
        f"\nMatriz CSV:\n{CONFUSION_CSV_PATH}"
    )

    print(
        f"\nFigura:\n{CONFUSION_FIGURE_PATH}"
    )

    return (
        pipeline,
        results_table,
        confusion_table,
    )


if __name__ == "__main__":
    main()