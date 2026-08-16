"""
rf_refinement.py
----------------
Realiza un barrido controlado de hiperparámetros sobre Random Forest.

Se evalúan:

    n_estimators: 100, 200 y 300 árboles.
    max_depth: 20 y sin límite.

Los modelos se entrenan únicamente con el conjunto de entrenamiento
y se evalúan sobre el conjunto de validación.

Métricas registradas:

    - F1-macro.
    - Recall de la clase Premium.
    - Recall de la clase Económico.
    - Tiempo de entrenamiento.
    - Tamaño serializado del pipeline.

Salida:

    reports/rf_refinement_results.csv

El script reutiliza las funciones get_splits() y build_preprocessor()
de train_ml.py para garantizar que el barrido utilice exactamente
la misma partición y el mismo preprocesamiento del proyecto.

Los modelos se serializan temporalmente para medir su tamaño y luego
los archivos temporales son eliminados.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import SEG_LABELS
from train_ml import (
    build_preprocessor,
    get_splits,
)


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "rf_refinement_results.csv"


# ---------------------------------------------------------
# Configuración experimental
# ---------------------------------------------------------

RANDOM_STATE = 42
N_JOBS = -1

PARAMETER_GRID = [
    {
        "n_estimators": 100,
        "max_depth": 20,
    },
    {
        "n_estimators": 100,
        "max_depth": None,
    },
    {
        "n_estimators": 200,
        "max_depth": 20,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
    },
    {
        "n_estimators": 300,
        "max_depth": None,
    },
]


# ---------------------------------------------------------
# Construcción del pipeline
# ---------------------------------------------------------

def build_pipeline(
    n_estimators,
    max_depth,
):
    """
    Construye un pipeline compuesto por:

        1. Preprocesamiento definido en train_ml.py.
        2. Random Forest con balanceo mediante class_weight.

    Parameters
    ----------
    n_estimators : int
        Cantidad de árboles del Random Forest.

    max_depth : int o None
        Profundidad máxima de los árboles. None significa que
        los árboles pueden crecer sin un límite explícito.
    """

    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                random_forest,
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# Medición del tamaño del modelo
# ---------------------------------------------------------

def measure_serialized_size_mb(model):
    """
    Serializa temporalmente el pipeline para medir su tamaño.

    El archivo temporal se elimina siempre, incluso si ocurre un error
    durante la medición.

    Parameters
    ----------
    model
        Pipeline entrenado que será serializado.

    Returns
    -------
    float
        Tamaño del archivo serializado en megabytes.
    """

    temporary_path = None

    try:
        with NamedTemporaryFile(
            suffix=".joblib",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        joblib.dump(
            model,
            temporary_path,
            compress=3,
        )

        size_bytes = temporary_path.stat().st_size
        size_mb = size_bytes / (1024 ** 2)

        return round(
            size_mb,
            1,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


# ---------------------------------------------------------
# Evaluación de una configuración
# ---------------------------------------------------------

def evaluate_configuration(
    X_train,
    X_validation,
    y_train,
    y_validation,
    n_estimators,
    max_depth,
):
    """
    Entrena y evalúa una configuración de Random Forest.

    La evaluación se realiza exclusivamente sobre el conjunto de
    validación. El conjunto de prueba permanece sin utilizar.
    """

    depth_label = (
        "Sin límite"
        if max_depth is None
        else str(max_depth)
    )

    print()
    print("=" * 65)
    print(
        f"Entrenando Random Forest: "
        f"{n_estimators} árboles, "
        f"profundidad {depth_label}"
    )
    print("=" * 65)

    pipeline = build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    start_time = perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_time = (
        perf_counter()
        - start_time
    )

    predictions = pipeline.predict(
        X_validation
    )

    f1_macro = f1_score(
        y_validation,
        predictions,
        average="macro",
    )

    recalls = recall_score(
        y_validation,
        predictions,
        labels=SEG_LABELS,
        average=None,
        zero_division=0,
    )

    recall_by_segment = dict(
        zip(
            SEG_LABELS,
            recalls,
        )
    )

    model_size_mb = measure_serialized_size_mb(
        pipeline
    )

    result = {
        "n_estimators": n_estimators,
        "max_depth": depth_label,
        "f1_macro": round(
            f1_macro,
            3,
        ),
        "recall_premium": round(
            recall_by_segment["Premium"],
            3,
        ),
        "recall_economico": round(
            recall_by_segment["Económico"],
            3,
        ),
        "training_time_seconds": round(
            training_time,
            1,
        ),
        "model_size_mb": model_size_mb,
    }

    print(
        f"F1-macro:        "
        f"{result['f1_macro']:.3f}"
    )

    print(
        f"Recall Premium:  "
        f"{result['recall_premium']:.3f}"
    )

    print(
        f"Recall Económico:"
        f" {result['recall_economico']:.3f}"
    )

    print(
        f"Tiempo:          "
        f"{result['training_time_seconds']:.1f} s"
    )

    print(
        f"Tamaño:          "
        f"{result['model_size_mb']:.1f} MB"
    )

    return result


# ---------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------

def save_results(results):
    """
    Convierte los resultados a DataFrame y los guarda en CSV.
    """

    results_table = pd.DataFrame(results)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_table.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "No fue posible crear el archivo de resultados:\n"
            f"{OUTPUT_PATH}"
        )

    return results_table


# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------

def main():
    """
    Ejecuta el barrido completo y guarda los resultados.
    """

    print("=" * 65)
    print("BARRIDO DE HIPERPARÁMETROS DE RANDOM FOREST")
    print("=" * 65)

    # Se utilizan las mismas particiones estratificadas 70/15/15
    # definidas en train_ml.py.
    (
        X_train,
        X_validation,
        _,
        y_train,
        y_validation,
        _,
    ) = get_splits()

    print(
        f"Registros de entrenamiento: "
        f"{len(X_train):,}"
    )

    print(
        f"Registros de validación:    "
        f"{len(X_validation):,}"
    )

    results = []

    for configuration in PARAMETER_GRID:
        result = evaluate_configuration(
            X_train=X_train,
            X_validation=X_validation,
            y_train=y_train,
            y_validation=y_validation,
            n_estimators=configuration[
                "n_estimators"
            ],
            max_depth=configuration[
                "max_depth"
            ],
        )

        results.append(result)

    results_table = save_results(
        results
    )

    print()
    print("=" * 65)
    print("RESULTADOS DEL BARRIDO")
    print("=" * 65)

    print(
        results_table.to_string(
            index=False
        )
    )

    print()
    print("Resultados guardados correctamente en:")
    print(OUTPUT_PATH)

    best_index = results_table[
        "f1_macro"
    ].idxmax()

    best_result = results_table.loc[
        best_index
    ]

    print()
    print("Mejor configuración según F1-macro:")

    print(
        f"  Árboles: "
        f"{int(best_result['n_estimators'])}"
    )

    print(
        f"  Profundidad: "
        f"{best_result['max_depth']}"
    )

    print(
        f"  F1-macro: "
        f"{best_result['f1_macro']:.3f}"
    )

    print(
        f"  Recall Premium: "
        f"{best_result['recall_premium']:.3f}"
    )

    print(
        f"  Recall Económico: "
        f"{best_result['recall_economico']:.3f}"
    )

    return results_table


if __name__ == "__main__":
    main()