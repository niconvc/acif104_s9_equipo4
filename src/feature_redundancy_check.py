"""
feature_redundancy_check.py
----------------------------
Verifica el efecto de eliminar la variable redundante `age`
(age = 2019 - model_year) sobre el desempeño del modelo final
(Random Forest + SMOTE), en respuesta a la retroalimentación docente
sobre la Sumativa 2.

Se reentrena el mismo pipeline final utilizando únicamente
`model_year` (sin `age`) y se compara contra el modelo original
sobre el mismo conjunto de test.

Salida: reports/feature_redundancy_results.csv
"""
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train_ml import get_splits

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "feature_redundancy_results.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 200

# Features sin 'age' (se conserva model_year, la variable original)
NUMERIC_NO_AGE = [f for f in NUMERIC_FEATURES if f != "age"]


def build_pipeline(numeric_features):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=None, n_jobs=-1,
            random_state=RANDOM_STATE)),
    ])


def evaluate(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    auc = roc_auc_score(y_test, probabilities, labels=pipeline.classes_,
                         multi_class="ovr", average="macro")
    return accuracy, f1_macro, auc


def main():
    X_tr, X_val, X_te, y_tr, y_val, y_te = get_splits()
    X_full = pd.concat([X_tr, X_val], ignore_index=True)
    y_full = pd.concat([y_tr, y_val], ignore_index=True)

    print("Entrenando modelo CON 'age' y 'model_year' (original)...")
    pipe_with_age = build_pipeline(NUMERIC_FEATURES)
    pipe_with_age.fit(X_full, y_full)
    acc_with, f1_with, auc_with = evaluate(pipe_with_age, X_te, y_te)

    print("Entrenando modelo SIN 'age' (solo model_year)...")
    pipe_no_age = build_pipeline(NUMERIC_NO_AGE)
    pipe_no_age.fit(X_full, y_full)
    acc_without, f1_without, auc_without = evaluate(pipe_no_age, X_te, y_te)

    tbl = pd.DataFrame([
        {"variante": "Con age + model_year (original)",
         "accuracy": round(acc_with, 3), "f1_macro": round(f1_with, 3),
         "auc_ovr_macro": round(auc_with, 3)},
        {"variante": "Sin age (solo model_year)",
         "accuracy": round(acc_without, 3), "f1_macro": round(f1_without, 3),
         "auc_ovr_macro": round(auc_without, 3)},
    ])

    print("\n" + "=" * 60)
    print("EFECTO DE ELIMINAR LA VARIABLE REDUNDANTE 'age'")
    print("=" * 60)
    print(tbl.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"\nGuardado en: {RESULTS_PATH}")
    return tbl, pipe_no_age


if __name__ == "__main__":
    main()
