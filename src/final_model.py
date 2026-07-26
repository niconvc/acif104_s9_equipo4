"""
final_model.py
--------------
Entrena el modelo final seleccionado (Random Forest + SMOTE) sobre
train+val, lo evalúa sobre el conjunto de TEST (datos nunca vistos) y
lo persiste para el backend.
"""
import os

import numpy as np
import pandas as pd
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, SEG_LABELS
from train_ml import get_splits, build_preprocessor

RS = 42


def main():
    Xtr, Xval, Xte, ytr, yval, yte = get_splits()
    # entrenar con train+val, testear en test
    Xfull = pd.concat([Xtr, Xval]); yfull = pd.concat([ytr, yval])

    pipe = ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RS)),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                       random_state=RS)),
    ])
    pipe.fit(Xfull, yfull)

    pred = pipe.predict(Xte)
    print("=" * 60)
    print("EVALUACIÓN FINAL SOBRE TEST (datos nunca vistos):")
    print(f"  Accuracy : {accuracy_score(yte, pred):.3f}")
    print(f"  F1-macro : {f1_score(yte, pred, average='macro'):.3f}")
    print("\n", classification_report(yte, pred, digits=3))
    print("Matriz de confusión (filas=real, cols=pred):")
    print(pd.DataFrame(confusion_matrix(yte, pred, labels=SEG_LABELS),
                       index=SEG_LABELS, columns=SEG_LABELS).to_string())

    # joblib no crea el directorio destino: sin esto, un clon sin models/
    # entrena varios minutos y falla recien al guardar
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/final_model.joblib")
    print("\nModelo guardado en models/final_model.joblib")


if __name__ == "__main__":
    main()
