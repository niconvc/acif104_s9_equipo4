"""
balancing.py
------------
Analiza el efecto de 3 técnicas de balanceo de clases sobre el
rendimiento del mejor modelo (Random Forest), midiendo especialmente
el recall de la clase minoritaria (Premium).

Técnicas:
  1) Class weights   (balanced)  - penaliza errores en clases pequeñas.
  2) SMOTE           (oversampling sintético de minoritarias).
  3) RandomUnderSampler (submuestreo de la mayoritaria).
Baseline sin balanceo como referencia.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, SEG_LABELS
from train_ml import get_splits, build_preprocessor

RS = 42


def rf(**kw):
    return RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                  random_state=RS, **kw)


def scores(name, pipe, Xtr, ytr, Xval, yval):
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xval)
    return {
        "técnica": name,
        "accuracy": round(accuracy_score(yval, p), 3),
        "f1_macro": round(f1_score(yval, p, average="macro"), 3),
        "recall_Premium": round(
            recall_score(yval, p, labels=["Premium"], average="macro"), 3),
        "recall_Económico": round(
            recall_score(yval, p, labels=["Económico"], average="macro"), 3),
    }


def main():
    Xtr, Xval, Xte, ytr, yval, yte = get_splits()
    pre = build_preprocessor()
    rows = []

    # baseline
    rows.append(scores("Sin balanceo",
                       Pipeline([("prep", pre), ("clf", rf())]),
                       Xtr, ytr, Xval, yval))
    # class weights
    rows.append(scores("Class weights",
                       Pipeline([("prep", pre), ("clf", rf(class_weight="balanced"))]),
                       Xtr, ytr, Xval, yval))
    # SMOTE
    rows.append(scores("SMOTE",
                       ImbPipeline([("prep", pre),
                                    ("smote", SMOTE(random_state=RS)),
                                    ("clf", rf())]),
                       Xtr, ytr, Xval, yval))
    # undersampling
    rows.append(scores("Undersampling",
                       ImbPipeline([("prep", pre),
                                    ("under", RandomUnderSampler(random_state=RS)),
                                    ("clf", rf())]),
                       Xtr, ytr, Xval, yval))

    tbl = pd.DataFrame(rows).set_index("técnica")
    print("=" * 65)
    print("EFECTO DEL BALANCEO (Random Forest, validación):")
    print(tbl.to_string())
    tbl.to_csv("reports/balancing_results.csv")
    return tbl


if __name__ == "__main__":
    main()
