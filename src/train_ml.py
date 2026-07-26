"""
train_ml.py
-----------
Entrena y compara 3 técnicas clásicas de Machine Learning para la
clasificación de segmento de precio:
    - Regresión Logística (baseline lineal, interpretable)
    - Random Forest (ensamble de árboles, no lineal, robusto)
    - XGBoost (gradient boosting, alto rendimiento)

Partición estratificada 70/15/15 (train/val/test).
Métricas: accuracy, F1-macro (clave por el desbalance) y F1 por clase.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

from preprocessing import (build_dataset, NUMERIC_FEATURES,
                           CATEGORICAL_FEATURES, SEG_LABELS)

RANDOM_STATE = 42


def get_splits(path="data/vehicles_us.csv"):
    df = build_dataset(path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["segment"].astype(str)

    # 70 / 15 / 15 estratificado
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="mlogloss", random_state=RANDOM_STATE),
    }


def evaluate(name, pipe, X, y_true):
    y_pred = pipe.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1c = f1_score(y_true, y_pred, average=None, labels=SEG_LABELS)
    return {"model": name, "accuracy": acc, "f1_macro": f1m,
            **{f"f1_{lbl}": v for lbl, v in zip(SEG_LABELS, f1c)}}


def main():
    X_tr, X_val, X_te, y_tr, y_val, y_te = get_splits()
    print(f"Train={len(X_tr)}  Val={len(X_val)}  Test={len(X_te)}\n")

    pre = build_preprocessor()
    results = []
    for name, clf in get_models().items():
        # XGBoost necesita etiquetas numéricas
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        if name == "XGBoost":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder().fit(y_tr)
            pipe.fit(X_tr, le.transform(y_tr))
            # envolver predict para devolver etiquetas de texto
            raw_predict = pipe.predict
            pipe.predict = lambda X, _le=le, _p=raw_predict: _le.inverse_transform(_p(X))
        else:
            pipe.fit(X_tr, y_tr)

        res = evaluate(name, pipe, X_val, y_val)
        results.append(res)
        print(f"== {name} (validación) ==")
        print(f"  accuracy={res['accuracy']:.3f}  f1_macro={res['f1_macro']:.3f}")

    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO (validación):")
    tbl = pd.DataFrame(results).set_index("model").round(3)
    print(tbl.to_string())
    return tbl


if __name__ == "__main__":
    main()
