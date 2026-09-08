"""
homogeneous_comparison.py
--------------------------
Comparación de técnicas de ML y DL bajo un protocolo de balanceo
homogéneo (SMOTE aplicado únicamente sobre el conjunto de
entrenamiento, para las seis técnicas), en respuesta a la
retroalimentación docente sobre la Sumativa 2: la comparación previa
no era controlada porque cada familia de modelos usaba un tratamiento
distinto del desbalance (class_weight en ML clásico y en las MLP,
ningún tratamiento en XGBoost).

Salida: reports/homogeneous_comparison_results.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEG_LABELS
from train_ml import get_splits

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "homogeneous_comparison_results.csv"

RANDOM_STATE = 42
EPOCHS = 40
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 6

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def create_early_stopping():
    """Instancia independiente de EarlyStopping (una por modelo)."""
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )


def prepare_flat_arrays():
    """
    Codifica numéricas (StandardScaler) + categóricas (OneHot) y aplica
    SMOTE sobre el conjunto de entrenamiento únicamente. Devuelve
    también el conjunto de validación sin remuestrear.
    """
    X_tr, X_val, X_te, y_tr, y_val, y_te = get_splits()

    label_encoder = LabelEncoder().fit(SEG_LABELS)
    y_tr_i = label_encoder.transform(y_tr)
    y_val_i = label_encoder.transform(y_val)

    scaler = StandardScaler().fit(X_tr[NUMERIC_FEATURES])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        X_tr[CATEGORICAL_FEATURES]
    )

    def build_flat(X):
        return np.hstack([
            scaler.transform(X[NUMERIC_FEATURES]),
            ohe.transform(X[CATEGORICAL_FEATURES]),
        ])

    flat_tr = build_flat(X_tr)
    flat_val = build_flat(X_val)

    smote = SMOTE(random_state=RANDOM_STATE)
    flat_tr_res, y_tr_res = smote.fit_resample(flat_tr, y_tr_i)

    return flat_tr_res, flat_val, y_tr_res, y_val_i, label_encoder


def build_shallow(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_deep(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"modelo": name, "accuracy": round(acc, 3), "f1_macro": round(f1m, 3)}


def main():
    flat_tr, flat_val, y_tr, y_val, label_encoder = prepare_flat_arrays()
    print(f"Train (post-SMOTE)={len(flat_tr)}  Val={len(flat_val)}")
    print("Distribución balanceada de entrenamiento:",
          np.bincount(y_tr))

    results = []

    # ---- ML clásico, sin class_weight (SMOTE ya balanceó) ----
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(flat_tr, y_tr)
    results.append(evaluate("Regresión Logística", y_val,
                             log_reg.predict(flat_val)))

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1,
                                 random_state=RANDOM_STATE)
    rf.fit(flat_tr, y_tr)
    results.append(evaluate("Random Forest", y_val, rf.predict(flat_val)))

    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9,
                         tree_method="hist", eval_metric="mlogloss",
                         random_state=RANDOM_STATE)
    xgb.fit(flat_tr, y_tr)
    results.append(evaluate("XGBoost", y_val, xgb.predict(flat_val)))

    # ---- Deep Learning, sin class_weight (SMOTE ya balanceó) ----
    input_dim = flat_tr.shape[1]

    shallow = build_shallow(input_dim)
    shallow.fit(flat_tr, y_tr, validation_data=(flat_val, y_val),
                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                callbacks=[create_early_stopping()])
    shallow_pred = np.argmax(shallow.predict(flat_val, verbose=0), axis=1)
    results.append(evaluate("MLP Shallow", y_val, shallow_pred))

    deep = build_deep(input_dim)
    deep.fit(flat_tr, y_tr, validation_data=(flat_val, y_val),
             epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
             callbacks=[create_early_stopping()])
    deep_pred = np.argmax(deep.predict(flat_val, verbose=0), axis=1)
    results.append(evaluate("MLP Profunda", y_val, deep_pred))

    tbl = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("COMPARACIÓN HOMOGÉNEA (SMOTE en todas las técnicas)")
    print("=" * 60)
    print(tbl.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"\nGuardado en: {RESULTS_PATH}")
    return tbl


if __name__ == "__main__":
    main()
