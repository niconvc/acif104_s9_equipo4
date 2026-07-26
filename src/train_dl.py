"""
train_dl.py
-----------
Tres arquitecturas de Deep Learning para clasificar el segmento de precio:

  1) MLP Shallow      : 1 capa oculta (64) - baseline neuronal.
  2) MLP Profunda     : 3 capas (128-64-32) + BatchNorm + Dropout - regularizada.
  3) MLP + Embeddings : embeddings para categóricas de alta cardinalidad
                        (brand, type...) concatenados con numéricas.

Todas usan softmax de salida (3 clases) y se entrenan con class_weight
para compensar el desbalance. Métrica principal: F1-macro.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

from preprocessing import (build_dataset, NUMERIC_FEATURES,
                           CATEGORICAL_FEATURES, SEG_LABELS)
from train_ml import get_splits

tf.random.set_seed(42)
np.random.seed(42)
EPOCHS = 40
BATCH = 256


def prepare_arrays():
    X_tr, X_val, X_te, y_tr, y_val, y_te = get_splits()

    # target -> entero
    le = LabelEncoder().fit(SEG_LABELS)
    y_tr_i, y_val_i, y_te_i = (le.transform(y) for y in (y_tr, y_val, y_te))

    # numéricas escaladas
    scaler = StandardScaler().fit(X_tr[NUMERIC_FEATURES])
    num_tr = scaler.transform(X_tr[NUMERIC_FEATURES])
    num_val = scaler.transform(X_val[NUMERIC_FEATURES])
    num_te = scaler.transform(X_te[NUMERIC_FEATURES])

    # categóricas -> enteros (para embeddings)
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    oe.fit(X_tr[CATEGORICAL_FEATURES])
    cat_tr = oe.transform(X_tr[CATEGORICAL_FEATURES]).astype(int) + 1  # +1: 0 = desconocido
    cat_val = oe.transform(X_val[CATEGORICAL_FEATURES]).astype(int) + 1
    cat_te = oe.transform(X_te[CATEGORICAL_FEATURES]).astype(int) + 1
    cat_dims = [int(cat_tr[:, i].max()) + 2 for i in range(cat_tr.shape[1])]

    # versión "plana" (numéricas + one-hot) para MLP 1 y 2
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        X_tr[CATEGORICAL_FEATURES])
    flat_tr = np.hstack([num_tr, ohe.transform(X_tr[CATEGORICAL_FEATURES])])
    flat_val = np.hstack([num_val, ohe.transform(X_val[CATEGORICAL_FEATURES])])
    flat_te = np.hstack([num_te, ohe.transform(X_te[CATEGORICAL_FEATURES])])

    cw = compute_class_weight("balanced", classes=np.unique(y_tr_i), y=y_tr_i)
    class_weight = {i: w for i, w in enumerate(cw)}

    return dict(
        num=(num_tr, num_val, num_te), cat=(cat_tr, cat_val, cat_te),
        flat=(flat_tr, flat_val, flat_te), y=(y_tr_i, y_val_i, y_te_i),
        cat_dims=cat_dims, class_weight=class_weight)


def build_shallow(input_dim):
    m = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ], name="MLP_Shallow")
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_deep(input_dim):
    m = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ], name="MLP_Deep")
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_embed(num_dim, cat_dims):
    num_in = keras.Input(shape=(num_dim,), name="num")
    cat_ins, embs = [], []
    for i, dim in enumerate(cat_dims):
        ci = keras.Input(shape=(1,), name=f"cat_{i}")
        emb_size = min(10, (dim + 1) // 2)
        e = layers.Embedding(dim, emb_size)(ci)
        embs.append(layers.Flatten()(e))
        cat_ins.append(ci)
    x = layers.Concatenate()([num_in] + embs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(3, activation="softmax")(x)
    m = keras.Model([num_in] + cat_ins, out, name="MLP_Embeddings")
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def evaluate(name, y_true, y_prob, histories):
    y_pred = y_prob.argmax(1)
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 3)}


def main():
    d = prepare_arrays()
    (flat_tr, flat_val, flat_te) = d["flat"]
    (num_tr, num_val, num_te) = d["num"]
    (cat_tr, cat_val, cat_te) = d["cat"]
    (y_tr, y_val, y_te) = d["y"]
    cw = d["class_weight"]
    es = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    results, hist = [], {}

    # 1 shallow
    m1 = build_shallow(flat_tr.shape[1])
    h1 = m1.fit(flat_tr, y_tr, validation_data=(flat_val, y_val), epochs=EPOCHS,
                batch_size=BATCH, class_weight=cw, callbacks=[es], verbose=0)
    results.append(evaluate("MLP_Shallow", y_val, m1.predict(flat_val, verbose=0), h1))
    hist["MLP_Shallow"] = h1.history

    # 2 deep
    m2 = build_deep(flat_tr.shape[1])
    h2 = m2.fit(flat_tr, y_tr, validation_data=(flat_val, y_val), epochs=EPOCHS,
                batch_size=BATCH, class_weight=cw, callbacks=[es], verbose=0)
    results.append(evaluate("MLP_Deep", y_val, m2.predict(flat_val, verbose=0), h2))
    hist["MLP_Deep"] = h2.history

    # 3 embeddings
    m3 = build_embed(num_tr.shape[1], d["cat_dims"])
    tr_in = [num_tr] + [cat_tr[:, i] for i in range(cat_tr.shape[1])]
    val_in = [num_val] + [cat_val[:, i] for i in range(cat_val.shape[1])]
    h3 = m3.fit(tr_in, y_tr, validation_data=(val_in, y_val), epochs=EPOCHS,
                batch_size=BATCH, class_weight=cw, callbacks=[es], verbose=0)
    results.append(evaluate("MLP_Embeddings", y_val, m3.predict(val_in, verbose=0), h3))
    hist["MLP_Embeddings"] = h3.history

    tbl = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 55)
    print("COMPARATIVO DEEP LEARNING (validación):")
    print(tbl.to_string())
    # guardar histories para graficar convergencia
    np.save("models/dl_histories.npy", hist, allow_pickle=True)
    return tbl, hist


if __name__ == "__main__":
    main()
