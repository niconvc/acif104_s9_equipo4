"""
train_dl.py
-----------
Tres arquitecturas de Deep Learning para clasificar el segmento de precio:

  1) MLP Shallow      : 1 capa oculta (64) - baseline neuronal.
  2) MLP Profunda     : 3 capas (128-64-32) + BatchNorm + Dropout.
  3) MLP + Embeddings : embeddings para variables categóricas,
                        concatenados con las variables numéricas.

Todas utilizan una salida softmax de tres clases y se entrenan con
class_weight para compensar el desbalance.

Métrica principal:
    F1-macro.

Importante:
    Cada entrenamiento crea su propia instancia de EarlyStopping.
    Los callbacks no se reutilizan entre modelos porque conservan
    estado interno del entrenamiento anterior.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEG_LABELS,
)
from train_ml import get_splits


# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HISTORIES_PATH = MODELS_DIR / "dl_histories.npy"

RANDOM_STATE = 42
EPOCHS = 40
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 6

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------

def create_early_stopping():
    """
    Crea una instancia independiente de EarlyStopping.

    No se debe reutilizar el mismo callback entre varios modelos,
    porque EarlyStopping conserva información interna, como la mejor
    pérdida de validación observada y el número de épocas sin mejora.
    """

    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )


# ---------------------------------------------------------
# Preparación de los datos
# ---------------------------------------------------------

def prepare_arrays():
    """
    Obtiene las particiones de entrenamiento, validación y prueba.

    Genera:

    - Variables numéricas estandarizadas.
    - Variables categóricas codificadas como enteros para embeddings.
    - Variables categóricas codificadas mediante one-hot para las MLP.
    - Target convertido a valores enteros.
    - Pesos de clase calculados desde el conjunto de entrenamiento.
    """

    X_tr, X_val, X_te, y_tr, y_val, y_te = get_splits()

    # -----------------------------------------------------
    # Target convertido a enteros
    # -----------------------------------------------------

    label_encoder = LabelEncoder()
    label_encoder.fit(SEG_LABELS)

    y_tr_i = label_encoder.transform(y_tr)
    y_val_i = label_encoder.transform(y_val)
    y_te_i = label_encoder.transform(y_te)

    # -----------------------------------------------------
    # Variables numéricas estandarizadas
    # -----------------------------------------------------

    scaler = StandardScaler()
    scaler.fit(X_tr[NUMERIC_FEATURES])

    num_tr = scaler.transform(
        X_tr[NUMERIC_FEATURES]
    )

    num_val = scaler.transform(
        X_val[NUMERIC_FEATURES]
    )

    num_te = scaler.transform(
        X_te[NUMERIC_FEATURES]
    )

    # -----------------------------------------------------
    # Variables categóricas para embeddings
    # -----------------------------------------------------

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    ordinal_encoder.fit(
        X_tr[CATEGORICAL_FEATURES]
    )

    # Se suma 1 para reservar el valor 0 a categorías desconocidas.
    cat_tr = (
        ordinal_encoder
        .transform(X_tr[CATEGORICAL_FEATURES])
        .astype(int)
        + 1
    )

    cat_val = (
        ordinal_encoder
        .transform(X_val[CATEGORICAL_FEATURES])
        .astype(int)
        + 1
    )

    cat_te = (
        ordinal_encoder
        .transform(X_te[CATEGORICAL_FEATURES])
        .astype(int)
        + 1
    )

    cat_dims = [
        int(cat_tr[:, index].max()) + 2
        for index in range(cat_tr.shape[1])
    ]

    # -----------------------------------------------------
    # Datos planos para MLP Shallow y MLP Profunda
    # -----------------------------------------------------

    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    one_hot_encoder.fit(
        X_tr[CATEGORICAL_FEATURES]
    )

    flat_tr = np.hstack([
        num_tr,
        one_hot_encoder.transform(
            X_tr[CATEGORICAL_FEATURES]
        ),
    ])

    flat_val = np.hstack([
        num_val,
        one_hot_encoder.transform(
            X_val[CATEGORICAL_FEATURES]
        ),
    ])

    flat_te = np.hstack([
        num_te,
        one_hot_encoder.transform(
            X_te[CATEGORICAL_FEATURES]
        ),
    ])

    # -----------------------------------------------------
    # Pesos para compensar el desbalance de clases
    # -----------------------------------------------------

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_tr_i),
        y=y_tr_i,
    )

    class_weight = {
        index: float(weight)
        for index, weight in enumerate(class_weights_array)
    }

    return {
        "num": (
            num_tr,
            num_val,
            num_te,
        ),
        "cat": (
            cat_tr,
            cat_val,
            cat_te,
        ),
        "flat": (
            flat_tr,
            flat_val,
            flat_te,
        ),
        "y": (
            y_tr_i,
            y_val_i,
            y_te_i,
        ),
        "cat_dims": cat_dims,
        "class_weight": class_weight,
        "label_encoder": label_encoder,
    }


# ---------------------------------------------------------
# Arquitectura 1: MLP Shallow
# ---------------------------------------------------------

def build_shallow(input_dim):
    """Construye una MLP con una capa oculta de 64 neuronas."""

    model = keras.Sequential(
        [
            layers.Input(
                shape=(input_dim,),
                name="features",
            ),
            layers.Dense(
                64,
                activation="relu",
                name="hidden_64",
            ),
            layers.Dense(
                len(SEG_LABELS),
                activation="softmax",
                name="segment_output",
            ),
        ],
        name="MLP_Shallow",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------
# Arquitectura 2: MLP Profunda
# ---------------------------------------------------------

def build_deep(input_dim):
    """
    Construye una MLP profunda con BatchNorm y Dropout.

    Arquitectura:
        128 -> BatchNorm -> Dropout
         64 -> BatchNorm -> Dropout
         32
          3 -> Softmax
    """

    model = keras.Sequential(
        [
            layers.Input(
                shape=(input_dim,),
                name="features",
            ),
            layers.Dense(
                128,
                activation="relu",
                name="hidden_128",
            ),
            layers.BatchNormalization(
                name="batch_norm_1",
            ),
            layers.Dropout(
                0.3,
                name="dropout_1",
            ),
            layers.Dense(
                64,
                activation="relu",
                name="hidden_64",
            ),
            layers.BatchNormalization(
                name="batch_norm_2",
            ),
            layers.Dropout(
                0.3,
                name="dropout_2",
            ),
            layers.Dense(
                32,
                activation="relu",
                name="hidden_32",
            ),
            layers.Dense(
                len(SEG_LABELS),
                activation="softmax",
                name="segment_output",
            ),
        ],
        name="MLP_Deep",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------
# Arquitectura 3: MLP con embeddings
# ---------------------------------------------------------

def build_embeddings(num_dim, cat_dims):
    """
    Construye una MLP con embeddings para variables categóricas.

    Cada variable categórica tiene su propia entrada y su propia
    capa de embedding. Posteriormente, las representaciones se
    concatenan con las variables numéricas.
    """

    numeric_input = keras.Input(
        shape=(num_dim,),
        name="numeric_features",
    )

    categorical_inputs = []
    embedding_outputs = []

    for index, dimension in enumerate(cat_dims):
        categorical_input = keras.Input(
            shape=(1,),
            dtype="int32",
            name=f"categorical_{index}",
        )

        embedding_size = min(
            10,
            (dimension + 1) // 2,
        )

        embedding = layers.Embedding(
            input_dim=dimension,
            output_dim=embedding_size,
            name=f"embedding_{index}",
        )(categorical_input)

        flattened_embedding = layers.Flatten(
            name=f"flatten_embedding_{index}",
        )(embedding)

        categorical_inputs.append(
            categorical_input
        )

        embedding_outputs.append(
            flattened_embedding
        )

    combined_features = layers.Concatenate(
        name="combined_features",
    )(
        [numeric_input] + embedding_outputs
    )

    hidden = layers.Dense(
        128,
        activation="relu",
        name="hidden_128",
    )(combined_features)

    hidden = layers.Dropout(
        0.3,
        name="dropout",
    )(hidden)

    hidden = layers.Dense(
        64,
        activation="relu",
        name="hidden_64",
    )(hidden)

    output = layers.Dense(
        len(SEG_LABELS),
        activation="softmax",
        name="segment_output",
    )(hidden)

    model = keras.Model(
        inputs=[numeric_input] + categorical_inputs,
        outputs=output,
        name="MLP_Embeddings",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------
# Evaluación
# ---------------------------------------------------------

def evaluate_model(name, y_true, y_probabilities):
    """
    Evalúa un modelo mediante accuracy y F1-macro.

    La evaluación se realiza sobre el conjunto de validación.
    """

    y_pred = y_probabilities.argmax(axis=1)

    return {
        "model": name,
        "accuracy": round(
            accuracy_score(y_true, y_pred),
            3,
        ),
        "f1_macro": round(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            ),
            3,
        ),
    }


# ---------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------

def main():
    """Entrena y compara las tres arquitecturas neuronales."""

    data = prepare_arrays()

    flat_tr, flat_val, _ = data["flat"]
    num_tr, num_val, _ = data["num"]
    cat_tr, cat_val, _ = data["cat"]
    y_tr, y_val, _ = data["y"]

    class_weight = data["class_weight"]

    results = []
    histories = {}

    # -----------------------------------------------------
    # 1. MLP Shallow
    # -----------------------------------------------------

    print("\nEntrenando MLP Shallow...")

    shallow_model = build_shallow(
        flat_tr.shape[1]
    )

    shallow_history = shallow_model.fit(
        flat_tr,
        y_tr,
        validation_data=(
            flat_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            create_early_stopping()
        ],
        verbose=1,
    )

    shallow_probabilities = shallow_model.predict(
        flat_val,
        verbose=0,
    )

    results.append(
        evaluate_model(
            "MLP_Shallow",
            y_val,
            shallow_probabilities,
        )
    )

    histories["MLP_Shallow"] = (
        shallow_history.history
    )

    # -----------------------------------------------------
    # 2. MLP Profunda
    # -----------------------------------------------------

    print("\nEntrenando MLP Profunda...")

    deep_model = build_deep(
        flat_tr.shape[1]
    )

    deep_history = deep_model.fit(
        flat_tr,
        y_tr,
        validation_data=(
            flat_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            create_early_stopping()
        ],
        verbose=1,
    )

    deep_probabilities = deep_model.predict(
        flat_val,
        verbose=0,
    )

    results.append(
        evaluate_model(
            "MLP_Deep",
            y_val,
            deep_probabilities,
        )
    )

    histories["MLP_Deep"] = (
        deep_history.history
    )

    # -----------------------------------------------------
    # 3. MLP con embeddings
    # -----------------------------------------------------

    print("\nEntrenando MLP con embeddings...")

    embeddings_model = build_embeddings(
        num_tr.shape[1],
        data["cat_dims"],
    )

    train_inputs = [
        num_tr
    ] + [
        cat_tr[:, index]
        for index in range(cat_tr.shape[1])
    ]

    validation_inputs = [
        num_val
    ] + [
        cat_val[:, index]
        for index in range(cat_val.shape[1])
    ]

    embeddings_history = embeddings_model.fit(
        train_inputs,
        y_tr,
        validation_data=(
            validation_inputs,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            create_early_stopping()
        ],
        verbose=1,
    )

    embeddings_probabilities = embeddings_model.predict(
        validation_inputs,
        verbose=0,
    )

    results.append(
        evaluate_model(
            "MLP_Embeddings",
            y_val,
            embeddings_probabilities,
        )
    )

    histories["MLP_Embeddings"] = (
        embeddings_history.history
    )

    # -----------------------------------------------------
    # Resultados
    # -----------------------------------------------------

    results_table = (
        pd.DataFrame(results)
        .set_index("model")
    )

    print("\n" + "=" * 55)
    print("COMPARATIVO DEEP LEARNING (validación):")
    print(results_table.to_string())

    # -----------------------------------------------------
    # Guardar historiales
    # -----------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        HISTORIES_PATH,
        histories,
        allow_pickle=True,
    )

    print()
    print("Historiales guardados correctamente en:")
    print(HISTORIES_PATH)

    return results_table, histories


if __name__ == "__main__":
    main()