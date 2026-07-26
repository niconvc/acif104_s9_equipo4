"""
preprocessing.py
----------------
Limpieza, ingeniería de características y creación del target de segmento
de precio para el dataset vehicles_us.csv.

Segmentos (umbrales de mercado):
    Económico : price <  5.000 USD
    Medio     : 5.000 <= price <= 20.000 USD
    Premium   : price >  20.000 USD

Autores: Grupo de trabajo - Fase 2
"""
import numpy as np
import pandas as pd

# --- Umbrales de segmento (cortes de mercado) ---
CUT_LOW = 5000
CUT_HIGH = 20000
SEG_LABELS = ["Económico", "Medio", "Premium"]

# Orden natural de la condición (para convertirla a numérica ordinal)
CONDITION_ORDER = {
    "salvage": 0, "fair": 1, "good": 2,
    "excellent": 3, "like new": 4, "new": 5,
}


def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV crudo."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de calidad de datos: outliers, valores imposibles y nulos."""
    df = df.copy()

    # 1) Precios basura / extremos (avisos de prueba y de lujo atípicos)
    df = df[(df["price"] >= 500) & (df["price"] <= 300000)]

    # 2) is_4wd: la ausencia significa "no 4x4" -> 0/1
    df["is_4wd"] = df["is_4wd"].fillna(0).astype(int)

    # 3) model_year: imputar por mediana del modelo, luego global
    df["model_year"] = df.groupby("model")["model_year"].transform(
        lambda s: s.fillna(s.median())
    )
    df["model_year"] = df["model_year"].fillna(df["model_year"].median())
    df = df[df["model_year"] >= 1980]  # descartar autos de colección

    # 4) odometer: 0 km es improbable en usados -> NaN e imputar por mediana del modelo
    df.loc[df["odometer"] == 0, "odometer"] = np.nan
    df.loc[df["odometer"] > 400000, "odometer"] = np.nan
    df["odometer"] = df.groupby("model")["odometer"].transform(
        lambda s: s.fillna(s.median())
    )
    df["odometer"] = df["odometer"].fillna(df["odometer"].median())

    # 5) cylinders: imputar por mediana del modelo
    df["cylinders"] = df.groupby("model")["cylinders"].transform(
        lambda s: s.fillna(s.median())
    )
    df["cylinders"] = df["cylinders"].fillna(df["cylinders"].median())

    # 6) paint_color: categoría explícita "unknown"
    df["paint_color"] = df["paint_color"].fillna("unknown")

    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas útiles para el modelo."""
    df = df.copy()

    # Año de referencia = año del aviso más reciente del dataset (2019)
    ref_year = pd.to_datetime(df["date_posted"]).dt.year.max()
    df["age"] = ref_year - df["model_year"]
    df["age"] = df["age"].clip(lower=0)

    # Kilometraje por año (proxy de uso/desgaste)
    df["odometer_per_year"] = df["odometer"] / df["age"].replace(0, 1)

    # Marca = primera palabra del modelo
    df["brand"] = df["model"].str.split().str[0]

    # Condición ordinal
    df["condition_ord"] = df["condition"].map(CONDITION_ORDER).fillna(2)

    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la etiqueta de segmento de precio."""
    df = df.copy()
    df["segment"] = pd.cut(
        df["price"],
        bins=[-np.inf, CUT_LOW - 1, CUT_HIGH, np.inf],
        labels=SEG_LABELS,
    )
    return df


def build_dataset(path: str) -> pd.DataFrame:
    """Pipeline completo: cargar -> limpiar -> feature eng -> target."""
    df = load_raw(path)
    df = clean(df)
    df = engineer(df)
    df = make_target(df)
    return df


# Columnas que entran al modelo (price se excluye: es el origen del target)
NUMERIC_FEATURES = [
    "model_year", "age", "odometer", "odometer_per_year",
    "cylinders", "condition_ord", "is_4wd", "days_listed",
]
CATEGORICAL_FEATURES = ["brand", "fuel", "transmission", "type", "paint_color"]


if __name__ == "__main__":
    d = build_dataset("data/vehicles_us.csv")
    print("Filas finales:", len(d))
    print("\nDistribución del target:")
    print(d["segment"].value_counts().sort_index())
    print("\nNulos restantes en features:")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    print(d[cols].isna().sum()[lambda s: s > 0] if d[cols].isna().sum().sum() else "0")
