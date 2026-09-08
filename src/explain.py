"""
explain.py
----------
Explicabilidad del modelo con SHAP, calculada directamente sobre el
Random Forest final (200 arboles, con SMOTE, sin la variable
redundante `age`) serializado en models/final_model.joblib. A
diferencia de la version anterior, este script YA NO entrena un
modelo sustituto compacto: usa el mismo clasificador que expone la
API, por lo que el analisis global y las explicaciones locales
corresponden exactamente al objeto en produccion.

Genera:
  - reports/figures/shap_global.png    : importancia global (mean |SHAP|).
  - reports/figures/shap_por_clase.png : drivers por segmento.
  - explain_instance(): funcion reutilizada por backend/app.py para
    generar explicaciones LOCALES (por prediccion) en vez de las
    reglas heuristicas usadas hasta la Sumativa 2.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocessing import SEG_LABELS
from train_ml import get_splits

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.joblib"
FIGURES_DIR = ROOT / "reports" / "figures"

GLOBAL_SAMPLE_SIZE = 60


def _feature_names(preprocessor):
    num = preprocessor.named_transformers_["num"].feature_names_in_.tolist()
    cat = preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
    return num + cat


def load_final_pipeline():
    return joblib.load(MODEL_PATH)


def explain_instance(pipeline, X_single_row):
    """
    Genera una explicacion LOCAL para una fila de entrada, usando el
    Random Forest final real. Devuelve (clase_predicha, lista de
    (variable, contribucion_shap) ordenada por magnitud).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = _feature_names(preprocessor)

    Z = preprocessor.transform(X_single_row)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()

    predicted_class = classifier.predict(Z)[0]
    class_index = list(classifier.classes_).index(predicted_class)

    explainer = shap.TreeExplainer(classifier)
    sv = explainer.shap_values(Z, check_additivity=False)
    sv_class = sv[:, :, class_index] if not isinstance(sv, list) else sv[class_index]

    contributions = pd.Series(sv_class[0], index=names)
    top = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    ).head(5)

    return predicted_class, list(zip(top.index, top.round(4).tolist()))


def main():
    pipeline = load_final_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = _feature_names(preprocessor)
    explainer = shap.TreeExplainer(classifier)

    _, _, X_te, _, _, y_te = get_splits()
    Z_te = preprocessor.transform(X_te)
    if hasattr(Z_te, "toarray"):
        Z_te = Z_te.toarray()

    sample = Z_te[:GLOBAL_SAMPLE_SIZE]
    sv = explainer.shap_values(sample, check_additivity=False)
    sv_list = sv if isinstance(sv, list) else [sv[:, :, k] for k in range(sv.shape[-1])]

    glob = np.mean([np.abs(s).mean(0) for s in sv_list], axis=0)
    imp = pd.Series(glob, index=names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    imp[::-1].plot(kind="barh", color="#008ABC")
    plt.title("Importancia global de variables (SHAP) - modelo final real")
    plt.xlabel("Impacto medio |SHAP|")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "shap_global.png", dpi=120)
    plt.close()

    print("Top 10 variables (SHAP global, modelo final real, sin 'age'):")
    print(imp.head(10).round(4).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for k, (ax, lbl) in enumerate(zip(axes, SEG_LABELS)):
        s = pd.Series(np.abs(sv_list[k]).mean(0), index=names)
        s.sort_values(ascending=False).head(8)[::-1].plot(
            kind="barh", ax=ax, color="#008ABC")
        ax.set_title(f"Segmento: {lbl}")
        ax.set_xlabel("|SHAP|")
    plt.suptitle("Variables que empujan hacia cada segmento (modelo final real)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_por_clase.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\nFiguras SHAP guardadas en reports/figures/")

    example_row = X_te.iloc[[0]]
    predicted_class, local = explain_instance(pipeline, example_row)
    print(f"\nEjemplo de explicacion LOCAL (prediccion: {predicted_class}):")
    for name, value in local:
        print(f"  {name}: {value:+.4f}")


if __name__ == "__main__":
    main()
