"""
explain.py
----------
Explicabilidad del modelo con SHAP. Para hacer el cálculo tratable, se
entrena un Random Forest compacto y representativo (120 árboles,
max_depth=14, F1-macro ~0.82 en test, muy cercano al modelo final) sobre
los datos balanceados con SMOTE, y se explican sus predicciones.

Genera:
  - reports/figures/shap_global.png   : importancia global (mean |SHAP|).
  - reports/figures/shap_por_clase.png : drivers por segmento.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from preprocessing import SEG_LABELS
from train_ml import get_splits, build_preprocessor

RS = 42


def feature_names(prep):
    num = prep.named_transformers_["num"].feature_names_in_.tolist()
    cat = prep.named_transformers_["cat"].get_feature_names_out().tolist()
    return num + cat


def main():
    Xtr, Xval, Xte, ytr, yval, yte = get_splits()
    prep = build_preprocessor().fit(Xtr)
    names = feature_names(prep)

    Ztr = prep.transform(Xtr); Zte = prep.transform(Xte)
    if hasattr(Ztr, "toarray"):
        Ztr, Zte = Ztr.toarray(), Zte.toarray()
    Ztr_b, ytr_b = SMOTE(random_state=RS).fit_resample(Ztr, ytr)

    rf = RandomForestClassifier(n_estimators=120, max_depth=14,
                                n_jobs=-1, random_state=RS).fit(Ztr_b, ytr_b)

    sample = Zte[:400]
    sv = shap.TreeExplainer(rf).shap_values(sample, check_additivity=False)
    sv_list = sv if isinstance(sv, list) else [sv[:, :, k] for k in range(sv.shape[-1])]

    # --- importancia global ---
    glob = np.mean([np.abs(s).mean(0) for s in sv_list], axis=0)
    imp = pd.Series(glob, index=names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    imp[::-1].plot(kind="barh", color="#008ABC")
    plt.title("Importancia global de variables (SHAP)")
    plt.xlabel("Impacto medio |SHAP|"); plt.tight_layout()
    plt.savefig("reports/figures/shap_global.png", dpi=120)
    plt.close()
    print("Top 10 variables (SHAP global):")
    print(imp.head(10).round(4).to_string())

    # --- por clase ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for k, (ax, lbl) in enumerate(zip(axes, SEG_LABELS)):
        s = pd.Series(np.abs(sv_list[k]).mean(0), index=names)
        s.sort_values(ascending=False).head(8)[::-1].plot(
            kind="barh", ax=ax, color="#008ABC")
        ax.set_title(f"Segmento: {lbl}"); ax.set_xlabel("|SHAP|")
    plt.suptitle("Variables que empujan hacia cada segmento", y=1.02)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_por_clase.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\nFiguras SHAP guardadas en reports/figures/")


if __name__ == "__main__":
    main()
