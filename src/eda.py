"""
eda.py
------
Genera la Figura 1 del análisis exploratorio utilizando exactamente
el mismo pipeline de limpieza, ingeniería de variables y definición
del target utilizado durante el entrenamiento.

Los segmentos mantienen los mismos colores en todos los gráficos:

    Económico -> verde
    Medio     -> naranja
    Premium   -> rojo

Salida:
    reports/figures/eda_overview.png
"""

from pathlib import Path

import matplotlib

# Permite generar la imagen sin abrir una ventana gráfica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import (
    build_dataset,
    CUT_LOW,
    CUT_HIGH,
    SEG_LABELS,
)


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vehicles_us.csv"
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "Figura1_eda_overview.png"


# ---------------------------------------------------------
# Configuración de colores
# ---------------------------------------------------------

# Se utiliza la misma paleta para representar los segmentos
# en el histograma, dispersión y gráfico de barras.
SEGMENT_COLORS = {
    "Económico": "#2ca02c",  # Verde
    "Medio": "#ff7f0e",      # Naranja
    "Premium": "#d62728",    # Rojo
}

# Colores de las líneas que representan los cortes.
CUT_LOW_COLOR = "#2ca02c"
CUT_HIGH_COLOR = "#d62728"


# ---------------------------------------------------------
# Variables utilizadas en la matriz de correlación
# ---------------------------------------------------------

CORRELATION_COLUMNS = [
    "price",
    "model_year",
    "cylinders",
    "odometer",
    "days_listed",
]


def configure_style() -> None:
    """Configura el estilo general de la figura."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.05,
    )

    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#bfbfbf",
        "axes.linewidth": 0.8,
        "axes.titleweight": "normal",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.labelcolor": "#262626",
        "xtick.color": "#262626",
        "ytick.color": "#262626",
        "text.color": "#262626",
        "grid.color": "#c8c8c8",
        "grid.alpha": 0.70,
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
    })


def format_thousands(value: int) -> str:
    """
    Formatea cantidades utilizando punto como separador de miles.

    Ejemplo:
        11746 -> 11.746
    """

    return f"{int(value):,}".replace(",", ".")


def plot_price_distribution(ax, df: pd.DataFrame) -> None:
    """
    Grafica la distribución de precios.

    Cada segmento utiliza el mismo color que el gráfico de barras:
        Económico -> verde
        Medio     -> naranja
        Premium   -> rojo
    """

    plot_data = df.loc[
        df["price"].between(500, 60000),
        ["price", "segment"],
    ].dropna()

    # Se usan intervalos comunes para que las distribuciones
    # de los tres segmentos sean directamente comparables.
    bins = range(0, 62001, 1000)

    for segment in SEG_LABELS:
        segment_prices = plot_data.loc[
            plot_data["segment"] == segment,
            "price",
        ]

        ax.hist(
            segment_prices,
            bins=bins,
            color=SEGMENT_COLORS[segment],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.90,
            label=segment,
        )

    # Corte entre Económico y Medio.
    ax.axvline(
        CUT_LOW,
        color=CUT_LOW_COLOR,
        linestyle="--",
        linewidth=2.2,
        label=f"Corte USD {format_thousands(CUT_LOW)}",
        zorder=5,
    )

    # Corte entre Medio y Premium.
    ax.axvline(
        CUT_HIGH,
        color=CUT_HIGH_COLOR,
        linestyle="--",
        linewidth=2.2,
        label=f"Corte USD {format_thousands(CUT_HIGH)}",
        zorder=5,
    )

    ax.set_title("Distribución de precio por segmento")
    ax.set_xlabel("Precio (USD)")
    ax.set_ylabel("Cantidad de vehículos")
    ax.set_xlim(0, 62000)

    ax.legend(
        loc="upper right",
        fontsize=8,
        title="Segmento y cortes",
        title_fontsize=9,
    )


def plot_price_vs_odometer(ax, df: pd.DataFrame) -> None:
    """
    Grafica la relación entre precio y odómetro.

    Los puntos se colorean según el segmento real del vehículo.
    """

    plot_data = df.loc[
        df["price"].between(500, 60000)
        & df["odometer"].between(0, 400000),
        ["price", "odometer", "segment"],
    ].dropna()

    # Se limita la muestra para mantener legibilidad y rendimiento.
    # random_state asegura que siempre se obtenga la misma muestra.
    if len(plot_data) > 4000:
        plot_data = plot_data.sample(
            n=4000,
            random_state=42,
        )

    for segment in SEG_LABELS:
        segment_data = plot_data.loc[
            plot_data["segment"] == segment
        ]

        ax.scatter(
            segment_data["odometer"],
            segment_data["price"],
            s=15,
            alpha=0.42,
            color=SEGMENT_COLORS[segment],
            edgecolors="none",
            label=segment,
        )

    # Líneas horizontales que indican los límites de los segmentos.
    ax.axhline(
        CUT_LOW,
        color=CUT_LOW_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.90,
    )

    ax.axhline(
        CUT_HIGH,
        color=CUT_HIGH_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.90,
    )

    ax.set_title("Precio vs. odómetro por segmento")
    ax.set_xlabel("Odómetro")
    ax.set_ylabel("Precio (USD)")
    ax.set_xlim(0, 400000)
    ax.set_ylim(0, 62000)

    ax.legend(
        loc="upper right",
        fontsize=9,
        title="Segmento",
        title_fontsize=9,
    )


def plot_correlation_matrix(ax, df: pd.DataFrame) -> None:
    """
    Grafica la matriz de correlación del conjunto de datos limpio.

    Los tonos azules representan correlaciones negativas y los tonos
    rojos representan correlaciones positivas.
    """

    correlation = (
        df[CORRELATION_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .corr()
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        square=False,
        cbar=True,
        ax=ax,
        annot_kws={
            "size": 10,
        },
        cbar_kws={
            "label": "Coeficiente de correlación",
            "shrink": 0.90,
        },
    )

    ax.set_title("Correlación de variables numéricas")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="x",
        rotation=90,
    )
    ax.tick_params(
        axis="y",
        rotation=0,
    )


def plot_class_balance(ax, df: pd.DataFrame) -> None:
    """
    Grafica la distribución real de la variable objetivo.

    Los colores coinciden con los utilizados en el histograma
    y en el gráfico de dispersión.
    """

    counts = (
        df["segment"]
        .value_counts(sort=False)
        .reindex(SEG_LABELS)
        .fillna(0)
        .astype(int)
    )

    percentages = counts / counts.sum() * 100

    colors = [
        SEGMENT_COLORS[segment]
        for segment in SEG_LABELS
    ]

    bars = ax.bar(
        counts.index,
        counts.values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )

    ax.set_title("Distribución real de la variable objetivo")
    ax.set_xlabel("Segmento")
    ax.set_ylabel("Cantidad de vehículos")
    ax.tick_params(
        axis="x",
        rotation=0,
    )

    max_count = counts.max()
    ax.set_ylim(0, max_count * 1.17)

    for bar, count, percentage in zip(
        bars,
        counts.values,
        percentages.values,
    ):
        label = (
            f"{format_thousands(count)}\n"
            f"({percentage:.1f} %)".replace(".", ",")
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#262626",
        )


def print_summary(df: pd.DataFrame) -> None:
    """Imprime los conteos y porcentajes utilizados en la figura."""

    counts = (
        df["segment"]
        .value_counts(sort=False)
        .reindex(SEG_LABELS)
        .fillna(0)
        .astype(int)
    )

    percentages = counts / counts.sum() * 100

    print("=" * 65)
    print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
    print("=" * 65)
    print(
        "Total después de la limpieza: "
        f"{format_thousands(len(df))}"
    )
    print()

    print("Definición del target:")
    print(
        "  Económico: precio < "
        f"USD {format_thousands(CUT_LOW)}"
    )
    print(
        "  Medio: "
        f"USD {format_thousands(CUT_LOW)} <= precio "
        f"<= USD {format_thousands(CUT_HIGH)}"
    )
    print(
        "  Premium: precio > "
        f"USD {format_thousands(CUT_HIGH)}"
    )
    print()

    for segment in SEG_LABELS:
        formatted_percentage = (
            f"{percentages[segment]:.1f}".replace(".", ",")
        )

        print(
            f"  {segment:<10}: "
            f"{format_thousands(counts[segment]):>6} vehículos "
            f"({formatted_percentage} %)"
        )


def validate_data(df: pd.DataFrame) -> None:
    """Comprueba que los datos necesarios estén disponibles."""

    required_columns = {
        "price",
        "model_year",
        "cylinders",
        "odometer",
        "days_listed",
        "segment",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "El dataset procesado no contiene las columnas requeridas: "
            f"{missing_text}"
        )

    if df.empty:
        raise ValueError(
            "El pipeline de preparación generó un dataset vacío."
        )

    unknown_segments = set(
        df["segment"].dropna().astype(str).unique()
    ).difference(SEG_LABELS)

    if unknown_segments:
        raise ValueError(
            "Se encontraron segmentos no reconocidos: "
            f"{sorted(unknown_segments)}"
        )


def main() -> None:
    """Construye y guarda la figura completa."""

    configure_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el archivo de datos:\n"
            f"{DATA_PATH}"
        )

    # Se aplica exactamente el mismo pipeline utilizado durante
    # el entrenamiento y la construcción del modelo final.
    df = build_dataset(
        str(DATA_PATH)
    )

    validate_data(df)
    print_summary(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14.5, 9.8),
    )

    plot_price_distribution(
        axes[0, 0],
        df,
    )

    plot_price_vs_odometer(
        axes[0, 1],
        df,
    )

    plot_correlation_matrix(
        axes[1, 0],
        df,
    )

    plot_class_balance(
        axes[1, 1],
        df,
    )

    fig.tight_layout(
        pad=1.2,
        h_pad=2.0,
        w_pad=1.8,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format="png",
    )

    plt.close(fig)

    print()
    print("Figura guardada correctamente en:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()