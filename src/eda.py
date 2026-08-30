"""Generate reproducible exploratory data analysis figures."""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, INVALID_ZERO_COLUMNS, TARGET_COLUMN
from src.data import get_dataset
from src.train import replace_invalid_zeros


def generate_figures() -> None:
    data = get_dataset()
    cleaned = replace_invalid_zeros(data)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x=TARGET_COLUMN, hue=TARGET_COLUMN, legend=False, palette="Set2", ax=ax)
    ax.set(title="Distribuzione della variabile target", xlabel="Diabete (0 = no, 1 = si)", ylabel="Numero di pazienti")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    missing = cleaned[INVALID_ZERO_COLUMNS].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=missing.index, y=missing.values, hue=missing.index, legend=False, palette="crest", ax=ax)
    ax.set(title="Valori zero trattati come mancanti", xlabel="Variabile", ylabel="Numero di valori")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_values.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cleaned.corr(numeric_only=True), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlazioni tra caratteristiche cliniche e outcome")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    print(f"Saved EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_figures()
