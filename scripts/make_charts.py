"""Makes the charts used in the report and slides.

    pip install matplotlib
    python scripts/make_charts.py

Writes three images to docs/charts/. Each one answers a question we expect to be
asked:

    class-balance.png      why is accuracy a misleading number here?
    threshold-tuning.png   what did we change, and what did it cost?
    confusion-matrix.png   does the model make the mistake that matters?
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "charts"
sys.path.insert(0, str(ROOT / "services" / "inference"))
from train import COLUMNS, FEATURES, LABELS, CLASS_WEIGHTS, apply_thresholds

INK, DIM, LINE = "#202823", "#4F5E55", "#CED8D2"
COLOURS = {"Optimal": "#2E7D32", "Warning": "#D88900", "Critical": "#D13D3D"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 9, "text.color": INK, "axes.edgecolor": LINE,
                     "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def tidy(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def chart_balance(df):
    """Shows how rare the Critical readings are."""
    counts = df.Growing_Condition.value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    bars = ax.barh(LABELS[::-1], counts[::-1],
                   color=[COLOURS[l] for l in LABELS[::-1]], height=0.6)
    for bar, label in zip(bars, LABELS[::-1]):
        n = counts[label]
        ax.text(bar.get_width() + 90, bar.get_y() + bar.get_height() / 2,
                f"{n:,}  ({n / len(df) * 100:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_xlabel("readings")
    ax.set_title("Only 215 of 10,000 readings are Critical", fontsize=11, fontweight="bold")
    ax.grid(axis="y", visible=False)
    tidy(ax)
    fig.savefig(OUT / "class-balance.png")
    plt.close(fig)


def chart_thresholds(proba, classes, y_true):
    """Shows accuracy staying flat while Critical recall collapses."""
    ci = classes.index("Critical")
    grid = np.linspace(0.05, 0.6, 35)
    recall, precision, accuracy = [], [], []
    for t in grid:
        pred = np.array([classes[i] for i in proba.argmax(axis=1)], dtype=object)
        pred[proba[:, classes.index("Warning")] >= 0.25] = "Warning"
        pred[proba[:, ci] >= t] = "Critical"
        r = classification_report(y_true, pred, output_dict=True, zero_division=0)
        recall.append(r["Critical"]["recall"])
        precision.append(r["Critical"]["precision"])
        accuracy.append(r["accuracy"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, recall, color=COLOURS["Critical"], lw=2.2, label="Critical readings found")
    ax.plot(grid, precision, color=DIM, lw=1.6, ls="--", label="Critical alerts that are real")
    ax.plot(grid, accuracy, color=COLOURS["Optimal"], lw=1.6, ls=":", label="overall accuracy")
    ax.axvline(0.15, color=INK, lw=1, alpha=0.6)
    ax.annotate("what we use\n0.15", xy=(0.15, 0.3), xytext=(0.19, 0.2), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate("the default", xy=(0.5, 0.47), xytext=(0.42, 0.3), fontsize=8.5, color=DIM,
                arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8))
    ax.set_xlabel("how sure the model must be before it says Critical")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.05, 0.6)
    ax.set_title("Accuracy barely moves while Critical detection collapses",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    tidy(ax)
    fig.savefig(OUT / "threshold-tuning.png")
    plt.close(fig)


def chart_confusion(y_true, y_pred):
    """Shows that no Critical reading is mistaken for a healthy one."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    shaded = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(shaded, cmap="Greens", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white" if shaded[i, j] > 0.5 else INK)
    ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False,
                               edgecolor=COLOURS["Critical"], lw=2.5))
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("what the model said")
    ax.set_ylabel("what it actually was")
    ax.set_title("No Critical reading was called healthy", fontsize=11, fontweight="bold")
    ax.grid(visible=False)
    tidy(ax, keep=())
    fig.savefig(OUT / "confusion-matrix.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "greenhouse_conditions.csv").rename(columns=COLUMNS)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df.Growing_Condition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                          class_weight=CLASS_WEIGHTS,
                                          random_state=42, n_jobs=-1))]).fit(X_train, y_train)
    classes = list(model.named_steps["forest"].classes_)
    proba = model.predict_proba(X_test)

    chart_balance(df)
    chart_thresholds(proba, classes, y_test)
    chart_confusion(y_test, apply_thresholds(proba, classes))

    for path in sorted(OUT.glob("*.png")):
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
