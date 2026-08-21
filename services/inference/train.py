"""Train the growing-condition classifier.

    python train.py

THE PROBLEM
-----------
The dataset is 10,000 readings labelled Optimal (83%), Warning (15%) and
Critical (2%). Only 215 rows are Critical, and those are the only ones that
matter: a missed Critical means a dead crop, a false Critical means a five
minute walk to a healthy rack.

A normal RandomForest on this data is 96% accurate and finds only 44% of the
Critical readings. It reaches that accuracy by almost never predicting the rare
class. Accuracy is the wrong number to report here.

THE FIX
-------
Two changes, both in this file:

1. Class weights. Optimal 1, Warning 5, Critical 30, so a mistake on a Critical
   reading is penalised 30 times more heavily during training.

2. A lower decision threshold. Instead of taking the most likely class, we call
   a reading Critical whenever P(Critical) >= 0.15, and Warning whenever
   P(Warning) >= 0.25.

    Configuration            Critical recall    Critical precision
    normal (argmax)               0.47                0.91
    thresholds 0.15 / 0.25        0.93                0.46

Recall doubles. The cost is precision: about half the Critical alerts are false.
That is only acceptable because the dashboard waits for TWO bad readings in a
row before alerting anyone, and false alarms do not repeat while real faults do.
The threshold and that rule were chosen together.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# CSV column -> the name we use everywhere else.
COLUMNS = {
    "Temperature_C": "temperature",
    "Soil_Humidity_pct": "soil_humidity",
    "Soil_Moisture_pct": "soil_moisture",
    "Air_Humidity_pct": "air_humidity",
    "pH": "ph",
    "Soil_EC_dS_m": "soil_ec",
    "Pressure_kPa": "pressure",
    "Rainfall_mm": "rainfall",
    "Nitrogen_kg_ha": "nitrogen",
    "Phosphorus_kg_ha": "phosphorus",
    "Potassium_kg_ha": "potassium",
}
FEATURES = list(COLUMNS.values())
LABELS = ["Optimal", "Warning", "Critical"]

CLASS_WEIGHTS = {"Optimal": 1, "Warning": 5, "Critical": 30}
CRITICAL_THRESHOLD = 0.15
WARNING_THRESHOLD = 0.25

HERE = Path(__file__).parent
DATA = HERE / "data" / "greenhouse_conditions.csv"
if not DATA.exists():  # running outside Docker
    DATA = HERE.parent.parent / "data" / "greenhouse_conditions.csv"


def apply_thresholds(proba, classes):
    """Turn probabilities into labels using our thresholds instead of argmax.

    Warning is applied first so that Critical can overwrite it, which means a
    reading that passes both thresholds is reported at the higher severity.
    """
    labels = np.array([classes[i] for i in proba.argmax(axis=1)], dtype=object)
    labels[proba[:, classes.index("Warning")] >= WARNING_THRESHOLD] = "Warning"
    labels[proba[:, classes.index("Critical")] >= CRITICAL_THRESHOLD] = "Critical"
    return labels


def main():
    df = pd.read_csv(DATA).rename(columns=COLUMNS)
    print(f"{len(df)} readings loaded")
    print("class balance:", dict(df["Growing_Condition"].value_counts()))

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["Growing_Condition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            # The dataset has missing values and so will a real farm. The imputer is
            # part of the model, so the serving code cannot forget to apply it.
            ("impute", SimpleImputer(strategy="median")),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight=CLASS_WEIGHTS,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    classes = list(model.named_steps["forest"].classes_)
    proba = model.predict_proba(X_test)

    print("\n--- normal prediction, for comparison ---")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    predicted = apply_thresholds(proba, classes)
    print(
        f"--- with our thresholds (Critical >= {CRITICAL_THRESHOLD}, "
        f"Warning >= {WARNING_THRESHOLD}) ---"
    )
    print(classification_report(y_test, predicted, zero_division=0))
    print("confusion matrix (rows = actual):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, predicted, labels=LABELS),
            index=LABELS,
            columns=LABELS,
        )
    )

    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    importance = dict(
        sorted(
            zip(FEATURES, model.named_steps["forest"].feature_importances_.round(4)),
            key=lambda pair: -pair[1],
        )
    )

    # Median of each feature across healthy readings. The serving code uses these
    # to say which sensor is furthest from normal when a zone is not Optimal.
    healthy = df[df["Growing_Condition"] == "Optimal"][FEATURES].median().round(2)

    info = {
        "version": datetime.now(timezone.utc).strftime("rf-%Y%m%d-%H%M"),
        "rows": len(df),
        "features": FEATURES,
        "thresholds": {"Critical": CRITICAL_THRESHOLD, "Warning": WARNING_THRESHOLD},
        "critical_recall": round(report["Critical"]["recall"], 3),
        "critical_precision": round(report["Critical"]["precision"], 3),
        "accuracy": round(report["accuracy"], 3),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "healthy_median": {k: float(v) for k, v in healthy.items()},
    }

    HERE.joinpath("artifacts").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "classes": classes, "info": info},
        HERE / "artifacts" / "model.joblib",
    )
    (HERE / "artifacts" / "model_info.json").write_text(json.dumps(info, indent=2))

    print(f"\nsaved {info['version']}")
    print(
        f"  Critical recall {info['critical_recall']}  "
        f"accuracy {info['accuracy']}  macro F1 {info['macro_f1']}"
    )


if __name__ == "__main__":
    main()
