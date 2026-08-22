"""Flask inference service for plant-condition classification."""

import logging
import math
import os
from pathlib import Path

import joblib  # used to load the trained model bundle that train.py saved to disk
import numpy as np
from flask import Flask, jsonify, request 

logging.basicConfig(level=logging.INFO, format="[inference] %(message)s")  # configure the root logger: show INFO and above, prefix every line with [inference]
log = logging.getLogger(__name__)  # a logger object scoped to this module, used everywhere below instead of print()

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

FEATURES = [
    "temperature",
    "soil_humidity",
    "soil_moisture",
    "air_humidity",
    "ph",
    "soil_ec",
    "pressure",
    "rainfall",
    "nitrogen",
    "phosphorus",
    "potassium",
]

CAUSES = {
    "temperature": {
        "low": "Air conditioning is overcooling the room",
        "high": "Air conditioning fault or lights running too long",
    },
    "soil_humidity": {
        "low": "Substrate is drying out, check the driplines",
        "high": "Drainage is blocked, substrate is waterlogged",
    },
    "soil_moisture": {
        "low": "Irrigation pump may have stopped or a dripline is blocked",
        "high": "Irrigation valve stuck open or drainage clogged",
    },
    "air_humidity": {
        "low": "Ventilation running too hard or humidifier failed",
        "high": "Ventilation fault or extraction fan stopped",
    },
    "ph": {
        "low": "Nutrient solution too acidic, check the dosing pump",
        "high": "Nutrient solution too alkaline, check the dosing pump",
    },
    "soil_ec": {
        "low": "Nutrient concentration has dropped, check the feed mix",
        "high": "Salt build-up in the substrate, flush and check the feed",
    },
    "pressure": {
        "low": "Pressure sensor reading low, check the sensor",
        "high": "Pressure sensor reading high, check the sensor",
    },
    "rainfall": {
        "low": "Less water delivered than scheduled, check pump and timer",
        "high": "More water delivered than scheduled, check for a stuck valve",
    },
    "nitrogen": {
        "low": "Nitrogen depleted, a feed is due",
        "high": "Nitrogen over-applied, hold the next feed",
    },
    "phosphorus": {
        "low": "Phosphorus depleted, a feed is due",
        "high": "Phosphorus over-applied, hold the next feed",
    },
    "potassium": {
        "low": "Potassium depleted, a feed is due",
        "high": "Potassium over-applied, hold the next feed",
    },
    "light": {
        "low": "Grow light failure, or lights off during the day",
        "high": "Lights left on overnight, check the timer",
    },
}

LIGHTS_ON_HOUR, LIGHTS_OFF_HOUR = 6, 22
LIGHT_MIN, LIGHT_MAX = 12000, 20000

app = Flask(__name__)
MODEL = None
CLASSES = None
INFO = None


def load_model():
    global MODEL, CLASSES, INFO
    if MODEL is not None:
        return
    bundle = joblib.load(MODEL_PATH) 
    MODEL, CLASSES, INFO = bundle["model"], bundle["classes"], bundle["info"]
    log.info("loaded model %s", INFO["version"])


def check_lights(light, hour):
    """Return a lighting severity and cause, or ``(None, None)``."""
    if light is None or hour is None:
        return None, None
    if not (LIGHTS_ON_HOUR <= hour < LIGHTS_OFF_HOUR):
        if light > LIGHT_MIN / 2:
            return "Warning", CAUSES["light"]["high"]
        return None, None
    if light < LIGHT_MIN / 2:
        return "Critical", CAUSES["light"]["low"]
    if light < LIGHT_MIN or light > LIGHT_MAX:
        direction = "low" if light < LIGHT_MIN else "high"
        return "Warning", CAUSES["light"][direction]
    return None, None


def find_problem(values):
    """Find the model input furthest from its healthy median."""
    worst_name, worst_gap = None, 0.0
    for name in FEATURES:
        value = values.get(name)
        healthy = INFO["healthy_median"][name]
        if value is None or not healthy:
            continue
        gap = abs(value - healthy) / abs(healthy)
        if gap > worst_gap:
            worst_name, worst_gap = name, gap
    if worst_name is None:
        return None, None
    direction = (
        "high" if values[worst_name] > INFO["healthy_median"][worst_name] else "low"
    )
    return worst_name, CAUSES[worst_name][direction]


def parse_payload(raw):
    if not isinstance(raw, dict):
        raise ValueError("request body must be a JSON object")

    values = {}
    for name in FEATURES + ["light_intensity"]:
        value = raw.get(name)
        if value is None:
            values[name] = None
            continue
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        values[name] = number

    hour = raw.get("hour")
    if hour is None:
        values["hour"] = None
    else:
        if isinstance(hour, bool):
            raise ValueError("hour must be a whole number from 0 to 23")
        try:
            parsed_hour = int(hour)
        except (TypeError, ValueError) as exc:
            raise ValueError("hour must be a whole number from 0 to 23") from exc
        if str(hour).strip() not in {str(parsed_hour), f"{parsed_hour}.0"}:
            raise ValueError("hour must be a whole number from 0 to 23")
        if not 0 <= parsed_hour <= 23:
            raise ValueError("hour must be a whole number from 0 to 23")
        values["hour"] = parsed_hour

    return values


@app.get("/health")
def health():
    return jsonify(service="inference", status="ok")


@app.get("/ready")
def ready():
    if MODEL is None:
        return jsonify(service="inference", status="not ready", model=False), 503
    return jsonify(service="inference", status="ready", model=True)


@app.get("/model")
def model_info():
    if INFO is None:
        return jsonify(detail="model is not loaded"), 503
    return jsonify(INFO)


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify(detail="model is not loaded"), 503
    try:
        values = parse_payload(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify(detail=str(exc)), 422

    row = np.array(
        [
            [
                values.get(name) if values.get(name) is not None else np.nan
                for name in FEATURES
            ]
        ],
        dtype=float,
    )
    probabilities = MODEL.predict_proba(row)[0]
    scores = dict(zip(CLASSES, probabilities.round(3)))

    label = CLASSES[int(probabilities.argmax())]
    if scores.get("Warning", 0) >= INFO["thresholds"]["Warning"]:
        label = "Warning"
    if scores.get("Critical", 0) >= INFO["thresholds"]["Critical"]:
        label = "Critical"

    result = {
        "label": label,
        "confidence": float(max(probabilities)),
        "scores": {name: float(value) for name, value in scores.items()},
        "model_version": INFO["version"],
    }

    if label != "Optimal":
        driver, cause = find_problem(values)
        result["driver"] = driver
        result["probable_cause"] = cause

    rank = {"Optimal": 0, "Warning": 1, "Critical": 2}
    light_label, light_cause = check_lights(
        values.get("light_intensity"), values.get("hour")
    )
    if light_label and rank[light_label] > rank[result["label"]]:
        result.update(
            label=light_label,
            driver="light",
            probable_cause=light_cause,
        )

    return jsonify(result)


try:
    load_model()
except Exception as exc:  # pragma: no cover
    # Without the model this service cannot answer, but it should still start
    # and say so through /ready rather than refusing to boot.
    log.error("could not load the model from %s: %s", MODEL_PATH, exc) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True) # start Flask's built-in server, listening on all interfaces (required inside a container), debug mode off, one thread per request
