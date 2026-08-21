"""Flask sensor simulator used for the Docker and Kubernetes demonstrations."""

import logging
import os
import random
import threading
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, jsonify

logging.basicConfig(level=logging.INFO, format="[simulator] %(message)s")
log = logging.getLogger(__name__)

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")
ZONES = [
    zone.strip()
    for zone in os.getenv("ZONES", "Zone_01,Zone_02,Zone_03,Zone_04").split(",")
    if zone.strip()
]
UTC_OFFSET = float(os.getenv("FARM_UTC_OFFSET_HOURS", "8"))
SECONDS_BETWEEN_READINGS = float(os.getenv("SECONDS_BETWEEN_READINGS", "5"))

NORMAL = {
    "temperature": 27.6,
    "soil_humidity": 57.0,
    "soil_moisture": 44.9,
    "air_humidity": 59.6,
    "ph": 6.62,
    "soil_ec": 0.94,
    "pressure": 101.14,
    "rainfall": 88.1,
}
WOBBLE = {
    "temperature": 1.2,
    "soil_humidity": 3.0,
    "soil_moisture": 4.0,
    "air_humidity": 3.0,
    "ph": 0.12,
    "soil_ec": 0.09,
    "pressure": 0.08,
    "rainfall": 12.0,
}
FAULTS = {
    "pump": ("soil_moisture", -2.6),
    "aircon": ("temperature", 0.8),
    "ventilation": ("air_humidity", 2.4),
    "lighting": ("light", -1500.0),
}

state = {
    zone: {
        "clock": datetime.now(timezone.utc) - timedelta(hours=3),
        "fault": None,
        "steps": 0,
    }
    for zone in ZONES
}
state_lock = threading.Lock()
stop_event = threading.Event()
app = Flask(__name__)


def make_reading(zone_id):
    with state_lock:
        zone = state[zone_id]
        local_hour = (zone["clock"] + timedelta(hours=UTC_OFFSET)).hour

        reading = {
            name: random.gauss(normal, WOBBLE[name]) for name, normal in NORMAL.items()
        }
        reading["light_intensity"] = (
            random.gauss(16000, 700) if 6 <= local_hour < 22 else 40.0
        )

        if zone["fault"]:
            sensor, drift = FAULTS[zone["fault"]]
            key = "light_intensity" if sensor == "light" else sensor
            reading[key] += drift * zone["steps"]
            zone["steps"] += 1

        reading = {name: round(max(0.0, value), 2) for name, value in reading.items()}
        reading["zone_id"] = zone_id
        reading["recorded_at"] = zone["clock"].isoformat()
        zone["clock"] += timedelta(minutes=10)
        return reading


def send_readings():
    """Background loop that starts after the other services have initialised."""
    if stop_event.wait(12):
        return
    while not stop_event.is_set():
        batch = [make_reading(zone) for zone in ZONES]
        try:
            reply = requests.post(
                f"{GATEWAY_URL}/api/readings",
                json={"readings": batch},
                timeout=15,
            )
            reply.raise_for_status()
        except requests.RequestException as exc:
            log.warning("could not send readings: %s", exc)
        stop_event.wait(SECONDS_BETWEEN_READINGS)


def start_sender():
    thread = threading.Thread(target=send_readings, name="sensor-sender", daemon=True)
    thread.start()
    return thread


@app.get("/health")
def health():
    return jsonify(service="simulator", status="ok")


@app.get("/ready")
def ready():
    return jsonify(service="simulator", status="ready", zones=ZONES)


@app.get("/state")
def show_state():
    with state_lock:
        result = {zone: {"fault": details["fault"]} for zone, details in state.items()}
    return jsonify(result)


@app.post("/fault/<zone_id>/<kind>")
def start_fault(zone_id, kind):
    if zone_id not in state:
        return jsonify(detail=f"no zone {zone_id}, have {ZONES}"), 404
    if kind not in FAULTS:
        return jsonify(detail=f"no fault {kind}, have {list(FAULTS)}"), 400
    with state_lock:
        state[zone_id].update(fault=kind, steps=1)
    log.info("fault '%s' started in %s", kind, zone_id)
    return jsonify(zone_id=zone_id, fault=kind)


@app.delete("/fault/<zone_id>")
def clear_fault(zone_id):
    if zone_id not in state:
        return jsonify(detail=f"no zone {zone_id}"), 404
    with state_lock:
        state[zone_id].update(fault=None, steps=0)
    return jsonify(zone_id=zone_id, fault=None)


if __name__ == "__main__":
    start_sender()
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
