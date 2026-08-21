"""Flask ingestion service.

Readings are validated and stored before classification. If inference is
temporarily unavailable, the raw reading is still retained with a ``pending``
prediction so no sensor data is lost.
"""

import atexit
import logging
import threading
import time
import math
import os
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, jsonify, request
from psycopg2.extras import RealDictCursor

from . import db

logging.basicConfig(level=logging.INFO, format="[ingestion] %(message)s")
log = logging.getLogger(__name__)

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")
UTC_OFFSET_HOURS = float(os.getenv("FARM_UTC_OFFSET_HOURS", "8"))

SENSOR_RANGES = {
    "temperature": (-50, 100),
    "soil_humidity": (0, 100),
    "soil_moisture": (0, 100),
    "air_humidity": (0, 100),
    "ph": (0, 14),
    "soil_ec": (0, 20),
    "pressure": (50, 150),
    "rainfall": (0, 2000),
    "light_intensity": (0, 100000),
}
MODEL_SENSORS = [
    "temperature",
    "soil_humidity",
    "soil_moisture",
    "air_humidity",
    "ph",
    "soil_ec",
    "pressure",
    "rainfall",
]

zones = {}
app = Flask(__name__)


class ValidationError(ValueError):
    pass


def parse_datetime(value):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("recorded_at must be an ISO date and time")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("recorded_at must be an ISO date and time") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_reading(raw):
    if not isinstance(raw, dict):
        raise ValidationError("each reading must be a JSON object")

    zone_id = raw.get("zone_id")
    if not isinstance(zone_id, str) or not 1 <= len(zone_id.strip()) <= 32:
        raise ValidationError("zone_id must contain 1 to 32 characters")

    reading = {
        "zone_id": zone_id.strip(),
        "recorded_at": parse_datetime(raw.get("recorded_at")),
    }

    for name, (minimum, maximum) in SENSOR_RANGES.items():
        value = raw.get(name)
        if value is None:
            reading[name] = None
            continue
        if isinstance(value, bool):
            raise ValidationError(f"{name} must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} must be a number") from exc
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")
        reading[name] = number

    return reading


def parse_limit(value, default, maximum):
    try:
        limit = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise ValidationError("limit must be a whole number") from exc
    if not 1 <= limit <= maximum:
        raise ValidationError(f"limit must be between 1 and {maximum}")
    return limit


def serialise(row):
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in dict(row).items()
    }


def load_zones():
    """Cache the soil-test data that changes only occasionally."""
    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM zones")
            rows = cursor.fetchall()
    zones.clear()
    zones.update({row["zone_id"]: dict(row) for row in rows})


@app.errorhandler(db.NotReady)
def handle_not_ready(exc):
    """The pool opens in the background, so a request can arrive before it is
    ready. Say so plainly instead of failing with an internal error."""
    return jsonify(service="ingestion", detail="database unavailable"), 503


@app.get("/health")
def health():
    return jsonify(service="ingestion", status="ok")


@app.get("/ready")
def ready():
    if not db.is_healthy():
        return jsonify(detail="database unavailable"), 503
    return jsonify(service="ingestion", status="ready", zones=len(zones))


def save(reading):
    """Insert one reading and return ``(id, was_duplicate)``."""
    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """INSERT INTO readings (zone_id, recorded_at, temperature,
                       soil_humidity, soil_moisture, air_humidity, ph, soil_ec,
                       pressure, rainfall, light_intensity)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (zone_id, recorded_at) DO NOTHING RETURNING id""",
                (
                    reading["zone_id"],
                    reading["recorded_at"],
                    reading["temperature"],
                    reading["soil_humidity"],
                    reading["soil_moisture"],
                    reading["air_humidity"],
                    reading["ph"],
                    reading["soil_ec"],
                    reading["pressure"],
                    reading["rainfall"],
                    reading["light_intensity"],
                ),
            )
            row = cursor.fetchone()
            if row:
                return row["id"], False

            cursor.execute(
                "SELECT id FROM readings WHERE zone_id=%s AND recorded_at=%s",
                (reading["zone_id"], reading["recorded_at"]),
            )
            return cursor.fetchone()["id"], True


def classify(reading):
    """Ask inference through the gateway, keeping the services decoupled."""
    zone = zones.get(reading["zone_id"], {})
    payload = {name: reading.get(name) for name in MODEL_SENSORS}
    payload.update(
        nitrogen=zone.get("nitrogen"),
        phosphorus=zone.get("phosphorus"),
        potassium=zone.get("potassium"),
        light_intensity=reading.get("light_intensity"),
        hour=(reading["recorded_at"] + timedelta(hours=UTC_OFFSET_HOURS)).hour,
    )
    try:
        reply = requests.post(f"{GATEWAY_URL}/api/predict", json=payload, timeout=10)
        reply.raise_for_status()
        return reply.json()
    except requests.RequestException as exc:
        log.warning("could not classify %s: %s", reading["zone_id"], exc)
        return {"label": "pending"}


def store_prediction(reading_id, zone_id, result):
    with db.connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            """INSERT INTO predictions (reading_id, zone_id, label,
                       confidence, driver, probable_cause)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (reading_id) DO UPDATE SET
                       label=EXCLUDED.label,
                       confidence=EXCLUDED.confidence,
                       driver=EXCLUDED.driver,
                       probable_cause=EXCLUDED.probable_cause""",
            (
                reading_id,
                zone_id,
                result["label"],
                result.get("confidence"),
                result.get("driver"),
                result.get("probable_cause"),
            ),
        )


def tell_dashboard(reading, result):
    try:
        reply = requests.post(
            f"{GATEWAY_URL}/api/evaluate",
            json={
                "zone_id": reading["zone_id"],
                "label": result["label"],
                "probable_cause": result.get("probable_cause"),
            },
            timeout=10,
        )
        reply.raise_for_status()
    except requests.RequestException as exc:
        log.warning("could not reach dashboard: %s", exc)


@app.post("/readings")
def receive():
    body = request.get_json(silent=True)
    raw_readings = body.get("readings") if isinstance(body, dict) else None
    if not isinstance(raw_readings, list) or not 1 <= len(raw_readings) <= 500:
        return jsonify(detail="readings must contain between 1 and 500 items"), 422

    try:
        readings = [parse_reading(item) for item in raw_readings]
    except ValidationError as exc:
        return jsonify(detail=str(exc)), 422

    results = []
    for reading in readings:
        reading_id, duplicate = save(reading)
        if duplicate:
            results.append(
                {
                    "zone_id": reading["zone_id"],
                    "label": "duplicate",
                    "duplicate": True,
                }
            )
            continue

        result = classify(reading)
        store_prediction(reading_id, reading["zone_id"], result)
        if result["label"] != "pending":
            tell_dashboard(reading, result)

        results.append(
            {
                "zone_id": reading["zone_id"],
                "label": result["label"],
                "confidence": result.get("confidence"),
                "probable_cause": result.get("probable_cause"),
                "duplicate": False,
            }
        )

    return jsonify(accepted=len(results), results=results), 201


@app.get("/zones")
def zone_status():
    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT DISTINCT ON (r.zone_id)
                          r.zone_id, r.recorded_at, r.temperature,
                          r.soil_humidity, r.soil_moisture, r.air_humidity,
                          r.ph, r.soil_ec, r.rainfall, r.light_intensity,
                          COALESCE(p.label,'pending') AS label,
                          p.driver, p.probable_cause,
                          z.soil_type, z.nitrogen, z.phosphorus, z.potassium
                   FROM readings r
                   LEFT JOIN predictions p ON p.reading_id = r.id
                   LEFT JOIN zones z ON z.zone_id = r.zone_id
                   ORDER BY r.zone_id, r.recorded_at DESC"""
            )
            rows = cursor.fetchall()
    return jsonify(zones=[serialise(row) for row in rows])


@app.get("/readings")
def history():
    zone_id = request.args.get("zone_id", "").strip()
    if not zone_id:
        return jsonify(detail="zone_id is required"), 422
    try:
        limit = parse_limit(request.args.get("limit"), 60, 500)
    except ValidationError as exc:
        return jsonify(detail=str(exc)), 422

    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT r.recorded_at, r.temperature, r.soil_moisture,
                          COALESCE(p.label,'pending') AS label
                   FROM readings r
                   LEFT JOIN predictions p ON p.reading_id = r.id
                   WHERE r.zone_id = %s
                   ORDER BY r.recorded_at DESC LIMIT %s""",
                (zone_id, limit),
            )
            rows = cursor.fetchall()
    return jsonify(zone_id=zone_id, readings=[serialise(row) for row in rows])


@app.get("/stats")
def stats():
    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT (SELECT COUNT(*) FROM readings) AS readings,
                          (SELECT COUNT(*) FROM predictions
                           WHERE label='pending') AS pending"""
            )
            row = cursor.fetchone()
    return jsonify(dict(row))


def load_zones_when_ready():
    """Zone records live in the database, so this waits for the pool instead of
    stopping the service from starting."""
    while not db.ready():
        time.sleep(1)
    try:
        load_zones()
        log.info("ready, %d zones configured", len(zones))
    except Exception as exc:
        log.warning("could not load zones: %s", exc)


def initialise():
    db.connect()
    threading.Thread(target=load_zones_when_ready, daemon=True).start()
    atexit.register(db.close)
    log.info("started; /ready stays 503 until the database is reachable")


initialise()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
