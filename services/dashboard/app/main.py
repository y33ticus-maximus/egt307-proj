#Flask dashboard and incident-management service.

import atexit #register a function to run when the process shuts down
import logging ## standard library logging, used instead of print() so output can be filtered/formatted
import os
from datetime import datetime
from pathlib import Path  # used to build a filesystem path to the static/ folder in an OS-independent way

from flask import Flask, jsonify, request, send_from_directory  # the web framework and its request/response helpers
from psycopg2.extras import RealDictCursor # a cursor type that returns rows as dicts (column name -> value) instead of plain tuples

from . import db

logging.basicConfig(level=logging.INFO, format="[dashboard] %(message)s")
log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent.parent / "static"
GATEWAY_PUBLIC_URL = os.getenv("GATEWAY_PUBLIC_URL", "http://localhost:8080")

OPEN_AFTER = int(os.getenv("OPEN_AFTER", "2")) # how many consecutive bad readings are needed before an incident is opened
CLOSE_AFTER = int(os.getenv("CLOSE_AFTER", "3")) # how many consecutive Optimal readings are needed before an incident is resolved
SEVERITY = {"Optimal": 0, "Warning": 1, "Critical": 2}

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static") # create the Flask app; __name__ tells Flask where this module lives; static_folder/static_url_path wire up the STATIC path above to the /static URL prefix

 # define a helper that turns one database row into something jsonify() can send
def serialise(row):
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in dict(row).items()
    }

 # define a helper: fetch the most recent real (non-pending) labels for one zone
def recent_labels(zone_id, limit):
    """Return the newest non-pending labels for a zone."""
    with db.connection() as conn:  # borrow a database connection from the pool; automatically returned when this block exits
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:  #open a cursor on that connection that returns dict-like rows
            cursor.execute(
                """SELECT p.label FROM predictions p
                   JOIN readings r ON r.id = p.reading_id
                   WHERE p.zone_id = %s AND p.label <> 'pending'
                   ORDER BY r.recorded_at DESC LIMIT %s""",
                (zone_id, limit), # the two values substituted for the two %s placeholders above, in order
            )
            rows = cursor.fetchall() # pull every matching row from the cursor into a Python list
    return [row["label"] for row in rows]  # extract just the 'label' field from each row into a plain list of strings


def open_incident(zone_id):
    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT * FROM incidents
                   WHERE zone_id=%s AND status='open'
                   ORDER BY opened_at DESC LIMIT 1""",
                (zone_id,),
            )
            return cursor.fetchone()


@app.errorhandler(db.NotReady)
def handle_not_ready(exc):
    """The pool opens in the background, so a request can arrive before it is
    ready. Say so plainly instead of failing with an internal error."""
    return jsonify(service="dashboard", detail="database unavailable"), 503


@app.get("/health")
def health():
    return jsonify(service="dashboard", status="ok")


@app.get("/ready")
def ready():
    if not db.is_healthy():
        return jsonify(detail="database unavailable"), 503
    return jsonify(service="dashboard", status="ready")


@app.post("/evaluate")
def evaluate():
    """Open, update or resolve an incident after a classified reading."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(detail="request body must be a JSON object"), 422

    zone_id = body.get("zone_id")
    label = body.get("label")
    cause = body.get("probable_cause")
    if not isinstance(zone_id, str) or not zone_id.strip():
        return jsonify(detail="zone_id is required"), 422
    if label not in SEVERITY:
        return jsonify(detail="label must be Optimal, Warning or Critical"), 422
    if cause is not None and not isinstance(cause, str):
        return jsonify(detail="probable_cause must be text or null"), 422
    zone_id = zone_id.strip()

    existing = open_incident(zone_id)

    if label == "Optimal":
        if not existing:
            return jsonify(action="none")
        labels = recent_labels(zone_id, CLOSE_AFTER)
        if len(labels) >= CLOSE_AFTER and all(item == "Optimal" for item in labels):
            with db.connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """UPDATE incidents SET status='resolved', closed_at=NOW()
                           WHERE id=%s""",
                    (existing["id"],),
                )
            log.info("resolved incident %d for %s", existing["id"], zone_id)
            return jsonify(action="resolved")
        return jsonify(action="recovering")

    cause = cause or "Zone is outside its healthy range"

    if not existing:
        labels = recent_labels(zone_id, OPEN_AFTER)
        if len(labels) < OPEN_AFTER or not all(
            item in ("Warning", "Critical") for item in labels
        ):
            return jsonify(action="waiting", seen=labels)

        with db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO incidents (zone_id, severity, probable_cause)
                       VALUES (%s,%s,%s) RETURNING id""",
                (zone_id, label, cause),
            )
            incident_id = cursor.fetchone()[0]
        log.info("opened %s incident %d for %s", label, incident_id, zone_id)
        return jsonify(action="opened", incident_id=incident_id)

    if SEVERITY[label] > SEVERITY[existing["severity"]]:
        with db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                """UPDATE incidents SET severity=%s, probable_cause=%s
                       WHERE id=%s""",
                (label, cause, existing["id"]),
            )
        return jsonify(action="escalated")

    return jsonify(action="ongoing")


@app.get("/incidents")
def incidents():
    status = request.args.get("status")
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        return jsonify(detail="limit must be a whole number"), 422
    if not 1 <= limit <= 200:
        return jsonify(detail="limit must be between 1 and 200"), 422

    with db.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if status:
                cursor.execute(
                    """SELECT * FROM incidents WHERE status=%s
                       ORDER BY opened_at DESC LIMIT %s""",
                    (status, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM incidents ORDER BY opened_at DESC LIMIT %s",
                    (limit,),
                )
            rows = cursor.fetchall()
    return jsonify(incidents=[serialise(row) for row in rows])


@app.get("/config.json")  # register a route: GET requests to /config.json call the function below
def config():
    """Tell the browser which public gateway address to use."""
    return jsonify(gatewayUrl=GATEWAY_PUBLIC_URL)


@app.get("/")  # register a route: GET requests to / (the root path) call the function below
def index():
    return send_from_directory(STATIC, "index.html")


def initialise():
    db.connect()
    atexit.register(db.close)
    log.info(
        "started (alert after %d bad readings, clear after %d good); "
        "/ready stays 503 until the database is reachable",
        OPEN_AFTER,
        CLOSE_AFTER,
    )


initialise()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
