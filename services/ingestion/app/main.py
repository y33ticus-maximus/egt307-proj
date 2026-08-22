"""Flask ingestion service.

Readings are validated and stored before classification. If inference is
temporarily unavailable, the raw reading is still retained with a ``pending``
prediction so no sensor data is lost.
""" 

import atexit  # lets us register a function to run automatically when the process shuts down
import logging  # standard library logging, used instead of print() so output can be filtered/formatted
import threading  # used to run zone-loading in the background without blocking startup
import time  
import math 
import os 
from datetime import datetime, timedelta, timezone  

import requests  # HTTP client library, used to call the gateway from this service
from flask import Flask, jsonify, request  # the web framework and its request/response helpers
from psycopg2.extras import RealDictCursor  # a cursor type that returns rows as dicts (column name -> value) instead of plain tuples

from . import db  # our own db.py module in this package: connection pool, NotReady exception, is_healthy(), etc.

logging.basicConfig(level=logging.INFO, format="[ingestion] %(message)s")  # configure the root logger: show INFO and above, prefix every line with [ingestion]
log = logging.getLogger(__name__)  # a logger object scoped to this module, used everywhere below instead of print()

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")  # internal address of the gateway; defaults to the Compose/Kubernetes service name if not set
UTC_OFFSET_HOURS = float(os.getenv("FARM_UTC_OFFSET_HOURS", "8"))  # hours to add to a UTC timestamp to get the farm's local time; defaults to Singapore's +8

SENSOR_RANGES = {  # start of a dict mapping each sensor name to its (minimum, maximum) physically possible values
    "temperature": (-50, 100),  
    "soil_humidity": (0, 100),  
    "soil_moisture": (0, 100),  
    "air_humidity": (0, 100),  
    "ph": (0, 14), 
    "soil_ec": (0, 20),  
    "pressure": (50, 150),  
    "rainfall": (0, 2000),  
    "light_intensity": (0, 100000),  
}  # end of SENSOR_RANGES dict
MODEL_SENSORS = [  # start of the list of sensor names the machine learning model actually consumes
    "temperature", 
    "soil_humidity",  
    "soil_moisture",  
    "air_humidity", 
    "ph", 
    "soil_ec", 
    "pressure", 
    "rainfall", 
]  # end of MODEL_SENSORS list; note light_intensity is deliberately excluded, since it is handled by a separate rule, not the model

zones = {}  # in-memory cache of zone_id -> zone row (soil type, nutrient levels); populated by load_zones()
app = Flask(__name__)  # create the Flask application; __name__ tells Flask where this module lives on disk


class ValidationError(ValueError):  # a custom exception type, so validation failures can be told apart from other ValueErrors
    pass  # no extra behaviour needed; it exists purely to be a distinct, catchable type


def parse_datetime(value):  # convert a raw JSON value into a proper timezone-aware datetime, or raise ValidationError
    if not isinstance(value, str) or not value.strip():  # reject anything that isn't a non-empty string
        raise ValidationError("recorded_at must be an ISO date and time")  # clear error message naming the field
    try: 
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:  
        raise ValidationError("recorded_at must be an ISO date and time") from exc  
    if parsed.tzinfo is None:  
        parsed = parsed.replace(tzinfo=timezone.utc)  
    return parsed  


def parse_reading(raw):  # validate one incoming reading dict and return a cleaned version, or raise ValidationError
    if not isinstance(raw, dict):  # each reading must itself be a JSON object
        raise ValidationError("each reading must be a JSON object")  # reject lists, strings, numbers, etc.

    zone_id = raw.get("zone_id")  # pull the zone_id field out, or None if missing
    if not isinstance(zone_id, str) or not 1 <= len(zone_id.strip()) <= 32:  # must be a string between 1 and 32 characters after trimming
        raise ValidationError("zone_id must contain 1 to 32 characters")  # reject missing, empty, or overly long zone ids

    reading = {  # start building the cleaned reading dict
        "zone_id": zone_id.strip(),  # store the trimmed zone id
        "recorded_at": parse_datetime(raw.get("recorded_at")),  # parse and validate the timestamp using the helper above
    } 

    for name, (minimum, maximum) in SENSOR_RANGES.items():  # iterate every known sensor and its valid range
        value = raw.get(name)
        if value is None:  
            reading[name] = None 
            continue 
        if isinstance(value, bool):  # Python bools are technically numbers (True == 1), so this must be checked explicitly
            raise ValidationError(f"{name} must be a number") 
        try: 
            number = float(value)  
        except (TypeError, ValueError) as exc:  # raised if the value can't be interpreted as a number at all, e.g. a string like "abc"
            raise ValidationError(f"{name} must be a number") from exc  # reject with a clear message, keeping the original error as context
        if not math.isfinite(number) or not minimum <= number <= maximum: 
            raise ValidationError(f"{name} must be between {minimum} and {maximum}")  # tell the caller exactly what range was expected
        reading[name] = number  

    return reading 


def parse_limit(value, default, maximum):  # validate a "how many rows" query parameter, applying a default and a ceiling
    try: 
        limit = int(value if value is not None else default)  # use the supplied value if present, otherwise fall back to default
    except (TypeError, ValueError) as exc:  
        raise ValidationError("limit must be a whole number") from exc  
    if not 1 <= limit <= maximum:  # enforce a sensible range so nobody can request zero or an enormous number of rows
        raise ValidationError(f"limit must be between 1 and {maximum}") 
    return limit 


def serialise(row):  # convert one database row into something jsonify() can safely send as JSON
    return { 
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in dict(row).items() 
    } 


def load_zones():  # refresh the in-memory zones cache from the database
    """Cache the soil-test data that changes only occasionally."""  # docstring explaining why this is cached rather than queried every time
    with db.connection() as conn:  # borrow a database connection from the pool; automatically returned when this block exits
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:  # open a cursor on that connection that returns dict-like rows
            cursor.execute("SELECT * FROM zones")  # fetch every column for every zone
            rows = cursor.fetchall()  # pull all matching rows into a Python list
    zones.clear()  # empty the existing in-memory cache before repopulating it
    zones.update({row["zone_id"]: dict(row) for row in rows})  # rebuild the cache as zone_id -> full row dict, for O(1) lookup by zone


@app.errorhandler(db.NotReady)  # register this function to run automatically whenever code anywhere raises db.NotReady
def handle_not_ready(exc):  
    """The pool opens in the background, so a request can arrive before it is
    ready. Say so plainly instead of failing with an internal error.""" 
    return jsonify(service="ingestion", detail="database unavailable"), 503  


@app.get("/health")  # register a route: GET requests to /health call the function below
def health():  # liveness check — used by Kubernetes to decide whether to restart the container
    return jsonify(service="ingestion", status="ok")  # always returns 200 OK if the process is alive enough to answer at all


@app.get("/ready")  # register a route: GET requests to /ready call the function below
def ready():  # readiness check — used by Kubernetes to decide whether to send this pod traffic
    if not db.is_healthy():  # ask the db module to actually try a query against the database
        return jsonify(detail="database unavailable"), 503  # if the database isn't reachable, say so with a 503 rather than pretending to be ready
    return jsonify(service="ingestion", status="ready", zones=len(zones))  # otherwise confirm readiness and report how many zones are cached


def save(reading):  # insert one validated reading into the database
    """Insert one reading and return ``(id, was_duplicate)``."""  # docstring explaining the return value's shape
    with db.connection() as conn:  # borrow a connection from the pool
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:  # open a dict-returning cursor
            cursor.execute(  # run a parameterised insert (safe against SQL injection)
                """INSERT INTO readings (zone_id, recorded_at, temperature,
                       soil_humidity, soil_moisture, air_humidity, ph, soil_ec,
                       pressure, rainfall, light_intensity)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (zone_id, recorded_at) DO NOTHING RETURNING id""",  # insert the row; if this exact zone/time already exists, do nothing instead of erroring or duplicating
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
            row = cursor.fetchone()  # RETURNING id gives back the new row's id, or nothing if ON CONFLICT DO NOTHING fired
            if row:  # if a row came back, the insert succeeded and this is a brand new reading
                return row["id"], False  # return its id and False for "was this a duplicate"

            cursor.execute(  # the insert was skipped because this reading already existed, so look up its existing id
                "SELECT id FROM readings WHERE zone_id=%s AND recorded_at=%s",  # find the pre-existing row by its unique key
                (reading["zone_id"], reading["recorded_at"]),  # substitute the zone and timestamp
            )  # end of execute() call
            return cursor.fetchone()["id"], True  # return the existing id and True for "was this a duplicate"


def classify(reading):  # send a reading to the inference service (via the gateway) and get back a verdict
    """Ask inference through the gateway, keeping the services decoupled."""  # docstring explaining why we go through the gateway rather than calling inference directly
    zone = zones.get(reading["zone_id"], {})  # look up this zone's cached soil data; {} if the zone isn't in the cache for some reason
    payload = {name: reading.get(name) for name in MODEL_SENSORS}  # build the payload from just the sensor fields the model expects
    payload.update(  # add the extra fields the model/rules need beyond the raw sensors
        nitrogen=zone.get("nitrogen"),  # from the zone's soil test, not from a sensor
        phosphorus=zone.get("phosphorus"),  # from the zone's soil test
        potassium=zone.get("potassium"),  # from the zone's soil test
        light_intensity=reading.get("light_intensity"),  # passed through for the separate lighting rule, not the model itself
        hour=(reading["recorded_at"] + timedelta(hours=UTC_OFFSET_HOURS)).hour,  # convert the UTC timestamp to local time and extract just the hour, for the lighting rule
    )  # end of payload.update() call
    try:  # attempt to reach the inference service through the gateway
        reply = requests.post(f"{GATEWAY_URL}/api/predict", json=payload, timeout=10)  # POST the payload, waiting at most 10 seconds
        reply.raise_for_status()  # raise an exception if the response status code indicates an error (4xx/5xx)
        return reply.json()  # parse and return the JSON body, e.g. {"label": "Warning", ...}
    except requests.RequestException as exc:  # catches any networking problem: timeout, connection refused, bad status, etc.
        log.warning("could not classify %s: %s", reading["zone_id"], exc)  # log the failure without crashing
        return {"label": "pending"}  # fall back to a sentinel label so the caller can still proceed


def store_prediction(reading_id, zone_id, result):  # save the model's verdict for a given reading
    with db.connection() as conn, conn.cursor() as cursor:  # borrow a connection and a plain (non-dict) cursor together
        cursor.execute(  # run an insert-or-update
            """INSERT INTO predictions (reading_id, zone_id, label,
                       confidence, driver, probable_cause)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (reading_id) DO UPDATE SET
                       label=EXCLUDED.label,
                       confidence=EXCLUDED.confidence,
                       driver=EXCLUDED.driver,
                       probable_cause=EXCLUDED.probable_cause""",  # insert a new prediction, or if one already exists for this reading, overwrite it with the new values
            (
                reading_id, 
                zone_id,  
                result["label"],  
                result.get("confidence"), 
                result.get("driver"),  
                result.get("probable_cause"), 
            ),  
        ) 


def tell_dashboard(reading, result):  # notify the dashboard service of a classified reading, via the gateway
    try: 
        reply = requests.post(  # send a POST request
            f"{GATEWAY_URL}/api/evaluate",  # the gateway route that forwards to the dashboard's /evaluate endpoint
            json={  # the JSON body being sent
                "zone_id": reading["zone_id"],  # which zone this concerns
                "label": result["label"],  # the severity the model/rules decided on
                "probable_cause": result.get("probable_cause"),  # the likely faulty sensor's explanation, if any
            },  
            timeout=10, 
        )  
        reply.raise_for_status()  # raise an exception if the dashboard responded with an error status
    except requests.RequestException as exc:  # catches any networking problem reaching the dashboard
        log.warning("could not reach dashboard: %s", exc)  # log it and continue; the reading and prediction are already saved regardless


@app.post("/readings")  # register a route: POST requests to /readings call the function below
def receive():  # the main entry point: accepts a batch of readings, validates, stores, classifies and forwards them
    body = request.get_json(silent=True)  # parse the request body as JSON; silent=True returns None instead of raising on bad JSON
    raw_readings = body.get("readings") if isinstance(body, dict) else None  # pull out the "readings" list if body is a proper JSON object, else None
    if not isinstance(raw_readings, list) or not 1 <= len(raw_readings) <= 500:  # must be a list with between 1 and 500 items
        return jsonify(detail="readings must contain between 1 and 500 items"), 422  # reject with a clear error otherwise

    try:  
        readings = [parse_reading(item) for item in raw_readings]  # run each raw item through parse_reading(), collecting the cleaned results
    except ValidationError as exc:  # if ANY reading in the batch fails validation
        return jsonify(detail=str(exc)), 422  # reject the whole batch with the specific validation error message

    results = []  # will hold one summary dict per reading, to return to the caller
    for reading in readings:  # process each validated reading in turn
        reading_id, duplicate = save(reading)  # store it in the database; find out if it was already there
        if duplicate:  # if this exact reading already existed
            results.append(  # record a simple "duplicate" result and skip classification entirely
                {
                    "zone_id": reading["zone_id"],  # which zone this was
                    "label": "duplicate",  # sentinel label indicating nothing new happened
                    "duplicate": True,  # explicit flag for the caller
                }
            ) 
            continue  # move on to the next reading without calling the model or the dashboard

        result = classify(reading)  # ask the inference service (via the gateway) for a verdict
        store_prediction(reading_id, reading["zone_id"], result)  # save that verdict against this reading
        if result["label"] != "pending":  # only bother the dashboard if we got a real verdict, not a fallback
            tell_dashboard(reading, result)  # let the dashboard decide whether this verdict should raise/update an alert

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


@app.get("/zones")  # register a route: GET requests to /zones call the function below
def zone_status():  # returns the latest reading and verdict for every zone, for the dashboard's zone grid
    with db.connection() as conn:  # borrow a connection from the pool
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:  # open a dict-returning cursor
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
                   ORDER BY r.zone_id, r.recorded_at DESC"""  # DISTINCT ON (r.zone_id) with this ordering picks exactly the newest reading per zone; joins bring in its prediction and zone details; COALESCE substitutes 'pending' if no prediction exists yet
            )  # end of execute() call
            rows = cursor.fetchall()  # pull all matching rows (one per zone) into a Python list
    return jsonify(zones=[serialise(row) for row in rows])  # convert each row to JSON-safe form and wrap in a top-level "zones" key


@app.get("/readings")  # register a route: GET requests to /readings call the function below
def history():  # returns recent readings for one zone, for the dashboard's trend chart
    zone_id = request.args.get("zone_id", "").strip()  # read the required ?zone_id= query parameter, defaulting to "" if absent
    if not zone_id:  # if it was missing or blank after trimming
        return jsonify(detail="zone_id is required"), 422  # reject with a clear error
    try:  # attempt to validate the optional ?limit= parameter
        limit = parse_limit(request.args.get("limit"), 60, 500)  # default to 60 rows, capped at a maximum of 500
    except ValidationError as exc:  # raised if the supplied limit isn't valid
        return jsonify(detail=str(exc)), 422 

    with db.connection() as conn:  # borrow a connection from the pool
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:  # open a dict-returning cursor
            cursor.execute(  
                """SELECT r.recorded_at, r.temperature, r.soil_moisture,
                          COALESCE(p.label,'pending') AS label
                   FROM readings r
                   LEFT JOIN predictions p ON p.reading_id = r.id
                   WHERE r.zone_id = %s
                   ORDER BY r.recorded_at DESC LIMIT %s""",  # fetch this zone's readings, newest first, capped at limit, with their verdicts if any
                (zone_id, limit),  # substitute the zone id and the limit
            ) 
            rows = cursor.fetchall()  # pull all matching rows into a Python list
    return jsonify(zone_id=zone_id, readings=[serialise(row) for row in rows])  # respond with the zone id and its serialised reading history


@app.get("/stats")  # register a route: GET requests to /stats call the function below
def stats():  # returns simple counters shown at the top of the dashboard
    with db.connection() as conn:  # borrow a connection from the pool
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:  # open a dict-returning cursor
            cursor.execute(  # run a query with two subqueries
                """SELECT (SELECT COUNT(*) FROM readings) AS readings,
                          (SELECT COUNT(*) FROM predictions
                           WHERE label='pending') AS pending"""  # total readings ever stored, and how many predictions are still stuck on 'pending'
            )  # end of execute() call
            row = cursor.fetchone()  # there is exactly one result row from this query
    return jsonify(dict(row))  # convert that single row to a plain dict and return it as JSON


def load_zones_when_ready():  # runs on a background thread; waits for the database before loading zones
    """Zone records live in the database, so this waits for the pool instead of
    stopping the service from starting."""  
    while not db.ready(): 
        time.sleep(1) 
    try:  # once the pool is ready, attempt to load the zones
        load_zones()  # populate the in-memory zones cache from the database
        log.info("ready, %d zones configured", len(zones))  # log success, including how many zones were found
    except Exception as exc:  # catch anything that goes wrong during the load
        log.warning("could not load zones: %s", exc)  # log it rather than crashing the background thread


def initialise():  # groups together everything that needs to happen once, when the service starts
    db.connect()  # open the database connection pool
    threading.Thread(target=load_zones_when_ready, daemon=True).start()  # start a second background thread that waits for the pool, then loads the zones cache
    atexit.register(db.close)  
    log.info("started; /ready stays 503 until the database is reachable") 


initialise()  # actually call initialise() as soon as this module is imported, so startup happens regardless of how the app is launched

if __name__ == "__main__":  # true only when this file is run directly (e.g. `python -m app.main`), false when imported by something else
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True) 
