"""Flask API gateway for the smart-farm services.

The dashboard, simulator and any future sensor device use this single entry
point. Docker Compose and Kubernetes provide service-name DNS, so the gateway
can reach ``ingestion``, ``inference`` and ``dashboard`` without fixed IP
addresses.
"""

import logging
import os

import requests as http_requests
from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO, format="[gateway] %(message)s")
log = logging.getLogger(__name__)

SERVICES = {
    "ingestion": os.getenv("INGESTION_URL", "http://ingestion:8000"),
    "inference": os.getenv("INFERENCE_URL", "http://inference:8000"),
    "dashboard": os.getenv("DASHBOARD_URL", "http://dashboard:8000"),
}

# Public path -> (service, internal path). Longest matches are checked first so
# /api/incidents/3 is sent to the same service as /api/incidents.
ROUTES = [
    ("/api/readings", "ingestion", "/readings"),
    ("/api/zones", "ingestion", "/zones"),
    ("/api/stats", "ingestion", "/stats"),
    ("/api/predict", "inference", "/predict"),
    ("/api/model", "inference", "/model"),
    ("/api/evaluate", "dashboard", "/evaluate"),
    ("/api/incidents", "dashboard", "/incidents"),
]

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    """Allow the dashboard on port 3000 to call the gateway on port 8080."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def find_route(path):
    for public, service, target in sorted(ROUTES, key=lambda item: -len(item[0])):
        if path == public or path.startswith(public + "/"):
            return SERVICES[service], target + path[len(public) :]
    return None


@app.get("/health")
def health():
    return jsonify(service="gateway", status="ok")


@app.get("/health/all")
def health_all():
    """Verify that each application service is ready and reachable."""
    services = {"gateway": "up"}
    for name, url in SERVICES.items():
        try:
            reply = http_requests.get(f"{url}/ready", timeout=3)
            reply.raise_for_status()
            services[name] = "up"
        except http_requests.RequestException:
            services[name] = "down"

    overall = "ok" if all(value == "up" for value in services.values()) else "degraded"
    return jsonify(status=overall, services=services)


@app.route("/api/<path:path>", methods=["GET", "POST", "DELETE", "OPTIONS"])
def forward(path):
    if request.method == "OPTIONS":
        return "", 204

    route = find_route(request.path)
    if route is None:
        return jsonify(detail="unknown path"), 404

    url, target_path = route
    headers = {"Content-Type": request.content_type or "application/json"}

    try:
        reply = http_requests.request(
            request.method,
            url + target_path,
            data=request.get_data(),
            params=list(request.args.items(multi=True)),
            headers=headers,
            timeout=15,
        )
    except http_requests.Timeout:
        return jsonify(detail="service timed out"), 504
    except http_requests.RequestException:
        return jsonify(detail="service unavailable"), 503

    log.info("%s %s -> %d", request.method, request.path, reply.status_code)
    return Response(
        reply.content,
        status=reply.status_code,
        content_type=reply.headers.get("Content-Type", "application/json"),
    )


if __name__ == "__main__":
    log.info("ready with %d routes", len(ROUTES))
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
