"""Fast checks for the Week 16 Flask and Docker implementation."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "services" / "inference" / "artifacts" / "model.joblib"


def load_file(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_package(package_name, relative_directory):
    package_directory = ROOT / relative_directory
    init_file = package_directory / "__init__.py"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_directory)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)

    main_spec = importlib.util.spec_from_file_location(
        f"{package_name}.main", package_directory / "main.py"
    )
    main = importlib.util.module_from_spec(main_spec)
    sys.modules[f"{package_name}.main"] = main
    main_spec.loader.exec_module(main)
    return main


def test_requirements_use_the_expected_week16_stack():
    allowed = {
        "flask",
        "requests",
        "psycopg2-binary",
        "scikit-learn",
        "joblib",
        "numpy",
        "pandas",
    }
    files = list((ROOT / "services").rglob("*requirements.txt"))
    files += list((ROOT / "simulator").rglob("*requirements.txt"))

    for path in files:
        packages = {
            line.split("==", 1)[0].strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert packages <= allowed, f"unexpected package in {path.relative_to(ROOT)}"


def test_dockerfiles_follow_the_taught_python_pattern():
    dockerfiles = list((ROOT / "services").glob("*/Dockerfile"))
    dockerfiles.append(ROOT / "simulator" / "Dockerfile")

    for path in dockerfiles:
        contents = path.read_text(encoding="utf-8")
        assert "FROM python:" in contents
        assert "WORKDIR " in contents
        assert "COPY " in contents
        assert "RUN pip install" in contents
        assert "EXPOSE 8000" in contents
        assert 'CMD ["python"' in contents


def test_gateway_is_a_working_flask_app():
    gateway = load_file("farm_gateway", "services/gateway/app/main.py")
    reply = gateway.app.test_client().get("/health")
    assert reply.status_code == 200
    assert reply.get_json() == {"service": "gateway", "status": "ok"}
    assert gateway.find_route("/api/predict")[1] == "/predict"


@pytest.mark.skipif(
    not MODEL_FILE.exists(), reason="run services/inference/train.py first"
)
def test_inference_is_a_working_flask_app():
    inference = load_file("farm_inference", "services/inference/app/main.py")
    inference.load_model()
    client = inference.app.test_client()

    assert client.get("/ready").get_json()["model"] is True
    reply = client.post(
        "/predict",
        json={
            "temperature": 27.6,
            "soil_humidity": 57.0,
            "soil_moisture": 44.9,
            "air_humidity": 59.6,
            "ph": 6.62,
            "soil_ec": 0.94,
            "pressure": 101.14,
            "rainfall": 88.1,
            "nitrogen": 54.3,
            "phosphorus": 38.3,
            "potassium": 52.0,
            "light_intensity": 16000,
            "hour": 12,
        },
    )
    assert reply.status_code == 200
    assert reply.get_json()["label"] in ("Optimal", "Warning", "Critical")


def test_simulator_fault_routes_work_in_flask():
    simulator = load_file("farm_simulator", "simulator/simulate.py")
    client = simulator.app.test_client()

    assert client.get("/health").status_code == 200
    assert client.post("/fault/Zone_01/pump").get_json()["fault"] == "pump"
    assert client.delete("/fault/Zone_01").get_json()["fault"] is None


def test_ingestion_flask_validation_and_readiness():
    ingestion = load_package("farm_ingestion", "services/ingestion/app")
    ingestion.db.is_healthy = lambda: True
    client = ingestion.app.test_client()

    assert client.get("/ready").status_code == 200
    reply = client.post(
        "/readings",
        json={
            "readings": [
                {
                    "zone_id": "Zone_01",
                    "recorded_at": "2026-08-21T00:00:00Z",
                    "soil_moisture": 999,
                }
            ]
        },
    )
    assert reply.status_code == 422
    assert "soil_moisture" in reply.get_json()["detail"]


def test_dashboard_is_a_working_flask_app():
    dashboard = load_package("farm_dashboard", "services/dashboard/app")
    dashboard.db.is_healthy = lambda: True
    client = dashboard.app.test_client()

    assert client.get("/ready").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/config.json").get_json()["gatewayUrl"].startswith("http")
    assert (
        client.post(
            "/evaluate", json={"zone_id": "Zone_01", "label": "Unknown"}
        ).status_code
        == 422
    )
