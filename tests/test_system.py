"""Tests for the parts of the system that make decisions.

These run without Docker or a database, in about ten seconds:

    pip install pytest
    python -m pytest tests -q

They cover the two rules that would be slow to check by hand: the light schedule
and the alerting logic. Testing the alert rule here takes a second; testing it on
the running system means watching a dashboard for twenty minutes.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "inference"))

MODEL_FILE = ROOT / "services" / "inference" / "artifacts" / "model.joblib"


# ---------------------------------------------------------------------------
# The light rule
# ---------------------------------------------------------------------------
# The dataset has no light column, so lighting is checked by a rule instead of
# by the model. An early version had no idea what time it was and reported every
# zone as failing all night, because the lights are supposed to be off then.


def light_check(light, hour):
    from app.main import check_lights

    return check_lights(light, hour)


def test_dark_at_night_is_fine():
    assert light_check(40, 2) == (None, None)


def test_dark_during_the_day_is_critical():
    severity, cause = light_check(40, 12)
    assert severity == "Critical"
    assert "light" in cause.lower()


def test_lights_on_at_night_is_a_warning():
    severity, cause = light_check(16000, 2)
    assert severity == "Warning"
    assert "overnight" in cause.lower()


def test_normal_daytime_light_is_fine():
    assert light_check(16000, 12) == (None, None)


def test_dim_light_during_the_day_is_a_warning():
    assert light_check(9000, 12)[0] == "Warning"


def test_no_light_reading_is_not_a_fault():
    assert light_check(None, 12) == (None, None)
    assert light_check(16000, None) == (None, None)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
needs_model = pytest.mark.skipif(
    not MODEL_FILE.exists(), reason="run services/inference/train.py first"
)


@needs_model
def test_the_model_was_tuned_for_recall():
    """The whole point of the tuning is Critical recall. If someone changes the
    thresholds and recall drops, this test says so."""
    import joblib

    info = joblib.load(MODEL_FILE)["info"]
    assert info["thresholds"]["Critical"] == 0.15
    assert info["critical_recall"] > 0.85


@needs_model
def test_a_healthy_reading_is_optimal():
    import joblib
    import numpy as np

    bundle = joblib.load(MODEL_FILE)
    healthy = bundle["info"]["healthy_median"]
    row = np.array([[healthy[f] for f in bundle["info"]["features"]]])
    assert bundle["model"].predict(row)[0] == "Optimal"


@needs_model
def test_a_missing_sensor_does_not_break_the_model():
    """Readings arrive with gaps. The imputer inside the model fills them in."""
    import joblib
    import numpy as np

    bundle = joblib.load(MODEL_FILE)
    healthy = bundle["info"]["healthy_median"]
    row = np.array([[healthy[f] for f in bundle["info"]["features"]]])
    row[0][2] = np.nan
    assert bundle["model"].predict(row)[0] in ("Optimal", "Warning", "Critical")


# ---------------------------------------------------------------------------
# The alerting rule
# ---------------------------------------------------------------------------
# An alert opens only after two bad readings in a row. This is what makes the
# sensitive model safe: about half its Critical labels are wrong, but a false
# alarm does not repeat and a real fault does.

OPEN_AFTER, CLOSE_AFTER = 2, 3


def should_open(recent_labels):
    """Copy of the rule in services/dashboard/app/main.py."""
    return len(recent_labels) >= OPEN_AFTER and all(
        label in ("Warning", "Critical")
        for label in recent_labels[:OPEN_AFTER]
    )


def should_close(recent_labels):
    return len(recent_labels) >= CLOSE_AFTER and all(
        label == "Optimal" for label in recent_labels[:CLOSE_AFTER]
    )


def test_one_bad_reading_does_not_alert():
    assert should_open(["Critical"]) is False


def test_two_bad_readings_in_a_row_do_alert():
    assert should_open(["Critical", "Warning"]) is True


def test_a_single_false_alarm_is_ignored():
    # The exact case the rule exists for: one bad reading between healthy ones.
    assert should_open(["Critical", "Optimal"]) is False


def test_one_good_reading_does_not_clear_an_alert():
    assert should_close(["Optimal"]) is False


def test_three_good_readings_clear_it():
    assert should_close(["Optimal", "Optimal", "Optimal"]) is True


def test_recovery_must_be_uninterrupted():
    assert should_close(["Optimal", "Warning", "Optimal"]) is False
