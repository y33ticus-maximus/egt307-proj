-- Database schema. Loaded automatically the first time PostgreSQL starts.

-- What is growing in each zone. Nutrient levels come from a soil test, not from
-- a sensor, so they are stored per zone and attached to every reading before
-- the model sees it.
CREATE TABLE IF NOT EXISTS zones (
    zone_id    TEXT PRIMARY KEY,
    soil_type  TEXT NOT NULL,
    nitrogen   REAL NOT NULL,
    phosphorus REAL NOT NULL,
    potassium  REAL NOT NULL
);

INSERT INTO zones (zone_id, soil_type, nitrogen, phosphorus, potassium) VALUES
    ('Zone_01', 'Loam Soil',     62.8, 48.2, 63.4),
    ('Zone_02', 'Sandy Soil',    58.8, 59.1, 52.4),
    ('Zone_03', 'Clay Soil',     45.1, 31.7, 44.9),
    ('Zone_04', 'Black Soil',    51.6, 36.2, 49.8)
ON CONFLICT (zone_id) DO NOTHING;

-- One row per sensor reading.
CREATE TABLE IF NOT EXISTS readings (
    id              BIGSERIAL PRIMARY KEY,
    zone_id         TEXT        NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL,
    temperature     REAL,
    soil_humidity   REAL,
    soil_moisture   REAL,
    air_humidity    REAL,
    ph              REAL,
    soil_ec         REAL,
    pressure        REAL,
    rainfall        REAL,
    light_intensity REAL,
    -- Sensors are nullable because the dataset contains genuine missing values
    -- and a real farm will have more. Rejecting a whole reading because one
    -- probe was silent would throw away the ten that worked.
    --
    -- The unique key makes a resent reading a no-op instead of a duplicate row.
    UNIQUE (zone_id, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_readings_zone_time ON readings (zone_id, recorded_at DESC);

-- The model's answer for each reading.
CREATE TABLE IF NOT EXISTS predictions (
    id             BIGSERIAL PRIMARY KEY,
    reading_id     BIGINT      NOT NULL REFERENCES readings(id) ON DELETE CASCADE,
    zone_id        TEXT        NOT NULL,
    label          TEXT        NOT NULL,   -- Optimal | Warning | Critical | pending
    confidence     REAL,
    driver         TEXT,                   -- the sensor furthest from healthy
    probable_cause TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (reading_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_zone ON predictions (zone_id, created_at DESC);

-- An incident is raised only after several bad readings in a row, so one zone
-- with a real problem produces one alert rather than one alert per reading.
CREATE TABLE IF NOT EXISTS incidents (
    id             BIGSERIAL PRIMARY KEY,
    zone_id        TEXT        NOT NULL,
    severity       TEXT        NOT NULL,   -- Warning | Critical
    probable_cause TEXT        NOT NULL,
    opened_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at      TIMESTAMPTZ,
    status         TEXT        NOT NULL DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_incidents_recent ON incidents (opened_at DESC);
