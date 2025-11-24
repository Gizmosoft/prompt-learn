'''
    Constant which holds the string for the SQLite3 database schema
'''

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  name TEXT,
  model TEXT,
  temperature REAL,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS prompt_variants (
  variant_id TEXT PRIMARY KEY,
  experiment_id TEXT,
  label TEXT,
  template TEXT,
  notes TEXT,
  FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  experiment_id TEXT,
  variant_id TEXT,
  input_hash TEXT,
  started_at TEXT,
  finished_at TEXT,
  status TEXT,
  error_message TEXT,
  FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
  FOREIGN KEY (variant_id) REFERENCES prompt_variants(variant_id)
);

CREATE TABLE IF NOT EXISTS run_metrics (
  run_id TEXT PRIMARY KEY,
  latency_ms INTEGER,
  input_tokens INTEGER,
  output_tokens INTEGER,
  total_tokens INTEGER,
  cost_usd REAL,
  trace_url TEXT,
  metrics_confidence TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS run_outputs (
  run_id TEXT PRIMARY KEY,
  response_text TEXT,
  raw_json TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""