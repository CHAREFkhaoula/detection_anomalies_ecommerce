CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    user_segment TEXT,
    country TEXT
);


CREATE TABLE IF NOT EXISTS transactions(
    transaction_id TEXT,
    user_id TEXT REFERENCES users(user_id),
    timestamp TIMESTAMPTZ NOT NULL,
    amount FLOAT,
    quantity INTEGER,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    month INTEGER,
    time_since_last FLOAT,
    ip_geodist FLOAT,
    amount_per_item FLOAT,
    payment_method TEXT,
    is_anomaly BOOLEAN,
    PRIMARY KEY (transaction_id, timestamp)
);


SELECT create_hypertable('transactions', 'timestamp', if_not_exists => TRUE);
