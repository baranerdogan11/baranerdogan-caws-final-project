-- Raw lifetime data: cumulative piece travel times (seconds) from furnace exit to each stage.
-- Each row is one signal reading for one piece at one timestamp.
CREATE TABLE raw_lifetime (
    timestamp   TIMESTAMPTZ     NOT NULL,
    signal      TEXT            NOT NULL,
    value       DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_raw_lifetime_timestamp ON raw_lifetime (timestamp);
CREATE INDEX idx_raw_lifetime_signal ON raw_lifetime (signal);

-- Raw piece identification: piece ID and die matrix per timestamp.
-- Value is TEXT because piece_id is a string identifier (e.g. "251106001720").
CREATE TABLE raw_piece_info (
    timestamp   TIMESTAMPTZ     NOT NULL,
    signal      TEXT            NOT NULL,
    value       TEXT            NOT NULL
);

CREATE INDEX idx_raw_piece_info_timestamp ON raw_piece_info (timestamp);
CREATE INDEX idx_raw_piece_info_signal ON raw_piece_info (signal);
