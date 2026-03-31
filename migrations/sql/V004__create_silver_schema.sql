-- Silver layer: cleaned and pivoted piece data.
-- Populated incrementally by the 01_bronze_to_silver notebook.

CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE silver.clean_pieces (
    timestamp           TIMESTAMPTZ         NOT NULL,
    piece_id            TEXT                NOT NULL,
    die_matrix          INTEGER             NOT NULL,
    lifetime_first_s    DOUBLE PRECISION,
    lifetime_second_s   DOUBLE PRECISION,
    lifetime_drill_s    DOUBLE PRECISION,
    lifetime_bath_s     DOUBLE PRECISION,
    lifetime_general_s  DOUBLE PRECISION,
    processed_at        TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_silver_pieces_timestamp ON silver.clean_pieces (timestamp);
CREATE INDEX idx_silver_pieces_die_matrix ON silver.clean_pieces (die_matrix);
CREATE INDEX idx_silver_pieces_processed_at ON silver.clean_pieces (processed_at);
