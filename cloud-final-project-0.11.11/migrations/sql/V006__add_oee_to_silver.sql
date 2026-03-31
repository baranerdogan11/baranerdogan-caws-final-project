-- Add OEE cycle time column to silver.
-- Computed as the rolling average of the last 5 inter-piece intervals.
ALTER TABLE silver.clean_pieces
    ADD COLUMN oee_cycle_time_s DOUBLE PRECISION;
