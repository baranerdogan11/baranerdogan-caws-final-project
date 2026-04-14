"""
pytest test suite for diagnose() pure function.
- 24 unit tests: 4 matrices × 6 scenarios
- 1 golden test: validates all 10 pieces against validation_expected.json
"""

import json
import csv
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from diagnose import diagnose

# ── Load fixtures ──────────────────────────────────────────────────────────────
API_DIR = Path(__file__).resolve().parent.parent

with open(API_DIR / "reference_times.json") as f:
    REFS = json.load(f)

with open(API_DIR / "validation_expected.json") as f:
    EXPECTED = json.load(f)

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_piece(piece_id, die_matrix, f2=None, d23=None, d34=None, d4a=None, dab=None):
    """Build cumulative piece dict from partial times + matrix references."""
    r = REFS[str(die_matrix)]
    def p(seg, delta=0.0):
        return (r[seg] + delta) if delta is not None else None

    t2 = f2 if f2 is not None else r["furnace_to_2nd_strike"]
    t3 = t2 + (r["2nd_to_3rd_strike"] if d23 is None else d23) if d23 != "NULL" else None
    t4 = t3 + (r["3rd_to_4th_strike"] if d34 is None else d34) if (t3 is not None and d34 != "NULL") else None
    ta = t4 + (r["4th_strike_to_aux_press"] if d4a is None else d4a) if (t4 is not None and d4a != "NULL") else None
    tb = ta + (r["aux_press_to_bath"] if dab is None else dab) if (ta is not None and dab != "NULL") else None

    return {
        "piece_id": piece_id,
        "die_matrix": die_matrix,
        "lifetime_2nd_strike_s": t2,
        "lifetime_3rd_strike_s": t3,
        "lifetime_4th_strike_s": t4,
        "lifetime_auxiliary_press_s": ta,
        "lifetime_bath_s": tb,
    }

PENALIZE = 2.0  # deviation that produces penalized=True (1.0 < 2.0 <= 5.0)

# ── 24 Unit Tests (4 matrices × 6 scenarios) ──────────────────────────────────
MATRICES = [4974, 5052, 5090, 5091]

@pytest.mark.parametrize("matrix", MATRICES)
def test_all_ok(matrix):
    piece = make_piece("T", matrix)
    result = diagnose(piece, REFS)
    assert result["delay"] is False
    assert result["probable_causes"] == []
    for seg in result["segments"]:
        assert seg["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_furnace_penalized(matrix):
    r = REFS[str(matrix)]
    piece = make_piece("T", matrix, f2=r["furnace_to_2nd_strike"] + PENALIZE)
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["furnace_to_2nd_strike"]["penalized"] is True
    for seg in ["2nd_to_3rd_strike", "3rd_to_4th_strike", "4th_strike_to_aux_press", "aux_press_to_bath"]:
        assert segs[seg]["penalized"] is False
    assert "Billet pick" in result["probable_causes"]


@pytest.mark.parametrize("matrix", MATRICES)
def test_2nd_to_3rd_penalized(matrix):
    r = REFS[str(matrix)]
    piece = make_piece("T", matrix, d23=r["2nd_to_3rd_strike"] + PENALIZE)
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["2nd_to_3rd_strike"]["penalized"] is True
    assert segs["furnace_to_2nd_strike"]["penalized"] is False
    assert "Retraction" in result["probable_causes"]


@pytest.mark.parametrize("matrix", MATRICES)
def test_3rd_to_4th_penalized(matrix):
    r = REFS[str(matrix)]
    piece = make_piece("T", matrix, d34=r["3rd_to_4th_strike"] + PENALIZE)
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["3rd_to_4th_strike"]["penalized"] is True
    assert segs["2nd_to_3rd_strike"]["penalized"] is False
    assert "synchronization" in result["probable_causes"]


@pytest.mark.parametrize("matrix", MATRICES)
def test_4th_to_aux_penalized(matrix):
    r = REFS[str(matrix)]
    piece = make_piece("T", matrix, d4a=r["4th_strike_to_aux_press"] + PENALIZE)
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["4th_strike_to_aux_press"]["penalized"] is True
    assert segs["3rd_to_4th_strike"]["penalized"] is False
    assert "interlocks" in result["probable_causes"]


@pytest.mark.parametrize("matrix", MATRICES)
def test_aux_to_bath_penalized(matrix):
    r = REFS[str(matrix)]
    piece = make_piece("T", matrix, dab=r["aux_press_to_bath"] + PENALIZE)
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["aux_press_to_bath"]["penalized"] is True
    assert segs["4th_strike_to_aux_press"]["penalized"] is False
    assert "bath queues" in result["probable_causes"]


# ── Edge case: sensor anomaly (deviation > 5.0) ────────────────────────────────
def test_anomaly_not_penalized():
    r = REFS["4974"]
    piece = make_piece("T", 4974, f2=r["furnace_to_2nd_strike"] + 6.0)
    result = diagnose(piece, REFS)
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["furnace_to_2nd_strike"]["penalized"] is None
    assert result["delay"] is False


# ── Edge case: null propagation ────────────────────────────────────────────────
def test_null_propagates():
    piece = {
        "piece_id": "T",
        "die_matrix": 5052,
        "lifetime_2nd_strike_s": 17.9,
        "lifetime_3rd_strike_s": None,
        "lifetime_4th_strike_s": 39.7,
        "lifetime_auxiliary_press_s": 55.9,
        "lifetime_bath_s": 57.7,
    }
    result = diagnose(piece, REFS)
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["2nd_to_3rd_strike"]["penalized"] is None
    assert segs["3rd_to_4th_strike"]["penalized"] is None
    assert segs["2nd_to_3rd_strike"]["actual_s"] is None


# ── Unknown die_matrix ─────────────────────────────────────────────────────────
def test_unknown_matrix():
    piece = make_piece("T", 4974)
    piece["die_matrix"] = 9999
    with pytest.raises(ValueError, match="unknown die_matrix"):
        diagnose(piece, REFS)


# ── Golden test: all 10 validation pieces ─────────────────────────────────────
def load_validation_pieces():
    rows = []
    with open(API_DIR / "validation_pieces.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


@pytest.mark.parametrize("expected", EXPECTED, ids=[e["piece_id"] for e in EXPECTED])
def test_golden(expected):
    piece_id = expected["piece_id"]
    rows = load_validation_pieces()
    row = next(r for r in rows if r["piece_id"] == piece_id)

    def to_float(v):
        return None if v == "" else float(v)

    piece = {
        "piece_id": row["piece_id"],
        "die_matrix": int(row["die_matrix"]),
        "lifetime_2nd_strike_s": to_float(row["lifetime_2nd_strike_s"]),
        "lifetime_3rd_strike_s": to_float(row["lifetime_3rd_strike_s"]),
        "lifetime_4th_strike_s": to_float(row["lifetime_4th_strike_s"]),
        "lifetime_auxiliary_press_s": to_float(row["lifetime_auxiliary_press_s"]),
        "lifetime_bath_s": to_float(row["lifetime_bath_s"]),
    }

    result = diagnose(piece, REFS)

    assert result["piece_id"] == expected["piece_id"]
    assert result["die_matrix"] == expected["die_matrix"]
    assert result["delay"] == expected["delay"]
    assert result["probable_causes"] == expected["probable_causes"]

    for r_seg, e_seg in zip(result["segments"], expected["segments"]):
        assert r_seg["segment"] == e_seg["segment"]
        assert r_seg["penalized"] == e_seg["penalized"]
        if r_seg["actual_s"] is not None and e_seg["actual_s"] is not None:
            assert round(r_seg["actual_s"], 1) == round(e_seg["actual_s"], 1)
        if r_seg["deviation_s"] is not None and e_seg["deviation_s"] is not None:
            assert round(r_seg["deviation_s"], 1) == round(e_seg["deviation_s"], 1)
