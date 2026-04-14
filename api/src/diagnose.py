"""
Pure diagnosis function — no FastAPI dependencies.
All delay-detection logic lives here so it can be tested directly.
"""

from typing import Optional

# Cause table from §1.1
CAUSES = {
    "furnace_to_2nd_strike": [
        "Billet pick", "gripper close", "grip retries",
        "trajectory", "permissions", "queues"
    ],
    "2nd_to_3rd_strike": [
        "Retraction", "gripper", "press/PLC handshake",
        "wait points", "regrip"
    ],
    "3rd_to_4th_strike": [
        "Retraction", "conservative trajectory", "synchronization",
        "positioning", "confirmations"
    ],
    "4th_strike_to_aux_press": [
        "Pick micro-corrections", "transfer",
        "queue at Auxiliary Press entry", "interlocks"
    ],
    "aux_press_to_bath": [
        "Retraction", "transport", "bath queues",
        "permissions", "bath deposit"
    ],
}

SEGMENT_ORDER = [
    "furnace_to_2nd_strike",
    "2nd_to_3rd_strike",
    "3rd_to_4th_strike",
    "4th_strike_to_aux_press",
    "aux_press_to_bath",
]


def compute_partials(piece: dict) -> dict:
    """Derive 5 partial times from cumulative timestamps per §1.5.
    If either operand is None/missing, the partial is None.
    """
    def get(key):
        v = piece.get(key)
        return None if v is None else float(v)

    t2 = get("lifetime_2nd_strike_s")
    t3 = get("lifetime_3rd_strike_s")
    t4 = get("lifetime_4th_strike_s")
    ta = get("lifetime_auxiliary_press_s")
    tb = get("lifetime_bath_s")

    def diff(a, b):
        return None if (a is None or b is None) else round(a - b, 4)

    return {
        "furnace_to_2nd_strike":   t2,
        "2nd_to_3rd_strike":       diff(t3, t2),
        "3rd_to_4th_strike":       diff(t4, t3),
        "4th_strike_to_aux_press": diff(ta, t4),
        "aux_press_to_bath":       diff(tb, ta),
    }


def classify(actual: Optional[float], reference: float) -> tuple:
    """Apply §1.3 rules. Returns (deviation, penalized)."""
    if actual is None:
        return (None, None)
    deviation = round(actual - reference, 4)
    if deviation > 5.0:
        penalized = None   # sensor anomaly
    elif deviation > 1.0:
        penalized = True
    else:
        penalized = False
    return (deviation, penalized)


def diagnose(piece: dict, reference_times: dict) -> dict:
    """
    Pure function: takes one piece dict + reference_times dict,
    returns the full diagnosis response dict per §1.4 schema.

    Raises ValueError for unknown die_matrix.
    """
    piece_id = piece["piece_id"]
    die_matrix = int(piece["die_matrix"])
    key = str(die_matrix)

    if key not in reference_times:
        raise ValueError(f"unknown die_matrix {die_matrix}")

    refs = reference_times[key]
    partials = compute_partials(piece)

    segments = []
    probable_causes = []

    for seg in SEGMENT_ORDER:
        actual = partials[seg]
        reference = refs[seg]
        deviation, penalized = classify(actual, reference)

        # Round for output
        actual_out = round(actual, 1) if actual is not None else None
        dev_out = round(deviation, 1) if deviation is not None else None

        segments.append({
            "segment": seg,
            "actual_s": actual_out,
            "reference_s": round(reference, 1),
            "deviation_s": dev_out,
            "penalized": penalized,
        })

        if penalized is True:
            probable_causes.extend(CAUSES[seg])

    delay = any(s["penalized"] is True for s in segments)

    return {
        "piece_id": piece_id,
        "die_matrix": die_matrix,
        "delay": delay,
        "segments": segments,
        "probable_causes": probable_causes,
    }
