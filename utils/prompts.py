"""Zero-shot prompts and answer parsers for ERMA and TATP."""

import json
import re
from dataclasses import dataclass

import numpy as np

from datasets.erma import ABBREV, CLASSES

# ── ERMA ──────────────────────────────────────────────────────────────

_OPTIONS = """You are analysing a short video recorded by a camera mounted on a motorcycle \
riding in dense, unstructured traffic. The {n} images are frames sampled in order from the \
beginning of a single manoeuvre performed by the rider.

Predict which manoeuvre the rider is performing. Choose exactly one:
1. Straight - continuing forward without changing direction or lane
2. Left Turn - turning left at a junction or corner
3. Right Turn - turning right at a junction or corner
4. Left Lane Change - moving into the lane to the left, continuing in the same direction
5. Right Lane Change - moving into the lane to the right, continuing in the same direction
6. Slow/Stop - decelerating or coming to a halt
7. U-Turn - reversing direction through roughly 180 degrees

"""

ERMA_PROMPTS = {
    "direct": _OPTIONS + "Answer with the manoeuvre name only, and nothing else.",
    "cot": _OPTIONS + """Reason briefly, then answer. Write ONE short sentence for each of:
1. Camera motion: rotating (a turn) or shifting sideways (a lane change)?
2. Speed: steady, or slowing to a stop?
3. Road: junction, intersection, or open road?

Do not repeat these questions. Do not describe the scene beyond the three sentences.
Then, on the final line, write exactly:
ANSWER: <manoeuvre name>""",
}

ERMA_ANSWERS = {
    "straight": "Straight", "left_turn": "Left Turn", "right_turn": "Right Turn",
    "left_lane_change": "Left Lane Change", "right_lane_change": "Right Lane Change",
    "slow_stop": "Slow/Stop", "u_turn": "U-Turn",
}

SURFACE = {
    "straight": ["going straight", "continue straight", "straight", "no turn"],
    "left_turn": ["turning left", "turn left", "left turn", "left_turn"],
    "right_turn": ["turning right", "turn right", "right turn", "right_turn"],
    "left_lane_change": ["changing lane to the left", "lane change left", "left lane change", "left_lane_change"],
    "right_lane_change": ["changing lane to the right", "lane change right", "right lane change", "right_lane_change"],
    "slow_stop": ["slow/stop", "slow down", "slowing", "stopping", "braking", "coming to a halt",
                  "decelerating", "slow stop", "slow_stop", "stop"],
    "u_turn": ["u-turn", "u turn", "u_turn", "uturn"],
}
_ORDERED = sorted(((s, c) for c, forms in SURFACE.items() for s in forms), key=lambda p: -len(p[0]))
_ANSWER_MARKER = re.compile(r"\b(?:final\s+)?answer\b\s*(?:is\b|:)|\bthe\s+(?:manoeuvre|maneuver)\s+is\b", re.I)
_CONCLUSION = re.compile(rf"{_ANSWER_MARKER.pattern}|\btherefore\b", re.I)
_NUMBERED = re.compile(r"^\s*\**([1-7])\**\s*[.):\-]?\s*")


def erma_prompt(n_frames, style="direct"):
    return ERMA_PROMPTS[style].format(n=n_frames)


def _find(text, last=False):
    best = None
    for surface, cls in _ORDERED:
        for m in re.finditer(rf"\b{re.escape(surface)}\b", text):
            if best is None or (last and m.start() > best[0]) or (not last and m.start() < best[0]):
                best = (m.start(), cls)
            if not last:
                break
        if best and not last:
            break
    return best[1] if best else None


def parse_maneuver(text, truncated=False):
    """Return (class index or None, rule). Unparsed answers are excluded from scoring."""
    raw = (text or "").strip()
    if not raw:
        return None, "empty"
    low = raw.lower()
    for cls in CLASSES:
        if low.rstrip(" .") in (cls, cls.replace("_", " "), ABBREV[cls].lower()):
            return CLASSES.index(cls), "exact"
    # a cut-off answer with no stated conclusion is not guessed
    if truncated and not _ANSWER_MARKER.search(low):
        return None, "truncated"
    tail = [ln.strip() for ln in raw.splitlines() if ln.strip()][-1].lower()
    if m := _NUMBERED.match(tail):
        return int(m.group(1)) - 1, "lastline"
    if cls := _find(tail):
        return CLASSES.index(cls), "lastline"
    conclusions = list(_CONCLUSION.finditer(low))
    if conclusions and (cls := _find(low[conclusions[-1].end():])):
        return CLASSES.index(cls), "conclusion"
    if m := _NUMBERED.match(low):
        return int(m.group(1)) - 1, "numbered"
    if cls := _find(low, last=True):
        return CLASSES.index(cls), "anywhere"
    return None, "unparsed"


# ── TATP ──────────────────────────────────────────────────────────────

TATP_PROMPT = """You are tracking one road agent seen from a camera mounted on a motorcycle \
riding in dense, unstructured traffic.

Each image is {width}x{height} pixels. The {obs} images are consecutive frames of the \
scene, in order. In each frame, the agent to track is the one inside the bounding box \
listed below for that frame.

Boxes are [x1, y1, x2, y2] in absolute pixel coordinates of the {width}x{height} image -- \
they are NOT normalised to 0-1. x runs from 0 to {width} and y from 0 to {height}, with \
(0, 0) at the top-left corner.

{observed}

Using both the frames and the boxes, predict where this agent will be in each of the \
next {pred} frames, continuing its motion through the scene. These are FUTURE frames \
that you have not been shown.

This is not an object detection task. Do NOT return detections for the frames above. \
Do NOT output labels, confidence scores, or any "bbox_2d"/"label" fields.

Output exactly {pred} lines and nothing else. Line i gives the predicted box for the \
i-th future frame, in this format:
1: [x1, y1, x2, y2]
2: [x1, y1, x2, y2]
...
{pred}: [x1, y1, x2, y2]

All four numbers are integers in absolute {width}x{height} pixel coordinates, the same \
convention as the boxes listed above. Every one of the {pred} lines must be present, \
even if the agent barely moves."""

_NUM = r"(-?\d+\.?\d*)"
_BOX = rf"\[\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*\]"


def box_lines(boxes, prefix=""):
    return "\n".join(f"{prefix}{i + 1}: [{', '.join(str(int(round(v))) for v in b)}]" for i, b in enumerate(boxes))


def tatp_prompt(observed, width=960, height=540, pred_len=45):
    return TATP_PROMPT.format(obs=len(observed), pred=pred_len, observed=box_lines(observed, "frame "),
                              width=width, height=height)


@dataclass
class Trajectory:
    boxes: np.ndarray
    parsed: bool
    how: str
    n_returned: int = 0


def _coerce(seq, pred_len):
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4 or len(arr) == 0:
        return None
    n = len(arr)
    # a short answer is padded by holding its last box
    return np.concatenate([arr, np.repeat(arr[-1:], max(0, pred_len - n), axis=0)])[:pred_len], n


def parse_trajectory(text, observed, pred_len=45):
    """45 boxes from the answer; an unparsed answer holds the last observed box."""
    raw = (text or "").strip()
    fenced = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    numbered = re.findall(rf"^\s*\d{{1,3}}\s*[:.)]\s*\[?\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*\]?",
                          fenced, re.M)
    candidates = []
    if len(numbered) >= 2:
        candidates.append(("numbered_lines", [[float(v) for v in g] for g in numbered]))
    for source in (fenced, raw):
        try:
            candidates.append(("json", json.loads(source)))
        except (json.JSONDecodeError, ValueError):
            pass
    if m := re.search(r"\[\s*\[.*?\]\s*\]", fenced, re.S):
        try:
            candidates.append(("embedded_json", json.loads(m.group(0))))
        except (json.JSONDecodeError, ValueError):
            pass
    if groups := re.findall(_BOX, fenced):
        candidates.append(("number_scan", [[float(v) for v in g] for g in groups]))
    for how, seq in candidates:
        try:
            got = _coerce(seq, pred_len)
        except (ValueError, TypeError):
            got = None
        if got is not None:
            return Trajectory(got[0], True, how, got[1])
    held = np.repeat(np.asarray(observed, dtype=np.float64)[-1:], pred_len, axis=0)
    return Trajectory(held, False, "empty" if not raw else "no boxes found")
