"""ERMA labels and tau-truncated observation windows."""

import multiprocessing as mp
from pathlib import Path

import cv2
import pandas as pd

CLASSES = [
    "straight", "left_turn", "right_turn", "left_lane_change",
    "right_lane_change", "slow_stop", "u_turn",
]
ABBREV = {
    "straight": "ST", "left_turn": "LT", "right_turn": "RT", "left_lane_change": "LLC",
    "right_lane_change": "RLC", "slow_stop": "SS", "u_turn": "UT",
}
TAUS = (0.2, 0.3, 0.5, 1.0)
PRE_ROLL_S = 1.0
CLIP_SIZE = (256, 256)


def load_labels(cfg):
    df = pd.read_csv(cfg["data_root"] / "erma.csv")
    df["label"] = df["maneuver"].map(CLASSES.index)
    return df


def tau_dir(cfg, tau):
    return cfg["work"] / "erma" / f"tau_{tau:g}"


def clip_path(cfg, clip_id, tau):
    return tau_dir(cfg, tau) / "clips" / f"{clip_id}.mp4"


def _cut(job):
    """Window = 1 s pre-roll + maneuver; keep its first tau fraction, resized to 256x256."""
    src, dst, start_s, end_s, tau = job
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, round(max(0.0, start_s - PRE_ROLL_S) * fps))
    end = min(round(end_s * fps), total) if total > 0 else round(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, CLIP_SIZE))
    cap.release()
    if not frames:
        return dst, False
    keep = max(1, round(len(frames) * tau))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, CLIP_SIZE)
    for frame in frames[:keep]:
        writer.write(frame)
    writer.release()
    return dst, True


def prepare(cfg, tau, workers=8):
    """Write the tau clips and PySlowFast ``path label`` CSVs for every split."""
    df = load_labels(cfg)
    out = tau_dir(cfg, tau)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in df.itertuples():
        dst = clip_path(cfg, r.clip_id, tau)
        if not dst.exists():
            jobs.append((cfg["data_root"] / "clips" / f"{r.clip_id}.mp4", dst, r.man_start_s, r.man_end_s, tau))
    with mp.Pool(workers) as pool:
        failed = [d for d, ok in pool.imap_unordered(_cut, jobs, chunksize=4) if not ok]
    df["path"] = [str(clip_path(cfg, c, tau)) for c in df["clip_id"]]
    df = df[df["path"].map(lambda p: Path(p).exists())]
    for split in ("train", "val", "test"):
        df[df["split"] == split][["path", "label"]].to_csv(out / f"{split}.csv", sep=" ", header=False, index=False)
    return len(df), failed
