"""TATP sliding windows over gap-free track runs.

Observe 15 frames, predict 45, stride 7. Tracks are split at frame gaps so no
window spans a missing frame. Training windows keep futures of >= 15 steps,
masked; scoring uses full 45-step futures only.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OBS_LEN = 15
PRED_LEN = 45
STRIDE = 7
MIN_FUTURE = 15
IMAGE_SIZE = (960, 540)
MAX_BBOX = np.array([*IMAGE_SIZE, *IMAGE_SIZE], dtype=np.float64)
AGENT_TYPES = ("Motorcycle", "Car", "Truck", "Bus")


def normalize(boxes):
    """x1y1x2y2 pixels -> zero-one cxcywh."""
    b = np.asarray(boxes, dtype=np.float64).copy()
    b[..., 2:] -= b[..., :2]
    b[..., :2] += b[..., 2:] / 2
    return b / MAX_BBOX


def to_pixels(boxes):
    """Zero-one cxcywh -> x1y1x2y2 pixels at 960x540."""
    b = np.asarray(boxes, dtype=np.float64) * MAX_BBOX
    return np.concatenate([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], axis=-1)


def segments(frames):
    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    bounds = np.concatenate([[0], cuts, [len(frames)]])
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


@dataclass
class Windows:
    obs: np.ndarray         # (N, 15, 4) zero-one cxcywh
    fut: np.ndarray         # (N, 45, 4), zero past n_future
    mask: np.ndarray        # (N, 45) bool
    n_future: np.ndarray
    agent_type: np.ndarray
    obs_frames: np.ndarray  # (N, 15) frame indices in the clip
    clip_id: np.ndarray
    track_id: np.ndarray
    segment: np.ndarray
    run_length: np.ndarray
    meta: dict = field(default_factory=dict)

    ARRAYS = ("obs", "fut", "mask", "n_future", "agent_type", "obs_frames",
              "clip_id", "track_id", "segment", "run_length")

    def __len__(self):
        return len(self.obs)

    @property
    def full(self):
        """Full 1.5 s future on a run of at least 61 frames (JAAD convention)."""
        return (self.n_future == PRED_LEN) & (self.run_length >= OBS_LEN + PRED_LEN + 1)

    def subset(self, keep):
        return Windows(**{k: getattr(self, k)[keep] for k in self.ARRAYS}, meta=dict(self.meta))


def load_tracks(cfg):
    return pd.read_csv(cfg["data_root"] / "tatp.csv", dtype={"track_id": str})


def build(tracks, min_future=PRED_LEN):
    rows = {k: [] for k in Windows.ARRAYS}
    tracks = tracks.sort_values(["clip_id", "track_id", "frame"])
    for (clip_id, track_id), t in tracks.groupby(["clip_id", "track_id"], sort=True):
        frames = t["frame"].to_numpy()
        boxes = normalize(t[["x1", "y1", "x2", "y2"]].to_numpy())
        agent = AGENT_TYPES.index(t["agent_type"].iloc[0])
        for seg, run in enumerate(segments(frames)):
            f, b = frames[run], boxes[run]
            if len(f) < OBS_LEN + min_future + 1:
                continue
            for s in range(0, len(f) - (OBS_LEN + min_future) + 1, STRIDE):
                future = b[s + OBS_LEN: s + OBS_LEN + PRED_LEN]
                k = len(future)
                fut = np.zeros((PRED_LEN, 4))
                fut[:k] = future
                rows["obs"].append(b[s: s + OBS_LEN])
                rows["fut"].append(fut)
                rows["mask"].append(np.arange(PRED_LEN) < k)
                rows["n_future"].append(k)
                rows["agent_type"].append(agent)
                rows["obs_frames"].append(f[s: s + OBS_LEN])
                rows["clip_id"].append(clip_id)
                rows["track_id"].append(track_id)
                rows["segment"].append(seg)
                rows["run_length"].append(len(f))
    dtypes = {"obs": np.float32, "fut": np.float32, "mask": bool, "clip_id": object, "track_id": object}
    return Windows(**{k: np.asarray(v, dtype=dtypes.get(k, int)) for k, v in rows.items()},
                   meta={"min_future": min_future})


def split_windows(cfg, split, masked):
    tracks = load_tracks(cfg)
    return build(tracks[tracks["split"] == split], MIN_FUTURE if masked else PRED_LEN)


def vlm_windows(cfg, split="test"):
    """First full-horizon window of every gap-free run: the VLM comparison set."""
    w = split_windows(cfg, split, masked=False)
    seen, keep = set(), []
    for i in np.flatnonzero(w.full):
        run = (w.clip_id[i], w.track_id[i], w.segment[i])
        if run not in seen:
            seen.add(run)
            keep.append(i)
    return w.subset(np.asarray(keep, dtype=int))


def window_key(w, i):
    return (w.clip_id[i], w.track_id[i], int(w.segment[i]), int(w.obs_frames[i][0]))


def frame_paths(cfg, clip_id, frames):
    return [cfg["work"] / "frames" / clip_id / f"{int(f):05d}.png" for f in frames]


class TorchWindows:
    def __init__(self, w):
        self.w = w

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        import torch

        return {
            "input_x": torch.from_numpy(self.w.obs[i]),
            "target_y": torch.from_numpy(self.w.fut[i]),
            "target_mask": torch.from_numpy(self.w.mask[i]),
        }
