#!/usr/bin/env python3
"""Score Static, PIETraj and BiTraP on the full-horizon test windows (960x540 pixels).

Usage:
    python eval_tatp.py
"""

import argparse

import numpy as np
import pandas as pd
import torch

from datasets import tatp
from train_tatp import build_model, predict
from utils.config import load
from utils.metrics import trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["PIETraj", "BiTraP"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = cfg["results"] / "tatp"
    out.mkdir(parents=True, exist_ok=True)
    test = tatp.split_windows(cfg, "test", masked=False)
    test = test.subset(test.full)
    target = tatp.to_pixels(test.fut)

    preds = {"Static": np.repeat(test.obs[:, -1:], tatp.PRED_LEN, axis=1)}
    for name in args.models:
        model = build_model(name, cfg, device)
        model.load_state_dict(torch.load(cfg["work"] / "tatp" / name / "best.pth", map_location=device)["model"])
        torch.manual_seed(args.seed)
        preds[name] = predict(model, test, device)

    rows = []
    for name, pred in preds.items():
        rows.append({"model": name, **trajectory(tatp.to_pixels(pred), target)})
        np.savez_compressed(out / f"{name}_test.npz", pred=pred.astype(np.float32),
                            clip_id=test.clip_id.astype(str), track_id=test.track_id.astype(str),
                            segment=test.segment, start=test.obs_frames[:, 0])
    df = pd.DataFrame(rows)
    df.to_csv(out / "table3.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
