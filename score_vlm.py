#!/usr/bin/env python3
"""Score VLM runs. ERMA: over parsed answers, with the parse rate.
TATP: on the 1,246-window comparison set, beside Static and the conventional models.

Usage:
    python score_vlm.py --task erma
    python score_vlm.py --task tatp
"""

import argparse
import json

import numpy as np
import pandas as pd

from datasets import tatp
from utils.config import load
from utils.metrics import classification, trajectory


def records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_erma(cfg):
    rows = []
    for path in sorted((cfg["results"] / "vlm").glob("*_erma*_tau*.jsonl")):
        name, tau = path.stem.rsplit("_tau", 1)
        recs = records(path)
        parsed = [r for r in recs if r["parsed"]]
        row = {"model": name.replace("_erma", ""), "tau": float(tau), "parse_rate": len(parsed) / len(recs)}
        if parsed:
            row.update(classification([r["truth_label"] for r in parsed], [r["pred_label"] for r in parsed]))
        rows.append(row)
    return pd.DataFrame(rows)


def score_tatp(cfg):
    vset = tatp.vlm_windows(cfg, "test")
    target = tatp.to_pixels(vset.fut)
    rows = []
    for path in sorted((cfg["results"] / "tatp").glob("*_test.npz")):
        saved = np.load(path)
        pos = {k: i for i, k in enumerate(zip(saved["clip_id"], saved["track_id"], saved["segment"].tolist(),
                                               saved["start"].tolist()))}
        take = [pos[tatp.window_key(vset, i)] for i in range(len(vset))]
        rows.append({"model": path.stem[:-5], "parse_rate": 1.0,
                     **trajectory(tatp.to_pixels(saved["pred"][take]), target)})
    for path in sorted((cfg["results"] / "vlm").glob("*_tatp.jsonl")):
        recs = {r["id"]: r for r in records(path)}
        if len(recs) < len(vset):
            print(f"{path.name}: {len(recs)}/{len(vset)} windows, skipped")
            continue
        pred = np.array([recs[i]["pred"] for i in range(len(vset))])
        parsed = np.array([recs[i]["parsed"] for i in range(len(vset))])
        # unparsed answers hold the last observed box
        row = {"model": path.stem[:-5], "parse_rate": float(parsed.mean())}
        if parsed.any():
            row.update(trajectory(pred, target))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["erma", "tatp"])
    args = parser.parse_args()

    cfg = load()
    df = score_erma(cfg) if args.task == "erma" else score_tatp(cfg)
    df.to_csv(cfg["results"] / f"vlm_{args.task}.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
