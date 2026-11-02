#!/usr/bin/env python3
"""Extract every frame of the TATP clips at 960x540, as <work>/frames/<clip_id>/00000.png.

Usage:
    python tools/extract_frames.py --split test
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import tatp
from utils.config import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load()
    tracks = tatp.load_tracks(cfg)
    clips = sorted(tracks.loc[tracks["split"].isin(args.split), "clip_id"].unique())
    for k, clip_id in enumerate(clips, 1):
        out = cfg["work"] / "frames" / clip_id
        video = cfg["data_root"] / "clips" / f"{clip_id}.mp4"
        if any(out.glob("*.png")) or not video.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / f"{i:05d}.png"), cv2.resize(frame, tatp.IMAGE_SIZE, interpolation=cv2.INTER_AREA))
            i += 1
        cap.release()
        print(f"[{k}/{len(clips)}] {clip_id}: {i} frames")


if __name__ == "__main__":
    main()
