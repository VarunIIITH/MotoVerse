#!/usr/bin/env python3
"""Rename PyTorchVideo's Kinetics-400 C2D-R50 into PySlowFast's layout.

PySlowFast's own C2D Kinetics link is broken; the PyTorchVideo model is the same network.

Usage:
    python tools/convert_c2d.py
"""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load

PARTS = {
    "branch1_conv": "branch1", "branch1_norm": "branch1_bn",
    "branch2.conv_a": "branch2.a", "branch2.norm_a": "branch2.a_bn",
    "branch2.conv_b": "branch2.b", "branch2.norm_b": "branch2.b_bn",
    "branch2.conv_c": "branch2.c", "branch2.norm_c": "branch2.c_bn",
}


def convert(ptv):
    stages = sorted({int(m.group(1)) for k in ptv if (m := re.match(r"blocks\.(\d+)\.res_blocks\.", k))})
    stage_of = {b: f"s{i + 2}" for i, b in enumerate(stages)}
    out = {}
    for k, v in ptv.items():
        if k.startswith("blocks.0.conv."):
            out["s1.pathway0_stem.conv." + k.split(".", 3)[3]] = v
        elif k.startswith("blocks.0.norm."):
            out["s1.pathway0_stem.bn." + k.split(".", 3)[3]] = v
        elif m := re.match(r"blocks\.(\d+)\.res_blocks\.(\d+)\.(.+)\.(\w+)$", k):
            block, idx, part, leaf = m.groups()
            out[f"{stage_of[int(block)]}.pathway0_res{idx}.{PARTS[part]}.{leaf}"] = v
        elif m := re.match(r"blocks\.\d+\.proj\.(\w+)$", k):
            out[f"head.projection.{m.group(1)}"] = v
        else:
            raise KeyError(f"unmapped parameter: {k}")
    return out


def main():
    root = load()["pretrained"]
    ptv = torch.load(root / "C2D_8x8_R50.pyth", map_location="cpu", weights_only=False)["model_state"]
    torch.save({"model_state": convert(ptv), "epoch": 0}, root / "C2D_8x8_R50_ptv2sf.pyth")
    print(f"-> {root / 'C2D_8x8_R50_ptv2sf.pyth'}")


if __name__ == "__main__":
    main()
