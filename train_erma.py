#!/usr/bin/env python3
"""Fine-tune one video backbone at one observation ratio tau.

Usage:
    python train_erma.py --model SlowFast --tau 0.5
"""

import argparse
import importlib.util
from pathlib import Path

from datasets import erma
from models import slowfast as sf
from utils.config import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(sf.CONFIGS))
    parser.add_argument("--tau", type=float, required=True, choices=erma.TAUS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load()
    sf.setup(cfg["slowfast_repo"])
    from slowfast.utils.misc import launch_job

    # PySlowFast's tools/ is not a package
    spec = importlib.util.spec_from_file_location("train_net", Path(cfg["slowfast_repo"]) / "tools" / "train_net.py")
    train_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_net)

    out_dir = cfg["work"] / "runs" / f"{args.model}_tau{args.tau:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf_cfg = sf.train_cfg(cfg["slowfast_repo"], args.model, erma.tau_dir(cfg, args.tau), out_dir,
                          cfg["pretrained"] / sf.PRETRAINED[args.model][0],
                          epochs=args.epochs, batch_size=args.batch_size, workers=args.workers)
    launch_job(cfg=sf_cfg, init_method="tcp://localhost:9999", func=train_net.train)


if __name__ == "__main__":
    main()
