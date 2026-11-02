#!/usr/bin/env python3
"""Select each run's checkpoint on val accuracy, then score it on test.

Usage:
    python eval_erma.py
    python eval_erma.py --models SlowFast --taus 0.5
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

from datasets import erma
from models import slowfast as sf
from utils.config import load
from utils.metrics import classification


def score(cfg, model, tau, ckpt, split, args):
    data_dir = sf.as_test(erma.tau_dir(cfg, tau), split)
    sf_cfg = sf.test_cfg(cfg["slowfast_repo"], model, data_dir, ckpt, args.views, args.batch_size, args.workers)
    logits, y_true = sf.predict(sf_cfg)
    y_pred = logits.argmax(-1)
    return {**classification(y_true, y_pred), "y_true": y_true, "y_pred": y_pred}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(sf.CONFIGS))
    parser.add_argument("--taus", type=float, nargs="+", default=list(erma.TAUS))
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load()
    sf.setup(cfg["slowfast_repo"])
    out = cfg["results"] / "erma"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in args.models:
        for tau in args.taus:
            run = cfg["work"] / "runs" / f"{model}_tau{tau:g}"
            ckpts = sorted(run.glob("checkpoints/checkpoint_epoch_*.pyth"))
            if not ckpts:
                print(f"{model} tau={tau:g}: no checkpoints in {run}")
                continue
            val = []
            for ckpt in ckpts:
                epoch = int(re.search(r"(\d+)\.pyth$", ckpt.name).group(1))
                r = score(cfg, model, tau, ckpt, "val", args)
                val.append({"epoch": epoch, "checkpoint": str(ckpt), "acc": r["acc"], "macro_f1": r["macro_f1"]})
                print(f"{model} tau={tau:g} epoch {epoch:3d}  val Acc {r['acc']:5.1f}  mF1 {r['macro_f1']:5.1f}")
            best = max(val, key=lambda v: (v["acc"], v["macro_f1"], -v["epoch"]))
            (run / "selection.json").write_text(json.dumps({"selected": best, "all": val}, indent=2))

            r = score(cfg, model, tau, best["checkpoint"], "test", args)
            np.savez_compressed(out / f"{model}_tau{tau:g}.npz", y_true=r["y_true"], y_pred=r["y_pred"])
            rows.append({"model": model, "tau": tau, "epoch": best["epoch"],
                         **{k: r[k] for k in ("acc", "weighted_f1", "macro_f1", "n")}})
            print(f"{model} tau={tau:g} test Acc {r['acc']:5.1f}  wF1 {r['weighted_f1']:5.1f}  mF1 {r['macro_f1']:5.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "table2.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
