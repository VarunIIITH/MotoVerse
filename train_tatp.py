#!/usr/bin/env python3
"""Train PIETraj or BiTraP-NP on TATP; best.pth is selected on val ADE at 1.5 s.

Usage:
    python train_tatp.py --model PIETraj
    python train_tatp.py --model BiTraP
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import tatp
from models import bitrap
from models.pietraj import PIETraj
from utils.config import load
from utils.metrics import trajectory

# per-model recipe of the reference implementations
RECIPES = {
    "PIETraj": {"epochs": 50, "batch_size": 16, "lr": 1e-3},
    "BiTraP": {"epochs": 50, "batch_size": 128, "lr": 1e-3},
}


def build_model(name, cfg, device):
    if name == "PIETraj":
        return PIETraj().to(device)
    model = bitrap.build(cfg["bitrap_repo"], device)
    model.param_scheduler = bitrap.kld_scheduler(device)
    return model


@torch.no_grad()
def predict(model, w, device, batch_size=512):
    """Zero-one cxcywh, (N, T, 4) for PIETraj or (N, K, T, 4) for BiTraP."""
    model.eval()
    out = []
    for i in range(0, len(w), batch_size):
        x = torch.from_numpy(w.obs[i:i + batch_size]).to(device)
        y = model(x)
        out.append((y if isinstance(model, PIETraj) else y[1].permute(0, 2, 1, 3)).cpu().numpy())
    return np.concatenate(out)


def val_ade(model, w, device):
    full = w.subset(w.full)
    return trajectory(tatp.to_pixels(predict(model, full, device)), tatp.to_pixels(full.fut))["ADE"]


def masked_mse(pred, target, mask):
    m = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(RECIPES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    recipe = RECIPES[args.model]
    epochs = args.epochs or recipe["epochs"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load()
    out_dir = cfg["work"] / "tatp" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    train_w = tatp.split_windows(cfg, "train", masked=True)
    val_w = tatp.split_windows(cfg, "val", masked=True)
    print(f"{len(train_w)} train windows | {len(val_w)} val windows ({int(val_w.full.sum())} full)")

    loader = DataLoader(tatp.TorchWindows(train_w), batch_size=recipe["batch_size"], shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    model = build_model(args.model, cfg, device)
    if args.model == "PIETraj":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=recipe["lr"], weight_decay=1e-4)
        plateau = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["lr"])
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, min_lr=1e-7)

    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loader:
            x, y, m = (batch[k].to(device, non_blocking=True) for k in ("input_x", "target_y", "target_mask"))
            if args.model == "PIETraj":
                loss = masked_mse(model(x), y, m)
            else:
                _, _, parts = model(x, y, m)
                kld_weight = float(model.param_scheduler.kld_weight)
                loss = parts["loss_goal"] + parts["loss_traj"] + kld_weight * parts["loss_kld"]
                model.param_scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            if args.model == "BiTraP":
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
            seen += len(x)

        if plateau is not None:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                for i in range(0, len(val_w), 512):
                    x, y, m = (torch.from_numpy(a[i:i + 512]).to(device) for a in (val_w.obs, val_w.fut, val_w.mask))
                    _, _, parts = model(x, y, m)
                    vloss += float(parts["loss_goal"] + parts["loss_traj"] + parts["loss_kld"]) * len(x)
            plateau.step(vloss / len(val_w))

        ade = val_ade(model, val_w, device)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_ade": ade})
        if ade < best:
            best = ade
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ade": ade}, out_dir / "best.pth")
        print(f"epoch {epoch:3d}/{epochs}  train {total / seen:.5f}  val ADE {ade:7.2f}{'  *' if ade == best else ''}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
