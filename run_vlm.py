#!/usr/bin/env python3
"""Zero-shot VLM inference on the ERMA and TATP test sets. Resumable.

Usage:
    python run_vlm.py --model Qwen3-VL-4B --task erma
    python run_vlm.py --model Qwen3-VL-4B --task tatp --batch-size 8
    python run_vlm.py --model Qwen3-VL-2B --task erma --taus 0.5 --adapter outputs/lora/Qwen3-VL-2B_erma_tau0.5/final
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datasets import erma, tatp
from models import vlm as V
from models.slowfast import NUM_FRAMES
from utils import prompts
from utils.config import load

MAX_NEW_TOKENS = {"erma": 256, "cot": 1024, "tatp": 1800}


def sample_frames(path, n=NUM_FRAMES):
    """The same 16 frames the video backbones see: linspace(0, T-1, n)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in idx]


def done_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def decode(model, items, max_new_tokens, state):
    """Yield (key, text, truncated) for (key, images, prompt); halves the batch on OOM."""
    import torch

    pending = list(items)
    while pending:
        chunk = pending[:state["batch_size"]]
        try:
            outs = V.generate(model, [c[1] for c in chunk], [c[2] for c in chunk], max_new_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if state["batch_size"] == 1:
                raise
            state["batch_size"] //= 2
            continue
        for c, (text, truncated) in zip(chunk, outs):
            yield c[0], text, truncated
        pending = pending[len(chunk):]


def run_erma(model, cfg, tau, style, out_path, batch_size):
    df = erma.load_labels(cfg)
    done = done_ids(out_path)
    rows = [r for r in df[df["split"] == "test"].itertuples() if r.clip_id not in done]
    state = {"batch_size": batch_size}
    t0 = time.time()
    with open(out_path, "a") as f:
        for start in range(0, len(rows), batch_size * 4):
            group = rows[start:start + batch_size * 4]
            items = [(r, frames, prompts.erma_prompt(len(frames), style))
                     for r in group for frames in [sample_frames(erma.clip_path(cfg, r.clip_id, tau))]]
            for r, text, truncated in decode(model, items, MAX_NEW_TOKENS["cot" if style == "cot" else "erma"], state):
                label, how = prompts.parse_maneuver(text, truncated)
                f.write(json.dumps({"id": r.clip_id, "tau": tau, "truth_label": int(r.label),
                                    "pred_label": label, "parsed": label is not None, "how": how,
                                    "truncated": truncated, "raw": text}) + "\n")
                f.flush()
            print(f"  tau={tau:g}: {min(start + len(group), len(rows))}/{len(rows)} ({time.time() - t0:.0f}s)")


def run_tatp(model, cfg, out_path, batch_size):
    w = tatp.vlm_windows(cfg, "test")
    done = done_ids(out_path)
    todo = [i for i in range(len(w)) if i not in done]
    state = {"batch_size": batch_size}
    t0 = time.time()
    with open(out_path, "a") as f:
        for start in range(0, len(todo), batch_size * 4):
            items = []
            for i in todo[start:start + batch_size * 4]:
                observed = tatp.to_pixels(w.obs[i])
                images = [Image.open(p).convert("RGB") for p in tatp.frame_paths(cfg, w.clip_id[i], w.obs_frames[i])]
                items.append(((i, observed), images, prompts.tatp_prompt(observed)))
            for (i, observed), text, truncated in decode(model, items, MAX_NEW_TOKENS["tatp"], state):
                traj = prompts.parse_trajectory(text, observed)
                f.write(json.dumps({"id": i, "agent": tatp.AGENT_TYPES[w.agent_type[i]], "parsed": traj.parsed,
                                    "how": traj.how, "n_returned": traj.n_returned, "truncated": truncated,
                                    "pred": traj.boxes.round(2).tolist(), "raw": text[:2000]}) + "\n")
                f.flush()
            print(f"  {min(start + batch_size * 4, len(todo))}/{len(todo)} windows ({time.time() - t0:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(V.MODELS))
    parser.add_argument("--task", required=True, choices=["erma", "tatp"])
    parser.add_argument("--taus", type=float, nargs="+", default=list(erma.TAUS))
    parser.add_argument("--prompt", default="direct", choices=["direct", "cot"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--adapter", default=None, help="LoRA adapter from train_lora.py")
    parser.add_argument("--name", default=None, help="output prefix (default: --model)")
    args = parser.parse_args()

    cfg = load()
    out = cfg["results"] / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    model = V.load(args.model, cfg["work"] / "vlm_derived")
    if args.adapter:
        from peft import PeftModel

        model.model = PeftModel.from_pretrained(model.model, args.adapter).merge_and_unload().eval()
    name = args.name or args.model
    try:
        if args.task == "erma":
            tag = "_cot" if args.prompt == "cot" else ""
            for tau in args.taus:
                run_erma(model, cfg, tau, args.prompt, out / f"{name}_erma{tag}_tau{tau:g}.jsonl", args.batch_size)
        else:
            run_tatp(model, cfg, out / f"{name}_tatp.jsonl", args.batch_size)
    finally:
        model.free()


if __name__ == "__main__":
    main()
