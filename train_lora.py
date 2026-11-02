#!/usr/bin/env python3
"""LoRA adaptation of Qwen3-VL-2B or UniDriveVLA to ERMA (one tau) or TATP.

LoRA r=16, alpha=32, dropout 0.05 on the language model's q/k/v/o projections;
AdamW 1e-4, cosine with 3% warmup, 3 epochs, bf16. The prompt is the zero-shot
one and the loss covers the answer tokens only. The last epoch is evaluated.

Usage:
    python train_lora.py --model Qwen3-VL-2B --task erma --tau 0.5
    python train_lora.py --model UniDriveVLA --task tatp
"""

import argparse
import json
import math

import numpy as np
import torch
from PIL import Image

from datasets import erma, tatp
from models import vlm as V
from run_vlm import sample_frames
from utils import prompts
from utils.config import load


def erma_items(cfg, split, tau):
    df = erma.load_labels(cfg)
    items = [(erma.clip_path(cfg, r.clip_id, tau), r.maneuver) for r in df[df["split"] == split].itertuples()]

    def get(item):
        frames = sample_frames(item[0])
        return frames, prompts.erma_prompt(len(frames)), prompts.ERMA_ANSWERS[item[1]]

    return items, get


def tatp_items(cfg, split):
    w = tatp.vlm_windows(cfg, split)
    items = [(paths, tatp.to_pixels(w.obs[i]), tatp.to_pixels(w.fut[i]))
             for i in range(len(w)) for paths in [tatp.frame_paths(cfg, w.clip_id[i], w.obs_frames[i])]
             if all(p.exists() for p in paths)]

    def get(item):
        paths, observed, future = item
        return [Image.open(p).convert("RGB") for p in paths], prompts.tatp_prompt(observed), prompts.box_lines(future)

    return items, get


def encode(processor, images, prompt, answer):
    """Chat-template inputs with labels on the answer and end-of-turn token only."""
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    prefix = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True)
    tail = answer + "<|im_end|>"
    inputs = processor(text=[prefix + tail], images=images, return_tensors="pt")
    tail_ids = processor.tokenizer(tail, add_special_tokens=False)["input_ids"]
    n = len(tail_ids)
    if inputs["input_ids"][0, -n:].tolist() != tail_ids:
        raise ValueError("answer tokenises differently after the prompt")
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -n:] = inputs["input_ids"][:, -n:]
    inputs["labels"] = labels
    return dict(inputs)


class Examples(torch.utils.data.Dataset):
    def __init__(self, processor, items, get):
        self.processor, self.items, self.get = processor, items, get

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return encode(self.processor, *self.get(self.items[i]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["Qwen3-VL-2B", "UniDriveVLA"])
    parser.add_argument("--task", required=True, choices=["erma", "tatp"])
    parser.add_argument("--tau", type=float, default=None, choices=erma.TAUS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.task == "erma" and args.tau is None:
        parser.error("--tau is required for erma")

    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    cfg = load()
    name = f"{args.model}_{args.task}" + (f"_tau{args.tau:g}" if args.task == "erma" else "")
    out = cfg["work"] / "lora" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    vlm = V.load(args.model)
    model = vlm.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"))
    model.train()

    if args.task == "erma":
        (train, get), (val, val_get) = erma_items(cfg, "train", args.tau), erma_items(cfg, "val", args.tau)
    else:
        (train, get), (val, val_get) = tatp_items(cfg, "train"), tatp_items(cfg, "val")
    print(f"{name}: {len(train)} train, {len(val)} val")

    params = [p for p in model.parameters() if p.requires_grad]
    total = math.ceil(len(train) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.03 * total), total)
    one = lambda b: b[0]  # noqa: E731
    to_dev = lambda b: {k: v.to(model.device) if hasattr(v, "to") else v for k, v in b.items()}  # noqa: E731
    log = open(out / "train_log.jsonl", "a")

    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        loader = torch.utils.data.DataLoader(Examples(vlm.processor, [train[j] for j in order], get),
                                             batch_size=1, num_workers=args.workers, collate_fn=one)
        opt.zero_grad(set_to_none=True)
        running = []
        for k, batch in enumerate(loader, 1):
            loss = model(**to_dev(batch)).loss
            (loss / args.grad_accum).backward()
            running.append(float(loss))
            if k % args.grad_accum and k < len(train):
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if (k // args.grad_accum) % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "example": k, "loss": round(np.mean(running), 4)}))
                running = []

        model.eval()
        with torch.no_grad():
            val_loader = torch.utils.data.DataLoader(Examples(vlm.processor, val, val_get), batch_size=1,
                                                     num_workers=args.workers, collate_fn=one)
            val_loss = float(np.mean([float(model(**to_dev(b)).loss) for b in val_loader]))
        model.train()
        model.save_pretrained(out / f"epoch{epoch + 1}")
        log.write(json.dumps({"epoch": epoch + 1, "val_loss": val_loss}) + "\n")
        log.flush()
        print(f"epoch {epoch + 1}: val loss {val_loss:.4f}")

    final = out / "final"
    final.unlink(missing_ok=True)
    final.symlink_to(f"epoch{args.epochs}")


if __name__ == "__main__":
    main()
