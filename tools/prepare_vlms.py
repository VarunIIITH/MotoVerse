#!/usr/bin/env python3
"""Build loadable copies of the two VLMs that cannot be loaded from their own repos.

DriveLMM-o1: its modeling_internvl_chat.py mixes tabs and spaces (TabError).
Alpamayo-1.5-10B: only the Qwen3-VL backbone (vlm.*) is kept; the trajectory
expert is dropped and the config comes from Cosmos-Reason2-8B, its named backbone.

Usage:
    python tools/prepare_vlms.py --model DriveLMM-o1 Alpamayo-1.5-10B
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vlm import MODELS
from utils.config import load


def snapshot(name):
    return Path(snapshot_download(MODELS[name]["repo"], revision=MODELS[name]["revision"]))


def drivelmm(dst):
    src = snapshot("DriveLMM-o1")
    for f in src.iterdir():
        target = dst / f.name
        target.unlink(missing_ok=True)
        if f.name == "modeling_internvl_chat.py":
            lines = f.read_text().splitlines(keepends=True)
            target.write_text("".join(ln.expandtabs(8) if re.match(r"^\t+ +\S", ln) else ln for ln in lines))
        else:
            target.symlink_to(f.resolve())
    for py in dst.glob("*.py"):
        compile(py.read_text(), str(py), "exec")


def alpamayo(dst):
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, backbone = snapshot("Alpamayo-1.5-10B"), snapshot("Cosmos-Reason2-8B")
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in weight_map if k.startswith("vlm.")]
    emb = next(k for k in keys if "embed_tokens" in k)
    with safe_open(src / weight_map[emb], framework="pt") as f:
        vocab = f.get_slice(emb).get_shape()[0]

    cfg = json.loads((backbone / "config.json").read_text())
    # the checkpoint's vocabulary is widened with trajectory tokens
    cfg["text_config"]["vocab_size"] = vocab
    cfg.setdefault("architectures", ["Qwen3VLForConditionalGeneration"])
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    shards = sorted({weight_map[k] for k in keys})
    new_map = {}
    for i, shard in enumerate(shards, 1):
        with safe_open(src / shard, framework="pt") as f:
            tensors = {k[4:]: f.get_tensor(k) for k in f.keys() if k.startswith("vlm.")}
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, dst / name, metadata={"format": "pt"})
        new_map.update({k: name for k in tensors})
        print(f"  {shard} -> {name} ({len(tensors)} tensors)")
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
              "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
              "video_preprocessor_config.json", "generation_config.json"):
        if (backbone / f).exists():
            shutil.copy(backbone / f, dst / f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["DriveLMM-o1", "Alpamayo-1.5-10B"],
                        choices=["DriveLMM-o1", "Alpamayo-1.5-10B"])
    args = parser.parse_args()

    root = load()["work"] / "vlm_derived"
    for name in args.model:
        dst = root / name
        dst.mkdir(parents=True, exist_ok=True)
        (drivelmm if name == "DriveLMM-o1" else alpamayo)(dst)
        print(f"{name} -> {dst}")


if __name__ == "__main__":
    main()
