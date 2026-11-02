"""Path config, resolved against the repo root."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path=None):
    with open(path or ROOT / "configs" / "paths.yaml") as f:
        cfg = yaml.safe_load(f)
    return {k: Path(v) if Path(v).is_absolute() else ROOT / v for k, v in cfg.items()}
