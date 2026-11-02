"""PySlowFast video backbones for ERMA: configs, torch-2 shims and inference."""

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

CONFIGS = {
    "C2D": "C2D_8x8_R50.yaml",
    "I3D": "I3D_8x8_R50.yaml",
    "SlowFast": "SLOWFAST_8x8_R50.yaml",
    "X3D": "X3D_M.yaml",
}
# Kinetics-400 initialisation: (file under cfg["pretrained"], checkpoint type)
PRETRAINED = {
    "C2D": ("C2D_8x8_R50_ptv2sf.pyth", "pytorch"),
    "I3D": ("I3D_8x8_R50.pkl", "caffe2"),
    "SlowFast": ("SLOWFAST_8x8_R50.pkl", "caffe2"),
    "X3D": ("x3d_m.pyth", "pytorch"),
}
# same 16 frames for every backbone and every VLM
NUM_FRAMES = 16
SAMPLING_RATE = 4
NUM_CLASSES = 7
EVAL_PERIOD = 5


def _install_shims():
    if "torch._six" not in sys.modules:
        import collections.abc

        six = types.ModuleType("torch._six")
        six.int_classes, six.string_classes, six.container_abcs = int, str, collections.abc
        six.inf, six.nan = math.inf, math.nan
        sys.modules["torch._six"] = six
    try:
        import detectron2.layers  # noqa: F401
    except ImportError:
        from torchvision.ops import RoIAlign

        class ROIAlign(RoIAlign):
            def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
                super().__init__(output_size, spatial_scale, sampling_ratio, aligned)

        pkg, layers = types.ModuleType("detectron2"), types.ModuleType("detectron2.layers")
        layers.ROIAlign = ROIAlign
        pkg.layers = layers
        sys.modules["detectron2"], sys.modules["detectron2.layers"] = pkg, layers


def _install_tau_sampling():
    """Spread NUM_FRAMES over the whole (tau-truncated) clip instead of a fixed span."""
    from slowfast.datasets import decoder

    decoder.get_start_end_idx = lambda video_size, *args, **kwargs: (0.0, float(max(video_size - 1, 0)), 0.0)


def _install_safe_json_logging():
    """Log inf/nan stats (AMP overflow steps) as null instead of crashing."""
    import slowfast.utils.logging as sf_logging

    original = sf_logging.log_json_stats

    def log_json_stats(stats, output_dir=None):
        clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in stats.items()}
        return original(clean, output_dir)

    sf_logging.log_json_stats = log_json_stats


def setup(repo):
    _install_shims()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_tau_sampling()
    _install_safe_json_logging()


def _base_cfg(repo, model, data_dir):
    from slowfast.config.defaults import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(str(Path(repo) / "configs" / "Kinetics" / CONFIGS[model]))
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.DATA.PATH_TO_DATA_DIR = str(data_dir)
    cfg.DATA.NUM_FRAMES = NUM_FRAMES
    cfg.DATA.SAMPLING_RATE = SAMPLING_RATE
    cfg.DATA.DECODING_BACKEND = "pyav"
    cfg.MODEL.NUM_CLASSES = NUM_CLASSES
    cfg.BN.USE_PRECISE_STATS = False
    return cfg


def train_cfg(repo, model, data_dir, out_dir, pretrained, epochs=100, batch_size=8, workers=4):
    cfg = _base_cfg(repo, model, data_dir)
    cfg.OUTPUT_DIR = str(out_dir)
    cfg.TRAIN.ENABLE = True
    cfg.TRAIN.DATASET = "kinetics"
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.EVAL_PERIOD = EVAL_PERIOD
    cfg.TRAIN.CHECKPOINT_PERIOD = EVAL_PERIOD
    cfg.TRAIN.AUTO_RESUME = True
    cfg.TRAIN.CHECKPOINT_FILE_PATH = str(pretrained)
    cfg.TRAIN.CHECKPOINT_TYPE = PRETRAINED[model][1]
    cfg.TRAIN.CHECKPOINT_EPOCH_RESET = True
    cfg.TRAIN.CHECKPOINT_INFLATE = False
    cfg.TRAIN.MIXED_PRECISION = True
    cfg.TEST.ENABLE = False
    # keep the config's warmup fraction under the shorter schedule
    cfg.SOLVER.WARMUP_EPOCHS *= epochs / (cfg.SOLVER.MAX_EPOCH or epochs)
    cfg.SOLVER.MAX_EPOCH = epochs
    cfg.DATA_LOADER.NUM_WORKERS = workers
    cfg.DATA_LOADER.PIN_MEMORY = True
    cfg.LOG_PERIOD = 20
    return cfg


def test_cfg(repo, model, data_dir, checkpoint, views=10, batch_size=8, workers=4):
    cfg = _base_cfg(repo, model, data_dir)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.TEST.DATASET = "kinetics"
    cfg.TEST.BATCH_SIZE = batch_size
    cfg.TEST.NUM_ENSEMBLE_VIEWS = views
    # clips are square 256x256, so the 3 upstream crops would be identical
    cfg.TEST.NUM_SPATIAL_CROPS = 1
    cfg.TEST.CHECKPOINT_FILE_PATH = str(checkpoint)
    cfg.TEST.CHECKPOINT_TYPE = "pytorch"
    cfg.DATA_LOADER.NUM_WORKERS = workers
    return cfg


def as_test(data_dir, split):
    """A data dir whose test.csv is ``split``'s CSV (PySlowFast's multi-view path only reads test.csv)."""
    data_dir = Path(data_dir)
    if split == "test":
        return data_dir
    view = data_dir / f"_as_test_{split}"
    view.mkdir(exist_ok=True)
    link = view / "test.csv"
    link.unlink(missing_ok=True)
    link.symlink_to((data_dir / f"{split}.csv").resolve())
    return view


@torch.no_grad()
def predict(cfg):
    """Per-clip (logits averaged over views, labels)."""
    import slowfast.utils.checkpoint as cu
    from slowfast.datasets import loader
    from slowfast.models import build_model

    model = build_model(cfg).eval()
    cu.load_test_checkpoint(cfg, model)
    num_views = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
    sums, counts, truth = {}, {}, {}
    for inputs, labels, index, *_ in loader.construct_loader(cfg, "test"):
        inputs = [x.cuda(non_blocking=True) for x in inputs] if isinstance(inputs, (list, tuple)) else inputs.cuda()
        logits = model(inputs).cpu().numpy()
        for p, y, v in zip(logits, labels.numpy(), index.numpy()):
            v = int(v) // num_views
            sums[v] = sums.get(v, 0) + p.astype(np.float64)
            counts[v] = counts.get(v, 0) + 1
            truth[v] = int(y)
    order = sorted(sums)
    return np.stack([sums[v] / counts[v] for v in order]), np.array([truth[v] for v in order])
