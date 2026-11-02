"""Load the benchmarked VLMs and decode a batch of (frames, prompt) pairs greedily."""

import gc
import sys
from dataclasses import dataclass, field

import torch

MODELS = {
    "InternVL2.5-1B": {"repo": "OpenGVLab/InternVL2_5-1B", "revision": "9d423ea1ae9f893897ee3f7493141073f5afcf22", "family": "internvl"},
    "InternVL2.5-4B": {"repo": "OpenGVLab/InternVL2_5-4B", "revision": "2cf4a8158bbc40d35015e7c63b527890de4d27b3", "family": "internvl"},
    "Qwen3-VL-2B": {"repo": "Qwen/Qwen3-VL-2B-Instruct", "revision": "89644892e4d85e24eaac8bacfd4f463576704203", "family": "qwen"},
    "Qwen3-VL-4B": {"repo": "Qwen/Qwen3-VL-4B-Instruct", "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17", "family": "qwen"},
    "Cosmos-Reason2-8B": {"repo": "nvidia/Cosmos-Reason2-8B", "revision": "a9fae2cf89dc64db96b12860417f0eb403013bb9", "family": "qwen"},
    "UniDriveVLA": {"repo": "owl10/UniDriveVLA_Nusc_Base_Stage1", "revision": "2a07638bc435eb163e0309a919a9d5e07465715d", "family": "qwen"},
    # built by tools/prepare_vlms.py
    "DriveLMM-o1": {"repo": "ayeshaishaq/DriveLMMo1", "revision": "b3a76fe56fdebc858023e45b15dbbe9e007db983", "family": "internvl", "derived": True},
    "Alpamayo-1.5-10B": {"repo": "nvidia/Alpamayo-1.5-10B", "revision": "7aba8293c09993f2e125c6819df05d7fa3e873ea", "family": "qwen", "derived": True},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class VLM:
    name: str
    family: str
    model: object
    processor: object = None
    tokenizer: object = None
    extras: dict = field(default_factory=dict)

    def free(self):
        self.model = self.processor = self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


def _dtype_kwarg(dtype):
    import transformers

    return {"dtype": dtype} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": dtype}


def load(name, derived_root=None, dtype=torch.bfloat16):
    spec = MODELS[name]
    if spec.get("derived"):
        path = derived_root / name
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"{name}: run tools/prepare_vlms.py --model {name} first")
        source, revision = str(path), None
    else:
        source, revision = spec["repo"], spec["revision"]

    if spec["family"] == "internvl":
        # InternVL2.5 remote code: AutoModel + its own chat template, no processor
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(source, revision=revision, trust_remote_code=True, use_flash_attn=True,
                                          low_cpu_mem_usage=False, **_dtype_kwarg(dtype)).eval().cuda()
        return VLM(name, "internvl", model, tokenizer=tokenizer)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    processor = AutoProcessor.from_pretrained(source, revision=revision, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        source, revision=revision, trust_remote_code=True, device_map="auto",
        attn_implementation=attn, **_dtype_kwarg(dtype)).eval()
    return VLM(name, "qwen", model, processor=processor, extras={"attn": attn})


def _split_rows(gen, stop_ids, decode):
    """(text, truncated) per row; truncated = no end-of-turn token within the budget."""
    out = []
    for row in gen.tolist():
        end = next((j for j, t in enumerate(row) if t in stop_ids), None)
        out.append((decode(row[:end] if end is not None else row).strip(), end is None))
    return out


def _qwen(vlm, batch_images, prompts, max_new_tokens):
    proc = vlm.processor
    texts = [proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": p}]}],
        tokenize=False, add_generation_prompt=True) for images, p in zip(batch_images, prompts)]
    proc.tokenizer.padding_side = "left"
    inputs = proc(text=texts, images=[im for images in batch_images for im in images], return_tensors="pt", padding=True)
    inputs = {k: v.to(vlm.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = vlm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    eos = vlm.model.generation_config.eos_token_id
    stop = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {proc.tokenizer.eos_token_id, proc.tokenizer.pad_token_id}
    return _split_rows(out[:, inputs["input_ids"].shape[1]:], stop - {None},
                       lambda r: proc.decode(r, skip_special_tokens=True))


def _internvl_pixels(image, size=448):
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(image.convert("RGB"))


def _internvl(vlm, batch_images, prompts, max_new_tokens):
    """InternVL's chat() prompt construction, one 448 tile per frame, decoded as a batch."""
    model, tok = vlm.model, vlm.tokenizer
    get_conv_template = sys.modules[type(model).__module__].get_conv_template
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    queries, pixels = [], []
    for images, prompt in zip(batch_images, prompts):
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], "".join(f"Frame{i + 1}: <image>\n" for i in range(len(images))) + prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for _ in images:
            query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>", 1)
        queries.append(query)
        pixels.extend(_internvl_pixels(im) for im in images)
    sep = template.sep.strip()
    tok.padding_side = "left"
    enc = tok(queries, return_tensors="pt", padding=True)
    with torch.no_grad():
        gen = model.generate(pixel_values=torch.stack(pixels).to(model.dtype).to(model.device),
                             input_ids=enc["input_ids"].to(model.device),
                             attention_mask=enc["attention_mask"].to(model.device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids(sep))
    stop = {tok.convert_tokens_to_ids(sep), tok.eos_token_id, tok.pad_token_id} - {None}
    return _split_rows(gen, stop, lambda r: tok.decode(r, skip_special_tokens=True).split(sep)[0])


def generate(vlm, batch_images, prompts, max_new_tokens):
    fn = _internvl if vlm.family == "internvl" else _qwen
    return fn(vlm, batch_images, prompts, max_new_tokens)
