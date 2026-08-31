"""Feature-parity check: native DINOv3 backbone vs. the exported HF DINOv3ViTModel.

Answers "is a low feature cosine a weight-mapping bug, or a genuine inference-path
divergence between the two codebases?" by separating the two:

  Pass A  identical pre-normalized pixel tensor fed to both models. Preprocessing is
          bypassed entirely, so any gap here is weights / architecture / RoPE.
  Pass B  each model fed through its own real preprocessing pipeline (native
          bicubic+CenterCrop vs. the HF image processor). The A-to-B delta is the
          cost of the preprocessing divergence alone.

Also reports a per-layer error curve (where a mapping bug first appears) and, with
--compare-student, how far the student backbone sits from the teacher.

Usage:
    conda activate dinov3   # needs transformers >= 4.56 for DINOv3ViTModel
    python babyview/huggingface/check_parity.py \
        --ckpt-dir babyview/outputs/grad_accum_1/ckpt/119999 \
        --hf-dir   babyview/outputs/grad_accum_1/ckpt/119999/huggingface \
        --per-layer
"""

import argparse
import math
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dinov3.models.vision_transformer import vit_large  # noqa: E402

# Student block of dinov3/configs/train/vitl_babyview.yaml. Kept explicit rather than
# parsed so the parity check states exactly which architecture it is asserting.
STUDENT_CFG = dict(
    patch_size=16,
    in_chans=3,
    layerscale_init=1.0e-05,
    norm_layer="layernorm",
    ffn_layer="mlp",
    # ffn_ratio=4.0 is hardcoded by vit_large()
    qkv_bias=True,
    proj_bias=True,
    ffn_bias=True,
    n_storage_tokens=0,
    mask_k_bias=False,
    drop_path_rate=0.0,  # eval: no stochastic depth
    pos_embed_rope_base=100.0,
    pos_embed_rope_min_period=None,
    pos_embed_rope_max_period=None,
    pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_shift_coords=None,
    pos_embed_rope_jitter_coords=None,
    pos_embed_rope_rescale_coords=None,
    pos_embed_rope_dtype="bf16",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dcp_model_state(ckpt_dir: Path) -> dict:
    """Read only the `model` entry out of the sharded DCP checkpoint.

    Avoids dcp_to_torch_save, which would materialize a ~7.7 GB model+optimizer file.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sd: dict = {}
    _load_state_dict(
        sd,
        storage_reader=dcp.FileSystemReader(str(ckpt_dir)),
        planner=_EmptyStateDictLoadPlanner(keys=["model"]),
        no_dist=True,
    )
    return sd["model"]


def build_native(model_state: dict, prefix: str, device, dtype):
    src = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    if not src:
        raise SystemExit(f"no keys with prefix {prefix!r}; got e.g. {list(model_state)[:3]}")

    model = vit_large(**STUDENT_CFG)
    missing, unexpected = model.load_state_dict(src, strict=False)
    # rope_embed.periods is a persistent buffer present in both; flag any real gap.
    if missing or unexpected:
        print(f"  [native] missing={missing} unexpected={unexpected}")
    return model.to(device=device, dtype=dtype).eval(), src


def build_hf(hf_dir: Path, device, dtype):
    from transformers import DINOv3ViTModel

    model = DINOv3ViTModel.from_pretrained(str(hf_dir), dtype=dtype)
    return model.to(device).eval()


def stats(a: torch.Tensor, b: torch.Tensor) -> str:
    """Cosine / max-abs / relative Frobenius between two feature tensors."""
    a32, b32 = a.float().flatten(0, -2), b.float().flatten(0, -2)
    cos = torch.nn.functional.cosine_similarity(a32, b32, dim=-1)
    rel = (a32 - b32).norm() / a32.norm().clamp_min(1e-12)
    return (
        f"cos mean={cos.mean():.6f} min={cos.min():.6f}  "
        f"max|d|={(a32 - b32).abs().max():.3e}  relF={rel:.3e}"
    )


def make_pixel_tensor(image_path, size, device, dtype):
    """A normalized NCHW tensor both models can consume directly (no preprocessing)."""
    if image_path is None:
        g = torch.Generator().manual_seed(0)
        # Smooth-ish noise: a real image's spatial statistics matter for RoPE effects.
        x = torch.rand(2, 3, size, size, generator=g)
        x = torch.nn.functional.avg_pool2d(x, 5, stride=1, padding=2)
    else:
        from PIL import Image
        from torchvision.transforms import v2

        img = Image.open(image_path).convert("RGB")
        tf = v2.Compose([
            v2.ToImage(),
            v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
        ])
        x = tf(img).unsqueeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(device=device, dtype=dtype)


@torch.no_grad()
def pass_a(native, hf, x, per_layer: bool):
    print(f"\n=== Pass A: identical pixel tensor {tuple(x.shape)} (preprocessing bypassed) ===")

    nat_acts, hf_acts = {}, {}

    handles = []
    if per_layer:
        def mk(store, key):
            def hook(_m, _i, out):
                # native SelfAttentionBlock returns a list (multi-crop path); HF returns
                # a tensor or a tuple.
                while isinstance(out, (list, tuple)):
                    out = out[0]
                store[key] = out.detach().float()
            return hook

        for i, blk in enumerate(native.blocks):
            handles.append(blk.register_forward_hook(mk(nat_acts, i)))
        for i, layer in enumerate(hf.layer):
            handles.append(layer.register_forward_hook(mk(hf_acts, i)))

    nat = native.forward_features(x)
    out = hf(pixel_values=x)
    for h in handles:
        h.remove()

    nat_cls, nat_patch = nat["x_norm_clstoken"], nat["x_norm_patchtokens"]
    hf_cls = out.last_hidden_state[:, 0]
    hf_patch = out.last_hidden_state[:, 1 + hf.config.num_register_tokens:]

    print(f"  CLS   : {stats(nat_cls.unsqueeze(1), hf_cls.unsqueeze(1))}")
    print(f"  patch : {stats(nat_patch, hf_patch)}")

    if per_layer:
        print("\n  per-layer max|diff| (native blocks.i vs HF layer.i):")
        for i in sorted(nat_acts):
            if i not in hf_acts:
                continue
            d = (nat_acts[i] - hf_acts[i]).abs().max().item()
            scale = nat_acts[i].abs().max().item()
            print(f"    layer {i:2d}: {d:.3e}   (rel {d / max(scale, 1e-12):.2e})")

    return nat_cls, nat_patch, hf_cls, hf_patch


@torch.no_grad()
def pass_b(native, hf, hf_dir, image_path, size, device, dtype):
    """Each model through its own real preprocessing. Quantifies the H2 gap."""
    if image_path is None:
        print("\n=== Pass B: skipped (needs --image) ===")
        return
    from PIL import Image
    from torchvision.transforms import v2
    from transformers import AutoImageProcessor

    print(f"\n=== Pass B: each model through its own preprocessing ===")
    img = Image.open(image_path).convert("RGB")

    native_tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    x_nat = native_tf(img).unsqueeze(0).to(device=device, dtype=dtype)

    proc = AutoImageProcessor.from_pretrained(str(hf_dir))
    x_hf = proc(images=img, return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)

    print(f"  native tensor {tuple(x_nat.shape)} (bicubic + CenterCrop)")
    print(f"  HF     tensor {tuple(x_hf.shape)} (processor config)")

    nat = native.forward_features(x_nat)
    out = hf(pixel_values=x_hf)
    nat_cls = nat["x_norm_clstoken"]
    hf_cls = out.last_hidden_state[:, 0]

    if nat_cls.shape == hf_cls.shape:
        print(f"  CLS   : {stats(nat_cls.unsqueeze(1), hf_cls.unsqueeze(1))}")
    else:
        print(f"  CLS shapes differ: {tuple(nat_cls.shape)} vs {tuple(hf_cls.shape)}")


@torch.no_grad()
def compare_student(model_state, native_teacher, x, device, dtype):
    """H1: how different is the student backbone from the teacher we exported?"""
    print("\n=== teacher vs. student backbone (H1) ===")
    student, _ = build_native(model_state, "student.backbone.", device, dtype)
    t = native_teacher.forward_features(x)["x_norm_clstoken"]
    s = student.forward_features(x)["x_norm_clstoken"]
    print(f"  CLS   : {stats(t.unsqueeze(1), s.unsqueeze(1))}")


@torch.no_grad()
def ablate_preproc(native, image_path, size, device, dtype):
    """Which preprocessing knob costs what, against the native canonical pipeline.

    Same model throughout, so every number here is purely a pixel-pipeline effect.
    """
    if image_path is None:
        print("\n=== preprocessing ablation: skipped (needs --image) ===")
        return
    from PIL import Image
    from torchvision.transforms import v2

    print("\n=== preprocessing ablation (native model, varying pixels) ===")
    img = Image.open(image_path).convert("RGB")
    BIC, BIL = v2.InterpolationMode.BICUBIC, v2.InterpolationMode.BILINEAR

    def feats(resize, interp, crop):
        steps = [v2.ToImage(), v2.Resize(resize, interpolation=interp)]
        if crop:
            steps.append(v2.CenterCrop(size))
        steps += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        x = v2.Compose(steps)(img).unsqueeze(0).to(device=device, dtype=dtype)
        return native.forward_features(x)["x_norm_clstoken"]

    ref = feats(size, BIC, True)  # canonical: bicubic short-side resize + center crop
    variants = {
        "bicubic short-side + CenterCrop (canonical)": (size, BIC, True),
        "bilinear short-side + CenterCrop": (size, BIL, True),
        "bicubic squash to square, no crop": ([size, size], BIC, False),
        "bilinear squash to square, no crop  <-- shipped processor": ([size, size], BIL, False),
    }
    for name, (r, i, c) in variants.items():
        cos = torch.nn.functional.cosine_similarity(ref.float(), feats(r, i, c).float(), dim=-1)
        print(f"  {name:<46} cos={cos.mean():.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True, help="DCP checkpoint dir")
    p.add_argument("--hf-dir", type=Path, required=True, help="exported HF folder")
    p.add_argument("--prefix", default="teacher.backbone.")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per-layer", action="store_true")
    p.add_argument("--compare-student", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    print(f"reading DCP model state from {args.ckpt_dir} ...")
    model_state = load_dcp_model_state(args.ckpt_dir)
    print(f"  {len(model_state)} keys")

    print(f"building native vit_large from {args.prefix!r} ...")
    native, src = build_native(model_state, args.prefix, device, dtype)
    print(f"  rope periods dtype={src['rope_embed.periods'].dtype} "
          f"values[:4]={src['rope_embed.periods'][:4].float().tolist()}")

    print(f"loading HF model from {args.hf_dir} ...")
    hf = build_hf(args.hf_dir, device, dtype)
    inv = hf.rope_embeddings.inv_freq
    print(f"  HF inv_freq dtype={inv.dtype} -> periods[:4]={(1.0 / inv[:4]).tolist()}")

    x = make_pixel_tensor(args.image, args.size, device, dtype)
    pass_a(native, hf, x, args.per_layer)

    # H3 control: recompute the native rope periods in fp32, matching HF exactly. If the
    # Pass A residual is the bf16 periods buffer, this collapses it to numerical noise.
    print("\n=== H3 control: native rope periods forced to fp32 ===")
    native.rope_embed.dtype = torch.float32
    native.rope_embed.periods.data = native.rope_embed.periods.data.float()
    native.rope_embed._init_weights()
    pass_a(native, hf, x, per_layer=False)
    pass_b(native, hf, args.hf_dir, args.image, args.size, device, dtype)
    ablate_preproc(native, args.image, args.size, device, dtype)
    if args.compare_student:
        compare_student(model_state, native, x, device, dtype)


if __name__ == "__main__":
    main()
