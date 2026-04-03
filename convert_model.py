#!/usr/bin/env python3
"""Convert super-resolution / restoration models to Core ML (.mlpackage).

Supported architectures (auto-detected):
  ESRGAN / RRDBNet, DAT, HAT, SPAN, FDAT, RealPLKSR,
  GateRv2, GateRv3

Usage:
    python convert_model.py --input model.pth   --output model.mlpackage
    python convert_model.py --input model.safetensors --output model.mlpackage
    python convert_model.py --input model.pth --tile 256      # square tiles
    python convert_model.py --input model.pth --tile 896x640  # non-square
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import torch

from rrdbnet_arch import RRDBNet


# ---------------------------------------------------------------------------
# Key remapping for various ESRGAN .pth formats
# ---------------------------------------------------------------------------

REALESRGAN_REMAP = {
    r"RRDB_trunk\.":    "body.",
    r"trunk_conv\.":    "conv_body.",
    r"upconv1\.":       "conv_up1.",
    r"upconv2\.":       "conv_up2.",
    r"HRconv\.":        "conv_hr.",
    r"conv_first\.":    "conv_first.",
    r"conv_last\.":     "conv_last.",
}


def _is_old_arch(state_dict: dict) -> bool:
    return any(k.startswith("model.") for k in state_dict.keys())


def _remap_old_arch(state_dict: dict) -> OrderedDict:
    new_sd = OrderedDict()
    sub_indices = set()
    for k in state_dict.keys():
        m = re.match(r"model\.1\.sub\.(\d+)\.", k)
        if m:
            sub_indices.add(int(m.group(1)))
    trunk_idx = max(sub_indices) if sub_indices else 23

    for k, v in state_dict.items():
        new_key = k
        new_key = re.sub(r"^model\.0\.", "conv_first.", new_key)
        new_key = re.sub(rf"^model\.1\.sub\.{trunk_idx}\.", "conv_body.", new_key)
        m = re.match(r"^model\.1\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(.*)", new_key)
        if m:
            block, rdb, conv, rest = m.groups()
            new_key = f"body.{block}.rdb{rdb}.conv{conv}.{rest}"
        new_key = re.sub(r"^model\.3\.", "conv_up1.", new_key)
        new_key = re.sub(r"^model\.6\.", "conv_up2.", new_key)
        new_key = re.sub(r"^model\.8\.", "conv_hr.", new_key)
        new_key = re.sub(r"^model\.10\.", "conv_last.", new_key)
        new_sd[new_key] = v
    return new_sd


def _remap_keys(state_dict: dict) -> dict:
    if _is_old_arch(state_dict):
        return _remap_old_arch(state_dict)
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        for pattern, replacement in REALESRGAN_REMAP.items():
            new_key = re.sub(pattern, replacement, new_key)
        new_sd[new_key] = v
    return new_sd


def _load_state_dict_raw(path: str) -> dict:
    """Load state dict from .pth or .safetensors, converting to float32."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd: dict = load_file(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected format for {path}")
        if "params_ema" in raw:
            sd = raw["params_ema"]
        elif "params" in raw:
            sd = raw["params"]
        elif "model" in raw:
            sd = raw["model"]
        else:
            sd = raw

    # Convert bfloat16/float16 → float32 for stable tracing
    return {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _try_spandrel(path: str):
    """Try spandrel auto-detection. Returns (model, scale, size_req_minimum) or None."""
    try:
        from spandrel import ModelLoader
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ModelLoader().load_from_file(path)
        model = result.model  # type: ignore[union-attr]
        model = model.float()
        minimum = 1
        if hasattr(result, "size_requirements") and result.size_requirements is not None:
            minimum = max(1, result.size_requirements.minimum or 1)
        return model, result.scale, minimum  # type: ignore[union-attr]
    except Exception:
        return None


def _try_gaterv3(sd: dict):
    from gaterv3_arch import is_gaterv3, load_gaterv3
    if not is_gaterv3(sd):
        return None
    model = load_gaterv3(sd)
    return model, model.scale


def _try_gaterv2(sd: dict):
    from gaterv2_arch import is_gaterv2, load_gaterv2
    if not is_gaterv2(sd):
        return None
    model = load_gaterv2(sd)
    return model, model.scale


def _try_rrdbnet(sd: dict):
    sd2 = _remap_keys(sd)
    num_feat = 64
    for k, v in sd2.items():
        if "conv_first.weight" in k:
            num_feat = v.shape[0]
            break
    body_blocks = set()
    for k in sd2.keys():
        m = re.match(r"body\.(\d+)\.", k)
        if m:
            body_blocks.add(int(m.group(1)))
    if not body_blocks:
        return None  # Not an RRDBNet-style model
    num_block = max(body_blocks) + 1
    has_up3 = any("conv_up3" in k for k in sd2.keys())
    scale = 8 if has_up3 else 4
    num_in_ch = 3
    for k, v in sd2.items():
        if "conv_first.weight" in k:
            in_ch = v.shape[1]
            if in_ch == 12:
                scale = 2
            elif in_ch == 48:
                scale = 1
            else:
                num_in_ch = in_ch
            break
    model = RRDBNet(num_in_ch=num_in_ch, num_out_ch=3, num_feat=num_feat,
                    num_block=num_block, scale=scale, num_grow_ch=32)
    model.load_state_dict(sd2, strict=True)
    return model, scale


def load_model(path: str) -> tuple[torch.nn.Module, int, int]:
    """Load any supported SR/restoration model. Returns (model, scale, min_tile)."""
    print(f"  Detecting architecture via spandrel …")
    result = _try_spandrel(path)
    if result is not None:
        model, scale, min_tile = result
        print(f"  → {type(model).__name__}, scale={scale}")
        return model, scale, min_tile

    sd = _load_state_dict_raw(path)
    top_keys = {k.split(".")[0] for k in sd}

    # Try custom architectures that spandrel doesn't know about
    for label, fn in [("GateRv3", _try_gaterv3), ("GateRv2", _try_gaterv2), ("RRDBNet", _try_rrdbnet)]:
        print(f"  Trying {label} …")
        try:
            result = fn(sd)
        except Exception as exc:
            print(f"    {label} failed: {exc}")
            continue
        if result is not None:
            model, scale = result
            # infer min tile from model.pad if available
            min_tile = model.pad if (hasattr(model, "pad") and model.pad > 1) else 1
            print(f"  → {type(model).__name__}, scale={scale}")
            return model, scale, min_tile

    raise RuntimeError(
        f"Could not detect architecture for {path}.\n"
        "Supported: RRDBNet/ESRGAN, DAT, HAT, SPAN, FDAT, RealPLKSR, GateRv2, GateRv3.\n"
        f"Top-level state-dict keys: {sorted(top_keys)}"
    )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def parse_tile_size(tile_str: str) -> tuple[int, int]:
    """Parse 'WxH' or single int → (width, height)."""
    if "x" in tile_str.lower():
        parts = tile_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    val = int(tile_str)
    return val, val


def convert(args: argparse.Namespace) -> None:
    import coremltools as ct

    path = args.input
    print(f"[1/4] Loading model from {path} …")
    model, scale, min_tile = load_model(path)
    model.eval()

    tile_w, tile_h = parse_tile_size(args.tile)

    # Validate tile size against model requirements
    if min_tile > 1:
        if tile_h < min_tile or tile_w < min_tile:
            print(
                f"  ⚠️  WARNING: tile {tile_w}×{tile_h} is smaller than the minimum "
                f"required {min_tile}. Rounding up."
            )
            tile_h = max(tile_h, min_tile)
            tile_w = max(tile_w, min_tile)
        # For models that need tiles to be exact multiples (GateRv2/v3)
        if hasattr(model, "pad") and model.pad > 1:  # type: ignore[operator]
            pad = int(model.pad)  # type: ignore[arg-type]
            if tile_h % pad != 0:
                tile_h = ((tile_h + pad - 1) // pad) * pad
                print(f"  ⚠️  Rounded tile height up to {tile_h} (must be multiple of {pad})")
            if tile_w % pad != 0:
                tile_w = ((tile_w + pad - 1) // pad) * pad
                print(f"  ⚠️  Rounded tile width up to {tile_w} (must be multiple of {pad})")

    print(f"[2/4] {type(model).__name__}, scale={scale}, tile={tile_w}×{tile_h}")
    print(f"[3/4] Tracing …")
    dummy = torch.randn(1, 3, tile_h, tile_w)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # type: ignore[attr-defined]
        warnings.filterwarnings("ignore", message=".*meshgrid.*")
        traced = torch.jit.trace(model, dummy)

    print("[4/4] Converting to Core ML …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not been tested.*")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.ImageType(
                    name="input",
                    shape=(1, 3, tile_h, tile_w),
                    scale=1.0 / 255.0,
                    bias=[0, 0, 0],
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16 if not args.fp32 else ct.precision.FLOAT32,
        )

    spec = mlmodel.get_spec()  # type: ignore[union-attr]
    spec.description.metadata.userDefined["esrgan.scale"] = str(scale)
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)  # type: ignore[union-attr]

    output_path = args.output
    print(f"  Saving → {output_path}")
    mlmodel.save(output_path)
    print(f"✅  Done! (tile={tile_w}×{tile_h}, scale={scale})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SR/restoration model (.pth/.safetensors) → Core ML .mlpackage"
    )
    parser.add_argument("--input",  "-i", required=True,             help="Path to .pth or .safetensors model")
    parser.add_argument("--output", "-o", default="model.mlpackage", help="Output .mlpackage path")
    parser.add_argument("--tile",         default="256",              help="Tile size (e.g. 256 or 896x640)")
    parser.add_argument("--fp32",         action="store_true",        help="Use float32 (default: float16)")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
