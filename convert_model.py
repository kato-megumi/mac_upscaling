#!/usr/bin/env python3
"""Convert an ESRGAN / Real-ESRGAN .pth model to Core ML (.mlpackage).

Usage:
    python convert_model.py --input model.pth --output model.mlpackage [--scale 4] [--num_block 23] [--tile 256]
    python convert_model.py --input model.pth --output model.mlpackage --tile 896x640   # non-square
"""

import argparse
import re
from collections import OrderedDict

import torch
import coremltools as ct

from rrdbnet_arch import RRDBNet


# ---------------------------------------------------------------------------
# Key remapping for various ESRGAN .pth formats
# ---------------------------------------------------------------------------

# Format A: Real-ESRGAN / BasicSR naming (e.g. RRDB_trunk., trunk_conv., etc.)
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
    """Detect old-arch ESRGAN format (model.0.weight, model.1.sub.N.RDB...)."""
    return any(k.startswith("model.") for k in state_dict.keys())


def _remap_old_arch(state_dict: dict) -> OrderedDict:
    """Remap old-arch ESRGAN keys (model.N.*) to RRDBNet convention.

    Old format:
        model.0        → conv_first
        model.1.sub.N  → body.N  (RRDB blocks, N < num_blocks)
        model.1.sub.B  → conv_body  (B == num_blocks, the trunk conv)
        model.3        → conv_up1
        model.6        → conv_up2
        model.8        → conv_hr
        model.10       → conv_last

    Inside RRDB blocks:
        model.1.sub.N.RDB{1,2,3}.conv{1-5}.0.{weight,bias}
        → body.N.rdb{1,2,3}.conv{1-5}.{weight,bias}
    """
    new_sd = OrderedDict()

    # Determine num_blocks: find max sub index used by RRDB blocks
    sub_indices = set()
    for k in state_dict.keys():
        m = re.match(r"model\.1\.sub\.(\d+)\.", k)
        if m:
            sub_indices.add(int(m.group(1)))
    # The trunk conv is the highest-numbered sub entry
    trunk_idx = max(sub_indices) if sub_indices else 23

    for k, v in state_dict.items():
        new_key = k

        # model.0.* → conv_first.*
        new_key = re.sub(r"^model\.0\.", "conv_first.", new_key)

        # model.1.sub.{trunk_idx}.* → conv_body.*
        new_key = re.sub(rf"^model\.1\.sub\.{trunk_idx}\.", "conv_body.", new_key)

        # model.1.sub.N.RDB{1,2,3}.conv{1-5}.0.* → body.N.rdb{1,2,3}.conv{1-5}.*
        m = re.match(r"^model\.1\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(.*)", new_key)
        if m:
            block, rdb, conv, rest = m.groups()
            new_key = f"body.{block}.rdb{rdb}.conv{conv}.{rest}"

        # model.3.* → conv_up1.*
        new_key = re.sub(r"^model\.3\.", "conv_up1.", new_key)
        # model.6.* → conv_up2.*
        new_key = re.sub(r"^model\.6\.", "conv_up2.", new_key)
        # model.8.* → conv_hr.*
        new_key = re.sub(r"^model\.8\.", "conv_hr.", new_key)
        # model.10.* → conv_last.*
        new_key = re.sub(r"^model\.10\.", "conv_last.", new_key)

        new_sd[new_key] = v

    return new_sd


def remap_keys(state_dict: dict) -> dict:
    """Remap old ESRGAN key names to the RRDBNet convention."""
    if _is_old_arch(state_dict):
        return _remap_old_arch(state_dict)

    # Real-ESRGAN / BasicSR format
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        for pattern, replacement in REALESRGAN_REMAP.items():
            new_key = re.sub(pattern, replacement, new_key)
        new_sd[new_key] = v
    return new_sd


def detect_params_from_state_dict(state_dict: dict) -> dict:
    """Auto-detect num_feat, num_block, and scale from state dict keys."""
    # num_feat from conv_first weight shape
    num_feat = 64
    for k, v in state_dict.items():
        if "conv_first.weight" in k:
            num_feat = v.shape[0]
            break

    # num_block by counting body.N keys
    body_blocks = set()
    for k in state_dict.keys():
        m = re.match(r"body\.(\d+)\.", k)
        if m:
            body_blocks.add(int(m.group(1)))
    num_block = max(body_blocks) + 1 if body_blocks else 23

    # scale: check for conv_up3 → 8x, else 4x
    has_up3 = any("conv_up3" in k for k in state_dict.keys())
    scale = 8 if has_up3 else 4

    # Check input channels (pixel_unshuffle for 2x)
    num_in_ch = 3
    for k, v in state_dict.items():
        if "conv_first.weight" in k:
            in_ch = v.shape[1]
            if in_ch == 12:
                num_in_ch = 3
                scale = 2
            elif in_ch == 48:
                num_in_ch = 3
                scale = 1
            else:
                num_in_ch = in_ch
            break

    return {
        "num_in_ch": num_in_ch,
        "num_out_ch": 3,
        "num_feat": num_feat,
        "num_block": num_block,
        "scale": scale,
        "num_grow_ch": 32,
    }


def load_pth(path: str) -> dict:
    """Load .pth and extract state dict, handling various save formats."""
    raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        if "params_ema" in raw:
            sd = raw["params_ema"]
        elif "params" in raw:
            sd = raw["params"]
        elif "model" in raw:
            sd = raw["model"]
        else:
            sd = raw
    else:
        raise ValueError("Unexpected .pth format — expected a dict.")

    return remap_keys(sd)


def parse_tile_size(tile_str: str) -> tuple[int, int]:
    """Parse tile size from 'WxH' or single int string. Returns (width, height)."""
    if "x" in tile_str.lower():
        parts = tile_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    val = int(tile_str)
    return val, val


def convert(args):
    print(f"[1/4] Loading weights from {args.input} ...")
    state_dict = load_pth(args.input)

    # Detect or override parameters
    detected = detect_params_from_state_dict(state_dict)
    if args.scale is not None:
        detected["scale"] = args.scale
    if args.num_block is not None:
        detected["num_block"] = args.num_block

    print(f"       Model params: scale={detected['scale']}, num_block={detected['num_block']}, "
          f"num_feat={detected['num_feat']}")

    print("[2/4] Building RRDBNet and loading weights ...")
    model = RRDBNet(**detected)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tile_w, tile_h = parse_tile_size(args.tile)
    print(f"[3/4] Tracing model with tile size {tile_w}×{tile_h} ...")
    dummy = torch.randn(1, 3, tile_h, tile_w)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    print("[4/4] Converting to Core ML ...")
    scale = detected["scale"]

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
        # Raw tensor output — the model outputs floats in [0, 1].
        # We handle scaling to [0, 255] in the upscale script.
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16 if not args.fp32 else ct.precision.FLOAT32,
    )

    # Store scale in user-defined metadata so upscale_folder can infer it
    spec = mlmodel.get_spec()  # type: ignore[union-attr]
    spec.description.metadata.userDefined["esrgan.scale"] = str(scale)
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)  # type: ignore[union-attr]

    output_path = args.output
    print(f"       Saving to {output_path} ...")
    mlmodel.save(output_path)
    print(f"✅  Done! Core ML model saved (tile={tile_w}×{tile_h}, scale={scale}).")


def main():
    parser = argparse.ArgumentParser(description="Convert ESRGAN .pth → Core ML .mlpackage")
    parser.add_argument("--input", "-i", required=True, help="Path to .pth model")
    parser.add_argument("--output", "-o", default="esrgan.mlpackage", help="Output .mlpackage path")
    parser.add_argument("--scale", type=int, default=None, help="Override upscale factor (2 or 4)")
    parser.add_argument("--num_block", type=int, default=None, help="Override number of RRDB blocks")
    parser.add_argument("--tile", default="256", help="Tile size: single int (e.g. 256) or WxH (e.g. 896x640)")
    parser.add_argument("--fp32", action="store_true", help="Use float32 precision (default: float16)")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
