#!/usr/bin/env python3
"""Upscale a folder of images using a Core ML ESRGAN model.

Usage:
    python upscale_folder.py --model esrgan.mlpackage --input ./low_res --output ./high_res [--tile 256] [--scale 4] [--padding 10]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Maps --format value → (file extension, Pillow format string, default save kwargs)
OUTPUT_FORMATS: dict[str, tuple[str, str, dict]] = {
    "webp":     (".webp", "WEBP", {"quality": 98, "method": 4}),
    "webp_ll":  (".webp", "WEBP", {"lossless": True}),
    "png":      (".png",  "PNG",  {}),
    "jpg":      (".jpg",  "JPEG", {"quality": 98, "subsampling": 0}),
    "jpeg":     (".jpg",  "JPEG", {"quality": 98, "subsampling": 0}),
}


def _save_image(img: Image.Image, path: Path, fmt: str, kwargs: dict) -> Path:
    """Save an image to disk (runs in a thread)."""
    img.save(path, fmt, **kwargs)
    return path


def load_coreml_model(model_path: str, compute_units: str = "ALL"):
    """Load a Core ML model."""
    cu_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    cu = cu_map.get(compute_units.upper(), ct.ComputeUnit.ALL)
    print(f"Loading Core ML model from {model_path} (compute_units={compute_units}) ...")
    model = ct.models.MLModel(model_path, compute_units=cu)
    return model


def infer_model_params(model) -> dict:
    """Infer tile_w, tile_h, and scale from a Core ML model's spec."""
    spec = model.get_spec()
    info: dict = {}

    # Get input tile size from imageType
    for inp in spec.description.input:
        if inp.type.HasField("imageType"):
            it = inp.type.imageType
            info["tile_w"] = int(it.width)
            info["tile_h"] = int(it.height)
            break

    # Try scale from user-defined metadata first
    meta = spec.description.metadata.userDefined
    if "esrgan.scale" in meta:
        info["scale"] = int(meta["esrgan.scale"])
    elif "tile_h" in info:
        # Infer scale from output shape vs input shape
        for out in spec.description.output:
            if out.type.HasField("multiArrayType"):
                shape = list(out.type.multiArrayType.shape)
                if len(shape) >= 2:
                    out_h = int(shape[-2])
                    info["scale"] = out_h // info["tile_h"]
                break
            if out.type.HasField("imageType"):
                info["scale"] = int(out.type.imageType.height) // info["tile_h"]
                break

    return info


def upscale_image(model, img: Image.Image, tile_w: int, tile_h: int, scale: int, padding: int) -> Image.Image:
    """Upscale a single image using tiling to handle arbitrary sizes.

    Args:
        tile_w: Tile width (pixels) the model expects.
        tile_h: Tile height (pixels) the model expects.
    """
    img = img.convert("RGB")
    w, h = img.size

    # Output canvas
    out_w, out_h = w * scale, h * scale
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # Calculate tiles (separate strides for width / height)
    stride_x = tile_w - 2 * padding
    stride_y = tile_h - 2 * padding
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"Padding ({padding}) is too large for tile size ({tile_w}×{tile_h}).")

    cols = max(1, (w + stride_x - 1) // stride_x)
    rows = max(1, (h + stride_y - 1) // stride_y)

    img_np = np.array(img, dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            # Input tile coordinates (with padding overlap)
            x0 = min(col * stride_x, max(0, w - tile_w))
            y0 = min(row * stride_y, max(0, h - tile_h))
            x1 = min(x0 + tile_w, w)
            y1 = min(y0 + tile_h, h)

            # Extract tile — pad to tile size if near edge
            tw = x1 - x0
            th = y1 - y0
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            tile[:th, :tw, :] = img_np[y0:y1, x0:x1, :]

            # Convert to PIL for CoreML (ensure exact tile dimensions)
            tile_pil = Image.fromarray(tile).resize((tile_w, tile_h), Image.Resampling.LANCZOS)

            # Run inference
            pred = model.predict({"input": tile_pil})

            # Get the output (first value if dict)
            out_tile = list(pred.values())[0] if isinstance(pred, dict) else pred

            # Convert output to numpy HWC float32 in [0, 255]
            if isinstance(out_tile, Image.Image):
                out_tile_np = np.array(out_tile.convert("RGB"), dtype=np.float32)
            else:
                out_tile_np = np.array(out_tile, dtype=np.float32)
                # Remove batch dim: (1, C, H, W) → (C, H, W)
                if out_tile_np.ndim == 4:
                    out_tile_np = out_tile_np[0]
                # CHW → HWC
                if out_tile_np.ndim == 3 and out_tile_np.shape[0] in (3, 4):
                    out_tile_np = out_tile_np.transpose(1, 2, 0)
                # Strip alpha if present
                if out_tile_np.shape[-1] == 4:
                    out_tile_np = out_tile_np[:, :, :3]
                # Scale [0, 1] → [0, 255] if values are in float range
                # Use mean < 1.5 to handle models whose output slightly exceeds 1.0
                if out_tile_np.mean() < 1.5:
                    out_tile_np = out_tile_np * 255.0
                out_tile_np = np.clip(out_tile_np, 0, 255)

            # Output tile coordinates
            ox0 = x0 * scale
            oy0 = y0 * scale
            ow = tw * scale
            oh = th * scale

            # Cropping region in output tile (account for padding)
            crop_x0 = (padding * scale) if col > 0 else 0
            crop_y0 = (padding * scale) if row > 0 else 0
            crop_x1 = ow - ((padding * scale) if col < cols - 1 else 0)
            crop_y1 = oh - ((padding * scale) if row < rows - 1 else 0)

            # Destination in output canvas
            dst_x0 = ox0 + crop_x0
            dst_y0 = oy0 + crop_y0
            dst_x1 = ox0 + crop_x1
            dst_y1 = oy0 + crop_y1

            # Clip to output bounds
            dst_x1 = min(dst_x1, out_w)
            dst_y1 = min(dst_y1, out_h)
            actual_w = dst_x1 - dst_x0
            actual_h = dst_y1 - dst_y0

            if actual_w <= 0 or actual_h <= 0:
                continue

            src_region = out_tile_np[crop_y0:crop_y0 + actual_h, crop_x0:crop_x0 + actual_w, :]
            output[dst_y0:dst_y1, dst_x0:dst_x1, :] += src_region
            weight[dst_y0:dst_y1, dst_x0:dst_x1, :] += 1.0

    # Average overlapping regions
    weight = np.maximum(weight, 1e-8)
    output = output / weight
    output = np.clip(output, 0, 255).astype(np.uint8)

    return Image.fromarray(output)


def process_folder(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather image files
    images = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No supported images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {input_dir}")

    # Load model and infer parameters
    model = load_coreml_model(args.model, args.compute_units)
    model_info = infer_model_params(model)

    tile_w = args.tile_w or model_info.get("tile_w")
    tile_h = args.tile_h or model_info.get("tile_h")
    scale = args.scale or model_info.get("scale")

    if not tile_w or not tile_h:
        print("ERROR: Could not infer tile size from model. Specify --tile.", file=sys.stderr)
        sys.exit(1)
    if not scale:
        print("ERROR: Could not infer scale from model. Specify --scale.", file=sys.stderr)
        sys.exit(1)

    print(f"Tile: {tile_w}×{tile_h}, Scale: {scale}× (padding: {args.padding})")

    # Output format
    ext, pil_fmt, save_kwargs = OUTPUT_FORMATS[args.format]
    if args.quality is not None and "quality" in save_kwargs:
        save_kwargs = {**save_kwargs, "quality": args.quality}
    print(f"Output: {pil_fmt} ({ext}), save options: {save_kwargs}")

    # Warm-up prediction
    print("Warming up model ...")
    dummy = Image.new("RGB", (tile_w, tile_h), (128, 128, 128))
    model.predict({"input": dummy})

    total_start = time.time()
    save_futures: list[tuple[Future, str, float]] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, img_path in enumerate(images, 1):
            # Wait for previous save to finish before reusing memory
            if save_futures:
                fut, name, t0 = save_futures[-1]
                fut.result()  # block until saved

            print(f"\n[{i}/{len(images)}] Processing {img_path.name} ...")
            start = time.time()

            img = Image.open(img_path)
            result = upscale_image(model, img, tile_w, tile_h, scale, args.padding)

            infer_elapsed = time.time() - start
            out_path = output_dir / f"{img_path.stem}{ext}"

            # Submit save to background thread
            future = pool.submit(_save_image, result, out_path, pil_fmt, save_kwargs)
            save_futures.append((future, out_path.name, infer_elapsed))
            print(f"  → {out_path.name} ({result.size[0]}×{result.size[1]}) "
                  f"inferred in {infer_elapsed:.1f}s, saving async ...")

        # Wait for all remaining saves
        for fut, name, _ in save_futures:
            fut.result()

    total_elapsed = time.time() - total_start
    print(f"\n✅  All done! {len(images)} images upscaled in {total_elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Upscale images using Core ML ESRGAN model")
    parser.add_argument("--model", "-m", required=True, help="Path to .mlpackage Core ML model")
    parser.add_argument("--input", "-i", required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", required=True, help="Output folder for upscaled images")
    parser.add_argument("--tile", default=None, help="Tile size: int or WxH (auto-inferred from model if omitted)")
    parser.add_argument("--scale", type=int, default=None, help="Upscale factor (auto-inferred from model if omitted)")
    parser.add_argument("--format", "-f", default="webp",
                        choices=list(OUTPUT_FORMATS.keys()),
                        help="Output format (default: webp lossy 98%%)")
    parser.add_argument("--quality", "-q", type=int, default=None,
                        help="Override quality for lossy formats (1-100)")
    parser.add_argument("--padding", type=int, default=10, help="Tile overlap padding in pixels (default: 10)")
    parser.add_argument("--compute_units", default="ALL",
                        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
                        help="Core ML compute units (default: ALL)")
    args = parser.parse_args()

    # Parse --tile into tile_w / tile_h
    args.tile_w = args.tile_h = None
    if args.tile:
        if "x" in args.tile.lower():
            parts = args.tile.lower().split("x")
            args.tile_w, args.tile_h = int(parts[0]), int(parts[1])
        else:
            args.tile_w = args.tile_h = int(args.tile)

    process_folder(args)


if __name__ == "__main__":
    main()
