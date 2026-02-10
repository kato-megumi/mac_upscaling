# macOS Image Upscaler (ESRGAN → Core ML)

Upscale images on macOS using ESRGAN models, accelerated via Core ML (GPU / Apple Neural Engine).

## Setup

```bash
cd /path/to/mac_upscaling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Download an ESRGAN model

Download a `.pth` model, for example from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases):

```bash
# 4× general photo upscaler
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

## Step 2: Convert .pth → Core ML

```bash
python convert_model.py --input RealESRGAN_x4plus.pth --output esrgan_x4.mlpackage --tile 256
```

Options:
- `--scale` — override upscale factor (auto-detected from weights)
- `--num_block` — override RRDB block count (auto-detected)
- `--tile` — tile size for model tracing (default: 256)
- `--fp32` — use float32 instead of float16

## Step 3: Upscale images

```bash
python upscale_folder.py \
    --model esrgan_x4.mlpackage \
    --input ./low_res \
    --output ./high_res \
    --tile 256 \
    --scale 4
```

Options:
- `--tile` — must match the tile size used during conversion
- `--scale` — upscale factor (default: 4)
- `--padding` — tile overlap in pixels to avoid seam artifacts (default: 10)
- `--compute_units` — `ALL` | `CPU_AND_GPU` | `CPU_AND_NE` | `CPU_ONLY`

## Supported image formats

PNG, JPEG, BMP, TIFF, WebP

## Notes

- **Tile-based processing** — images of any size are split into tiles, upscaled individually, and reassembled. This keeps memory usage bounded.
- **Float16 by default** — the Core ML model uses float16 for ~2× faster inference on GPU/ANE. Use `--fp32` in conversion if you need higher precision.
- **First run** — the first prediction may be slow as Core ML compiles the model for your hardware.
