# ESRGAN → Core ML: Complete Research Findings

## Table of Contents
1. [RRDBNet Architecture (Full Code)](#1-rrdbnet-architecture)
2. [Loading .pth Models in PyTorch](#2-loading-pth-models)
3. [Converting PyTorch ESRGAN to Core ML](#3-converting-to-core-ml)
4. [Core ML Inference in Python on macOS](#4-core-ml-inference-in-python)
5. [Required pip Packages](#5-required-pip-packages)
6. [Gotchas & Important Notes](#6-gotchas--important-notes)

---

## 1. RRDBNet Architecture

The full architecture from `basicsr.archs.rrdbnet_arch` (used by both ESRGAN and Real-ESRGAN):

```python
import torch
from torch import nn as nn
from torch.nn import functional as F


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle (inverse of PixelShuffle).
    
    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
        
    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    
    Used in RRDB block in ESRGAN.
    
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    
    Used in RRDB-Net in ESRGAN.
    
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN.
    
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    
    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle) to reduce
    the spatial size and enlarge the channel size before feeding inputs into the main
    ESRGAN architecture.
    
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the trunk network. Default: 23.
        num_grow_ch (int): Channels for each growth. Default: 32.
        scale (int): Upsampling factor. Default: 4.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
```

### Standard Model Parameters

| Model                           | num_in_ch | num_out_ch | num_feat | num_block | num_grow_ch | scale |
|---------------------------------|-----------|------------|----------|-----------|-------------|-------|
| RealESRGAN_x4plus               | 3         | 3          | 64       | 23        | 32          | 4     |
| RealESRNet_x4plus               | 3         | 3          | 64       | 23        | 32          | 4     |
| RealESRGAN_x4plus_anime_6B      | 3         | 3          | 64       | **6**     | 32          | 4     |
| RealESRGAN_x2plus               | 3         | 3          | 64       | 23        | 32          | **2** |
| Original ESRGAN (RRDB_ESRGAN)   | 3         | 3          | 64       | 23        | 32          | 4     |

- **num_in_ch**: Input channels (3 = RGB)
- **num_out_ch**: Output channels (3 = RGB)
- **num_feat**: 64 (always for standard models)
- **num_block**: 23 for full models, 6 for anime/lightweight
- **num_grow_ch**: 32 (always for standard models)
- **scale**: 4 for x4 models, 2 for x2 models

---

## 2. Loading .pth Models in PyTorch

### Key Pattern for Loading Weights

```python
import torch

# 1. Create the model architecture (must match the .pth file)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# 2. Load the state dict from .pth file
loadnet = torch.load('RealESRGAN_x4plus.pth', map_location=torch.device('cpu'))

# 3. Real-ESRGAN .pth files may store weights under different keys
if 'params_ema' in loadnet:
    keyname = 'params_ema'
elif 'params' in loadnet:
    keyname = 'params'
else:
    keyname = None  # weights are at top level (old ESRGAN format)

if keyname:
    model.load_state_dict(loadnet[keyname], strict=True)
else:
    model.load_state_dict(loadnet, strict=True)

# 4. Set to evaluation mode
model.eval()
```

### Old ESRGAN Format Conversion
Old ESRGAN `.pth` files (e.g. `RRDB_ESRGAN_x4.pth`) use different key names. The key mapping:
- `RDB` → `rdb`
- `RRDB_trunk` → `body`
- `trunk_conv` → `conv_body`
- `upconv` → `conv_up`
- `HRconv` → `conv_hr`

BasicSR has a conversion script for this (`scripts/model_conversion/convert_models.py`).

---

## 3. Converting PyTorch ESRGAN to Core ML

### Step-by-Step Process

```python
import torch
import coremltools as ct

# ----- Step 1: Create and load the PyTorch model -----
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

loadnet = torch.load('RealESRGAN_x4plus.pth', map_location='cpu')
keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
model.load_state_dict(loadnet[keyname], strict=True)
model.eval()

# ----- Step 2: Trace with torch.jit.trace -----
# Use a dummy input with the desired input resolution
# For a model that upscales x4: input 128x128 → output 512x512
example_input = torch.rand(1, 3, 128, 128)

with torch.no_grad():
    traced_model = torch.jit.trace(model, example_input)

# ----- Step 3: Convert to Core ML -----
# Option A: Fixed input shape (simplest, best performance)
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape, name="input")],
    outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
)

# Option B: Flexible input shape with RangeDim
input_shape = ct.Shape(
    shape=(
        1,
        3,
        ct.RangeDim(lower_bound=32, upper_bound=1024, default=128),
        ct.RangeDim(lower_bound=32, upper_bound=1024, default=128),
    )
)
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape, name="input")],
    outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
)

# Option C: Enumerated shapes (best performance for specific sizes)
input_shape = ct.EnumeratedShapes(
    shapes=[[1, 3, 64, 64], [1, 3, 128, 128], [1, 3, 256, 256]],
    default=[1, 3, 128, 128]
)
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape, name="input")],
    outputs=[ct.TensorType(name="output")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
)

# ----- Step 4: Save -----
coreml_model.save("RealESRGAN_x4plus.mlpackage")
```

### Using ImageType for Input/Output

```python
# With ImageType - input and output are images directly
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        name="input_image",
        shape=(1, 3, 128, 128),        # (batch, C, H, W) for PyTorch
        color_layout=ct.colorlayout.RGB,
        scale=1.0 / 255.0,             # ESRGAN expects [0,1] range
    )],
    outputs=[ct.ImageType(
        name="output_image",
        color_layout=ct.colorlayout.RGB,
    )],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
)
```

### Input Shape Recommendations

| Approach            | Pros                                                     | Cons                                            |
|---------------------|----------------------------------------------------------|-------------------------------------------------|
| **Fixed shape**     | Best performance, simplest                               | Only one resolution                             |
| **EnumeratedShapes**| Good performance, multiple sizes                         | Limited to predefined sizes (up to 128)         |
| **RangeDim**        | Most flexible, any size in range                         | Slightly slower, no unbounded in mlprogram      |

**Recommendation**: For super-resolution, use **fixed shape** or **EnumeratedShapes** with a few common tile sizes (e.g., 128, 256, 512). Process large images by tiling.

---

## 4. Core ML Inference in Python on macOS

### Basic Prediction with TensorType (MLMultiArray)

```python
import coremltools as ct
import numpy as np
from PIL import Image

# Load the Core ML model
model = ct.models.MLModel("RealESRGAN_x4plus.mlpackage")

# Load and preprocess an image
img = Image.open("input.png").convert("RGB")
img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1] range

# Convert to (1, 3, H, W) for PyTorch-style models
img_tensor = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dim

# Predict
output = model.predict({"input": img_tensor})

# Post-process output
output_array = output["output"]  # shape: (1, 3, H*scale, W*scale)
output_array = np.squeeze(output_array, axis=0)  # Remove batch dim
output_array = np.transpose(output_array, (1, 2, 0))  # CHW → HWC
output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)

# Save
result_img = Image.fromarray(output_array)
result_img.save("output.png")
```

### Prediction with ImageType

```python
import coremltools as ct
from PIL import Image

# Load model (converted with ImageType input/output)
model = ct.models.MLModel("RealESRGAN_x4plus.mlpackage")

# Load image - PIL Image can be passed directly when using ImageType
img = Image.open("input.png").convert("RGB")
img = img.resize((128, 128))  # Must match the model's expected input size

# Predict - returns a PIL Image when output is ImageType
output = model.predict({"input_image": img})
result = output["output_image"]  # This is a PIL Image

# Save
result.save("output.png")
```

### Batch Processing a Folder of Images

```python
import os
import numpy as np
import coremltools as ct
from PIL import Image


def upscale_folder(input_dir, output_dir, model_path, tile_size=128, scale=4):
    """Process all images in a folder using Core ML ESRGAN model."""
    
    os.makedirs(output_dir, exist_ok=True)
    model = ct.models.MLModel(model_path)
    
    supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    for filename in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_exts:
            continue
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        
        print(f"Processing: {filename}")
        
        # Load image
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # For small images, process directly
        h, w = img_np.shape[:2]
        
        # Convert to NCHW
        img_tensor = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]
        
        # Predict
        output = model.predict({"input": img_tensor})
        result = output["output"]
        
        # Post-process
        result = np.squeeze(result, axis=0)
        result = np.transpose(result, (1, 2, 0))
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        Image.fromarray(result).save(output_path)
        print(f"  Saved: {output_path} ({w}x{h} → {w*scale}x{h*scale})")
    
    print("Done!")


# Usage:
upscale_folder(
    input_dir="./input_images",
    output_dir="./output_images",
    model_path="RealESRGAN_x4plus.mlpackage",
    scale=4,
)
```

### Tiled Processing for Large Images

```python
def upscale_with_tiles(img_np, model, tile_size=128, tile_pad=10, scale=4):
    """
    Process a large image by splitting into tiles.
    
    Args:
        img_np: numpy array of shape (H, W, 3), float32, range [0, 1]
        model: loaded Core ML model
        tile_size: size of each tile (before upscaling)
        tile_pad: padding to avoid border artifacts
        scale: upscaling factor
    """
    h, w, c = img_np.shape
    output = np.zeros((h * scale, w * scale, c), dtype=np.float32)
    
    tiles_y = (h + tile_size - 1) // tile_size
    tiles_x = (w + tile_size - 1) // tile_size
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Input tile boundaries
            y_start = ty * tile_size
            x_start = tx * tile_size
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            # Add padding
            y_start_pad = max(y_start - tile_pad, 0)
            x_start_pad = max(x_start - tile_pad, 0)
            y_end_pad = min(y_end + tile_pad, h)
            x_end_pad = min(x_end + tile_pad, w)
            
            # Extract tile
            tile = img_np[y_start_pad:y_end_pad, x_start_pad:x_end_pad, :]
            
            # Pad to expected size if needed (for fixed-size models)
            th, tw = tile.shape[:2]
            # ... pad tile to tile_size + 2*tile_pad if needed ...
            
            # Convert to NCHW
            tile_tensor = np.transpose(tile, (2, 0, 1))[np.newaxis, ...]
            
            # Predict
            result = model.predict({"input": tile_tensor})
            out_tile = result["output"]
            out_tile = np.squeeze(out_tile, axis=0)
            out_tile = np.transpose(out_tile, (1, 2, 0))
            
            # Remove padding from output
            out_y_start = (y_start - y_start_pad) * scale
            out_x_start = (x_start - x_start_pad) * scale
            out_y_end = out_y_start + (y_end - y_start) * scale
            out_x_end = out_x_start + (x_end - x_start) * scale
            
            out_tile_cropped = out_tile[out_y_start:out_y_end, out_x_start:out_x_end, :]
            
            # Place into output
            output[y_start*scale:y_end*scale, x_start*scale:x_end*scale, :] = out_tile_cropped
    
    return np.clip(output * 255.0, 0, 255).astype(np.uint8)
```

---

## 5. Required pip Packages

### Minimal (for conversion + inference)
```
pip install torch torchvision coremltools Pillow numpy
```

### With basicsr (for loading models using the official architecture)
```
pip install torch torchvision coremltools Pillow numpy basicsr
```

### With Real-ESRGAN package (includes RealESRGANer helper class)
```
pip install torch torchvision coremltools Pillow numpy realesrgan basicsr
```

### Package Purposes

| Package       | Purpose                                                                       |
|---------------|-------------------------------------------------------------------------------|
| `torch`       | PyTorch framework, loading .pth files, tracing models                         |
| `torchvision` | Sometimes needed as a torch dependency                                        |
| `coremltools` | Converting PyTorch → Core ML, running Core ML inference in Python             |
| `Pillow`      | Image loading/saving (PIL)                                                    |
| `numpy`       | Array manipulation for pre/post-processing                                    |
| `basicsr`     | Contains official `RRDBNet` architecture (`from basicsr.archs.rrdbnet_arch import RRDBNet`) |
| `realesrgan`  | Higher-level Real-ESRGAN utilities (RealESRGANer class, tiling, etc.)         |

### Version Notes
- **coremltools >= 7.0** recommended for `mlprogram` and macOS 13+ targets
- **torch >= 1.9** for `torch.jit.trace` compatibility
- **Python 3.8+** required for modern coremltools
- You **don't need** `basicsr` or `realesrgan` if you copy the `RRDBNet` architecture code directly (as shown in Section 1)

---

## 6. Gotchas & Important Notes

### Core ML Conversion Gotchas

1. **`F.interpolate` with `mode='nearest'`**: The RRDBNet `forward()` uses `F.interpolate(feat, scale_factor=2, mode='nearest')` for upsampling. This traces cleanly in most coremltools versions. However:
   - If you hit issues, try setting `recompute_scale_factor=True` explicitly
   - Older coremltools versions may need `align_corners` specified for `bilinear` mode

2. **`inplace=True` on LeakyReLU**: The architecture uses `nn.LeakyReLU(negative_slope=0.2, inplace=True)`. During tracing this is fine, but if you run into issues, set `inplace=False`.

3. **`pixel_unshuffle` for scale 1 and 2**: For x2 models, the input channels are multiplied by 4 via pixel unshuffle. For x1 models, channels are multiplied by 16. This operation uses `view` and `permute` which should trace correctly. **Make sure scale matches between the model architecture and the .pth weights.**

4. **Fixed vs Flexible Input Shapes**: 
   - **Fixed shapes** give the best Core ML performance (compiler can optimize)
   - **EnumeratedShapes** are the next best option (up to 128 shapes)
   - **RangeDim** with bounded ranges works but may be slower
   - **Unbounded ranges are NOT allowed** for `mlprogram` format
   - Use `ct.ReshapeFrequency.Infrequent` optimization hint for flexible shapes (iOS 17.4+)

5. **Output scaling**: ESRGAN outputs values in [0, 1] range (float32). When using `TensorType`, you'll need to multiply by 255 and clip in post-processing. When using `ImageType` for output, coremltools handles this automatically.

6. **Memory concerns for large images**: Core ML models with large tensor shapes can consume significant memory. Consider:
   - Processing images in tiles (128×128 or 256×256)
   - Using the tile padding technique to avoid seam artifacts

7. **`convert_to="mlprogram"` vs neural network**: 
   - `mlprogram` (.mlpackage) is the modern format, required for macOS 12+/iOS 15+
   - Neural network (.mlmodel) is the legacy format
   - `mlprogram` generally gives better performance on Apple Silicon

8. **Float16 precision**: ML Programs use float16 by default which is fine for ESRGAN. If you need float32:
   ```python
   coreml_model = ct.convert(
       traced_model,
       inputs=[ct.TensorType(shape=example_input.shape)],
       compute_precision=ct.precision.FLOAT32,
       convert_to="mlprogram",
   )
   ```

### Weight Loading Gotchas

1. **`params_ema` vs `params` vs raw state dict**: Real-ESRGAN saves weights under `params_ema` (exponential moving average, preferred) or `params`. Older ESRGAN models store the state dict directly at the top level.

2. **Old ESRGAN key names**: If you get key mismatch errors, the .pth might be in the old ESRGAN format with different naming (e.g., `RRDB_trunk` instead of `body`). Use the conversion mapping described in Section 2.

3. **`strict=True` is important**: Always use `strict=True` in `load_state_dict()` to catch architecture mismatches early.

4. **`model.eval()` is critical**: Always call `model.eval()` before tracing. This disables dropout and uses running stats for batch norm (though RRDBNet doesn't use batch norm, it's good practice).

### Core ML Inference Gotchas

1. **macOS only**: `model.predict()` in coremltools only works on macOS (requires the Core ML framework).

2. **Input format matters**:
   - With `TensorType`: pass a NumPy array in NCHW format (batch, channels, height, width)
   - With `ImageType`: pass a PIL Image directly

3. **First prediction is slow**: The first call to `predict()` compiles the model on-device. Subsequent calls are faster.

4. **Output names**: The output name depends on what you specified during conversion. Check with `model.output_description` or by inspecting the model spec.

### Performance Tips

1. **Use Apple Neural Engine (ANE)**: ML Programs on Apple Silicon can run on the ANE for massive speedup. Fixed input shapes help the ANE compiler.

2. **Compute units**: You can specify which hardware to use:
   ```python
   model = ct.models.MLModel("model.mlpackage", compute_units=ct.ComputeUnit.ALL)
   # Options: ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE
   ```

3. **Tile size tuning**: For the ANE, smaller tiles (128×128) may actually be faster than larger tiles due to memory constraints.
