"""GateRv2 architecture — standalone implementation.

Based on https://github.com/umzi2/GateRv2/blob/master/GateRv2.py
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SampleMods = Literal["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv"]

# ---------------------------------------------------------------------------
# Upsample helpers
# ---------------------------------------------------------------------------

class UniUpsample(nn.Sequential):
    """Universal upsample module — mirrors the one in GateRv2.py."""

    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,
        group: int = 4,
    ) -> None:
        m: list[nn.Module] = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend([nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)])
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend([nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)])
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(f"scale {scale} not supported")
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend([
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=2),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ])
                m.extend([
                    nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ])
            elif scale == 3:
                m.extend([
                    nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                    nn.Upsample(scale_factor=scale),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ])
            else:
                raise ValueError(f"scale {scale} not supported")
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(f"unsupported upsample: {upsample}")

        super().__init__(*m)

        # MetaUpsample buffer — encodes config for state-dict round-trips
        samp_idx = list(SampleMods.__args__).index(upsample)  # type: ignore[union-attr]
        self.register_buffer(
            "MetaUpsample",
            torch.tensor([1, samp_idx, scale, in_dim, out_dim, mid_dim, group], dtype=torch.uint8),
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-0.5))
        x_normed = x / (rms_x + self.eps)
        return self.scale[..., None, None] * x_normed + self.offset[..., None, None]


class InceptionDWConv2d(nn.Module):
    """Inception depthwise convolution."""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes: list[int] = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


def _l2_norm(x: Tensor) -> Tensor:
    return torch.einsum("bcn,bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class Attention(nn.Module):
    """L2 self-attention used in GateRv2."""

    def __init__(self, in_places: int, scale: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        self.in_places = in_places
        self.eps = eps
        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = _l2_norm(Q).permute(-3, -1, -2)
        K = _l2_norm(K)
        tailor_sum = 1 / (width * height + torch.einsum("bnc,bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum("bmn,bcn->bmc", K, V)
        matrix_sum = value_sum + torch.einsum("bnm,bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn,bn->bcn", matrix_sum, tailor_sum)
        return weight_value.view(batch_size, chnnels, height, width).contiguous()


class GatedCNNBlock(nn.Module):
    """Modernised MambaOut block — https://github.com/yuweihao/MambaOut."""

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
        att: bool = False,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1, 1)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.token_mix = Attention(conv_channels, 16) if att else InceptionDWConv2d(dim)
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1, 0)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.token_mix(c)
        x = self.act(g) * torch.cat((i, c), dim=1)
        return self.act(self.fc2(x))


class SimpleGate(nn.Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MetaGated(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim * 2
        self.local = nn.Sequential(
            RMSNorm(dim),
            nn.Conv2d(dim, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=dim),
            SimpleGate(),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
        )
        self.glob = GatedCNNBlock(dim)
        self.gamma0 = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        short = x
        x = self.local(x)
        x = x * self.sca(x)
        x = x * self.gamma0 + short
        x = self.glob(x) * self.gamma1 + x
        return x


class Down(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )


class UpShuffle(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )


class Block(nn.Module):
    def __init__(self, dim: int, num_gated: int, down: bool = True) -> None:
        super().__init__()
        if down:
            self.gated = nn.Sequential(*[MetaGated(dim) for _ in range(num_gated)])
            self.scale = Down(dim)
        else:
            self.scale = UpShuffle(dim)
            self.gated = nn.Sequential(*[MetaGated(dim // 2) for _ in range(num_gated)])
            self.shor = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.down = down

    def forward(self, x: Tensor, short: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        if self.down:
            x = self.gated(x)
            return self.scale(x), x
        else:
            assert short is not None
            x = torch.cat([self.scale(x), short], dim=1)
            x = self.shor(x)
            return self.gated(x)


# ---------------------------------------------------------------------------
# GateRV2 main model
# ---------------------------------------------------------------------------

class GateRV2(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 60,
        enc_blocks: tuple[int, ...] = (2, 2, 4, 6),
        dec_blocks: tuple[int, ...] = (2, 2, 2, 2),
        num_latent: int = 10,
        scale: int = 1,
        upsample: SampleMods = "pixelshuffledirect",
        upsample_mid_dim: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)

        self.encode = nn.ModuleList(
            [Block(dim * (2**i), enc_blocks[i]) for i in range(len(enc_blocks))]
        )
        self.latent = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim * (2 ** len(enc_blocks)),
                    expansion_ratio=1.5,
                    conv_ratio=1.0,
                    att=True,
                )
                for _ in range(num_latent)
            ]
        )
        self.decode = nn.ModuleList(
            [
                Block(
                    dim * (2 ** (len(dec_blocks) - i)),
                    dec_blocks[i],
                    False,
                )
                for i in range(len(dec_blocks))
            ]
        )
        self.pad = 2 ** len(enc_blocks)

        if scale != 1:
            self.short_to_dim: nn.Module = nn.Upsample(scale_factor=scale)
            self.dim_to_in: nn.Module = UniUpsample(upsample, scale, dim, in_ch, upsample_mid_dim)
        else:
            self.dim_to_in = nn.Conv2d(dim, in_ch, 3, 1, 1)
            self.short_to_dim = nn.Identity()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        if "dim_to_in.MetaUpsample" in state_dict:
            state_dict["dim_to_in.MetaUpsample"] = self.dim_to_in.MetaUpsample  # type: ignore[union-attr]
        return super().load_state_dict(state_dict, *args, **kwargs)

    def check_img_size(self, x: Tensor, resolution: tuple[int, int]) -> Tensor:
        scaled_size = self.pad
        mod_pad_h = (scaled_size - resolution[0] % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - resolution[1] % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, inp: Tensor) -> Tensor:
        _b, _c, h, w = inp.shape
        inp = self.check_img_size(inp, (h, w))
        x = self.in_to_dim(inp)
        shorts: list[Tensor] = []
        for block in self.encode:
            x, short = block(x)  # type: ignore[misc]
            shorts.append(short)
        x = self.latent(x)
        shorts.reverse()
        for index, dec_block in enumerate(self.decode):
            x = dec_block(x, shorts[index])  # type: ignore[assignment]
        x = self.dim_to_in(x) + self.short_to_dim(inp)
        return x[:, :, : h * self.scale, : w * self.scale]


# ---------------------------------------------------------------------------
# Auto-detection from state dict
# ---------------------------------------------------------------------------

def detect_gaterv2_params(state_dict: dict) -> dict:
    """Infer GateRv2 hyperparameters from a loaded state dict."""
    dim: int = state_dict["in_to_dim.weight"].shape[0]
    in_ch: int = state_dict["in_to_dim.weight"].shape[1]

    enc_blocks: list[int] = []
    i = 0
    while True:
        blocks = [int(k.split(".")[3]) for k in state_dict if k.startswith(f"encode.{i}.gated.")]
        if not blocks:
            break
        enc_blocks.append(max(blocks) + 1)
        i += 1

    dec_blocks: list[int] = []
    i = 0
    while True:
        blocks = [int(k.split(".")[3]) for k in state_dict if k.startswith(f"decode.{i}.gated.")]
        if not blocks:
            break
        dec_blocks.append(max(blocks) + 1)
        i += 1

    latent_ids = [int(k.split(".")[1]) for k in state_dict if k.startswith("latent.")]
    num_latent = max(latent_ids) + 1 if latent_ids else 10

    if "dim_to_in.MetaUpsample" in state_dict:
        meta = state_dict["dim_to_in.MetaUpsample"].tolist()
        scale = int(meta[2])
    else:
        scale = 1  # plain Conv2d → no upscaling

    return {
        "in_ch": in_ch,
        "dim": dim,
        "enc_blocks": tuple(enc_blocks),
        "dec_blocks": tuple(dec_blocks),
        "num_latent": num_latent,
        "scale": scale,
    }


def is_gaterv2(state_dict: dict) -> bool:
    """Return True when the state dict looks like a GateRv2 model."""
    top_keys = {k.split(".")[0] for k in state_dict}
    required = {"in_to_dim", "encode", "latent", "decode", "dim_to_in"}
    gaterv3_exclusive = {"gater_encode", "span_block0", "gamma"}
    return required.issubset(top_keys) and not gaterv3_exclusive.intersection(top_keys)


def load_gaterv2(state_dict: dict) -> GateRV2:
    params = detect_gaterv2_params(state_dict)
    model = GateRV2(**params)
    model.load_state_dict(state_dict, strict=True)
    return model
