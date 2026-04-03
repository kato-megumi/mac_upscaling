"""GateRv3 architecture — standalone implementation.

Based on https://github.com/the-database/traiNNer-redux/blob/master/traiNNer/archs/gaterv3_arch.py
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SampleMods = Literal["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv"]

# ---------------------------------------------------------------------------
# Upsample helpers
# ---------------------------------------------------------------------------

class UniUpsample(nn.Sequential):
    """Universal upsample module — mirrors the one used in GateRv3."""

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

        samp_idx = list(SampleMods.__args__).index(upsample)  # type: ignore[union-attr]
        self.register_buffer(
            "MetaUpsample",
            torch.tensor([1, samp_idx, scale, in_dim, out_dim, mid_dim, group], dtype=torch.uint8),
        )


# ---------------------------------------------------------------------------
# SPAN re-parameterisable conv (Conv3XC)
# ---------------------------------------------------------------------------

class Conv3XC(nn.Module):
    def __init__(self, c_in: int, c_out: int, gain1: int = 1, s: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.bias = bias
        self.weight_concat: Tensor | None = None
        self.bias_concat: Tensor | None = None
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, kernel_size=1, padding=0, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        nn.init.trunc_normal_(self.sk.weight, std=0.02)
        if not self.training:
            self.eval_conv.weight.requires_grad = False
            if self.eval_conv.bias is not None:
                self.eval_conv.bias.requires_grad = False
            self.update_params()

    def update_params(self) -> None:
        w1 = cast(nn.Conv2d, self.conv[0]).weight.data.clone().detach()
        w2 = cast(nn.Conv2d, self.conv[1]).weight.data.clone().detach()
        w3 = cast(nn.Conv2d, self.conv[2]).weight.data.clone().detach()
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        sk_w = self.sk.weight.data.clone().detach()

        if self.bias:
            b1 = self.conv[0].bias.data.clone().detach()  # type: ignore[union-attr]
            b2 = self.conv[1].bias.data.clone().detach()  # type: ignore[union-attr]
            b3 = self.conv[2].bias.data.clone().detach()  # type: ignore[union-attr]
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
            self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
            sk_b = self.sk.bias.data.clone().detach()  # type: ignore[union-attr]

        h_pad = (3 - 1) // 2
        sk_w = F.pad(sk_w, [h_pad, h_pad, h_pad, h_pad])
        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat

        if self.bias:
            self.bias_concat = self.bias_concat + sk_b  # type: ignore[operator,possibly-undefined]
            self.eval_conv.bias.data = self.bias_concat  # type: ignore[union-attr]

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            return self.conv(x_pad) + self.sk(x)
        return self.eval_conv(x)


class SPAB(nn.Module):
    """SPAN Attention Block used as SISR branch in GateRv3."""

    def __init__(
        self,
        in_channels: int,
        mid_dim: int | None = None,
        out_dim: int | None = None,
        bias: bool = False,
        end: bool = False,
    ) -> None:
        super().__init__()
        mid_dim = mid_dim if mid_dim else in_channels
        out_dim = out_dim if out_dim else in_channels
        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_dim, gain1=2, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_dim, mid_dim, gain1=2, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_dim, out_dim, gain1=2, s=1, bias=bias)
        self.act1 = nn.SiLU(inplace=True)
        self.end = end

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


# ---------------------------------------------------------------------------
# GatedCNN building blocks (GateRv3 variant — flash/channel attention)
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
    def __init__(self, in_channels: int, square_kernel_size: int = 3, band_kernel_size: int = 11, branch_ratio: float = 0.125) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class ChannelAttention(nn.Module):
    """Channel self-attention used in GateRv3's GatedCNNBlock (flash or manual)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)
        # Use scaled_dot_product_attention for efficiency; fall back to manual otherwise
        out = F.scaled_dot_product_attention(
            q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3), is_causal=False
        )
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return self.project_out(out)


class GatedCNNBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
        att: bool = False,
        flash: bool = True,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1, 1)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.token_mix: nn.Module = ChannelAttention(conv_channels, 16) if att else InceptionDWConv2d(dim)
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
# GateRV3 main model
# ---------------------------------------------------------------------------

class GateRV3(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        dim: int = 32,
        enc_blocks: Sequence[int] = (2, 2, 4, 6),
        dec_blocks: Sequence[int] = (2, 2, 2, 2),
        num_latent: int = 8,
        scale: int = 2,
        upsample: SampleMods = "pixelshuffle",
        upsample_mid_dim: int = 48,
        end_gamma_init: int = 1,
        attention: bool = False,
        sisr_blocks: int = 4,
        flash: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)

        self.gater_encode = nn.ModuleList(
            [Block(dim * (2**i), enc_blocks[i]) for i in range(len(enc_blocks))]
        )

        # SISR (SPAN) branch
        self.span_block0 = SPAB(dim, end=False)
        self.span_n_b = nn.Sequential(*[SPAB(dim, end=False) for _ in range(sisr_blocks)])
        self.span_end = SPAB(dim, end=True)
        self.sisr_end_conv = Conv3XC(dim, dim, bias=True)
        self.sisr_cat_conv = nn.Conv2d(dim * 4, dim, 1)
        nn.init.trunc_normal_(self.sisr_cat_conv.weight, std=0.02)

        self.latent = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim * (2 ** len(enc_blocks)),
                    expansion_ratio=1.5,
                    conv_ratio=1.0,
                    att=attention,
                    flash=flash,
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

        self.gamma = nn.Parameter(torch.ones(1, in_ch, 1, 1) * end_gamma_init)

        if scale != 1:
            self.short_to_dim: nn.Module = nn.Upsample(scale_factor=scale)
            self.dim_to_in: nn.Module = UniUpsample(upsample, scale, dim, in_ch, upsample_mid_dim)
        else:
            self.dim_to_in = nn.Conv2d(dim, in_ch, 3, 1, 1)
            self.short_to_dim = nn.Identity()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        if "dim_to_in.MetaUpsample" in state_dict:
            state_dict["dim_to_in.MetaUpsample"] = self.dim_to_in.MetaUpsample  # type: ignore[union-attr]
        if "gamma" not in state_dict:
            state_dict["gamma"] = self.gamma
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

        # SISR branch
        sisr: Tensor = self.span_block0(x)  # type: ignore[assignment]
        sisr_short = sisr
        sisr = self.span_n_b(sisr)
        sisr_result = self.span_end(sisr)
        sisr_out: Tensor
        sisr, sisr_out = sisr_result  # type: ignore[misc]
        sisr = self.sisr_end_conv(sisr)
        sisr = self.sisr_cat_conv(torch.cat([x, sisr, sisr_short, sisr_out], dim=1))

        # U-Net encoder
        shorts: list[Tensor] = []
        for block in self.gater_encode:
            x, short = block(x)  # type: ignore[misc]
            shorts.append(short)

        x = self.latent(x)

        # U-Net decoder
        shorts.reverse()
        for index, dec_block in enumerate(self.decode):
            x = dec_block(x, shorts[index])  # type: ignore[assignment]

        x = self.dim_to_in(x + sisr) + self.gamma * self.short_to_dim(inp)
        return x[:, :, : h * self.scale, : w * self.scale]


# ---------------------------------------------------------------------------
# Auto-detection from state dict
# ---------------------------------------------------------------------------

def detect_gaterv3_params(state_dict: dict) -> dict:
    """Infer GateRv3 hyperparameters from a loaded state dict."""
    dim: int = state_dict["in_to_dim.weight"].shape[0]
    in_ch: int = state_dict["in_to_dim.weight"].shape[1]

    enc_blocks: list[int] = []
    i = 0
    while True:
        blocks = [int(k.split(".")[3]) for k in state_dict if k.startswith(f"gater_encode.{i}.gated.")]
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
    num_latent = max(latent_ids) + 1 if latent_ids else 8

    span_ids = [int(k.split(".")[1]) for k in state_dict if k.startswith("span_n_b.")]
    sisr_blocks = max(span_ids) + 1 if span_ids else 4

    if "dim_to_in.MetaUpsample" in state_dict:
        meta = state_dict["dim_to_in.MetaUpsample"].tolist()
        scale = int(meta[2])
    else:
        scale = 1

    # Detect whether latent blocks use channel attention or InceptionDW
    attention = any(k.startswith("latent.") and "token_mix.qkv." in k for k in state_dict)

    return {
        "in_ch": in_ch,
        "dim": dim,
        "enc_blocks": tuple(enc_blocks),
        "dec_blocks": tuple(dec_blocks),
        "num_latent": num_latent,
        "sisr_blocks": sisr_blocks,
        "scale": scale,
        "attention": attention,
    }


def is_gaterv3(state_dict: dict) -> bool:
    """Return True when the state dict looks like a GateRv3 model."""
    top_keys = {k.split(".")[0] for k in state_dict}
    required = {"in_to_dim", "gater_encode", "latent", "decode", "dim_to_in", "span_block0"}
    return required.issubset(top_keys)


def load_gaterv3(state_dict: dict) -> GateRV3:
    params = detect_gaterv3_params(state_dict)
    model = GateRV3(**params)
    model.load_state_dict(state_dict, strict=True)
    return model
