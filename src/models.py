import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =========================================================
# time embedding
# =========================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B]  (float or int)
        return: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =========================================================
# RRDB blocks for image encoder G
# =========================================================
class DenseResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        growth_channels: int = 32,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.residual_scale * x5


class RRDB(nn.Module):
    def __init__(
        self,
        channels: int,
        growth_channels: int = 32,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.db1 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.db2 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.db3 = DenseResidualBlock(channels, growth_channels, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + self.residual_scale * out


class ImageEncoderG(nn.Module):
    """
    SegDiff の画像エンコーダ G を意識した RRDB ベース実装
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 64,
        out_channels: int = 64,
        num_rrdb_blocks: int = 6,
        growth_channels: int = 32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.rrdb_body = nn.Sequential(
            *[RRDB(feat_channels, growth_channels=growth_channels) for _ in range(num_rrdb_blocks)]
        )
        self.body_conv = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv_out = nn.Conv2d(feat_channels, out_channels, 3, padding=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(img)
        trunk = self.rrdb_body(feat)
        trunk = self.body_conv(trunk)
        feat = feat + trunk
        feat = self.act(feat)
        out = self.conv_out(feat)
        return out


# =========================================================
# mask encoder F
# =========================================================
class MaskEncoderF(nn.Module):
    """
    SegDiff の F は浅い 2D Conv として実装
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.conv(x_t)


# =========================================================
# basic UNet blocks
# =========================================================
def group_norm(num_channels: int) -> nn.GroupNorm:
    # 小さいチャネル数でも動きやすいように調整
    groups = min(32, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x

        x = self.norm(x).view(b, c, h * w)       # [B, C, HW]
        qkv = self.qkv(x)                        # [B, 3C, HW]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        scale = head_dim ** -0.5
        attn = torch.einsum("bncd,bnce->bnde", q * scale, k)   # [B, heads, HW, HW]
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bnde,bnce->bncd", attn, v)         # [B, heads, dim, HW]
        out = out.reshape(b, c, h * w)
        out = self.proj_out(out).view(b, c, h, w)

        return x_in + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# =========================================================
# UNet trunk (E / D)
# =========================================================
class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_attention: bool = False,
        dropout: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class SegDiffUNet(nn.Module):
    """
    E/D 相当の U-Net trunk
    入力: fused features = F(x_t) + G(I)
    出力: predicted noise epsilon
    """
    def __init__(
        self,
        in_channels: int = 64,          # fusion後のチャネル数
        out_channels: int = 1,          # 予測ノイズのチャネル数（maskと同じ）
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_levels: Iterable[int] = (2, 3),  # 例: 16x16, 8x8 相当を想定
        dropout: float = 0.0,
        num_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels = []

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    UNetBlock(
                        ch, out_ch, time_emb_dim,
                        use_attention=(level in attn_levels),
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_blocks.append(blocks)

            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # middle
        self.mid1 = UNetBlock(ch, ch, time_emb_dim, use_attention=True, dropout=dropout, num_heads=num_heads)
        self.mid2 = UNetBlock(ch, ch, time_emb_dim, use_attention=False, dropout=dropout, num_heads=num_heads)

        # up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))
        skip_channels = list(reversed(self.skip_channels))

        for level, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop(0)
                blocks.append(
                    UNetBlock(
                        ch + skip_ch, out_ch, time_emb_dim,
                        use_attention=((len(channel_mults) - 1 - level) in attn_levels),
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
                ch = out_ch
            self.up_blocks.append(blocks)

            if level != len(reversed_mults) - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = group_norm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def _run_block(self, block, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, h, t_emb, use_reentrant=False)
        return block(h, t_emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = self._run_block(block, h, t_emb)
                skips.append(h)
            h = down(h)

        h = self._run_block(self.mid1, h, t_emb)
        h = self._run_block(self.mid2, h, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self._run_block(block, h, t_emb)
            h = up(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


class ResidualBlockNoTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)

class UNetBlockNoTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        dropout: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.res = ResidualBlockNoTime(in_channels, out_channels, dropout)
        self.attn = AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        x = self.attn(x)
        return x
    

class SourceUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 38,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_levels: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        num_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels = []

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    UNetBlockNoTime(
                        ch,
                        out_ch,
                        use_attention=(level in attn_levels),
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_blocks.append(blocks)

            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = UNetBlockNoTime(ch, ch, use_attention=True, dropout=dropout, num_heads=num_heads)
        self.mid2 = UNetBlockNoTime(ch, ch, use_attention=False, dropout=dropout, num_heads=num_heads)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))
        skip_channels = list(reversed(self.skip_channels))

        for level, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop(0)
                blocks.append(
                    UNetBlockNoTime(
                        ch + skip_ch,
                        out_ch,
                        use_attention=((len(channel_mults) - 1 - level) in attn_levels),
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
                ch = out_ch
            self.up_blocks.append(blocks)

            if level != len(reversed_mults) - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = group_norm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def _run_block(self, block, h: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, h, use_reentrant=False)
        return block(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)

        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = self._run_block(block, h)
                skips.append(h)
            h = down(h)

        h = self._run_block(self.mid1, h)
        h = self._run_block(self.mid2, h)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self._run_block(block, h)
            h = up(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h
# =========================================================
# full SegDiff model
# =========================================================
class SegDiffModel(nn.Module):
    """
    full model:
        x_t --F--> mask_feat
        I   --G--> image_feat
        fused = mask_feat + image_feat
        fused --UNet(E/D)--> epsilon_pred
    """
    def __init__(
        self,
        image_channels: int = 3,
        mask_channels: int = 19,
        fusion_channels: int = 64,
        rrdb_blocks: int = 6,
        rrdb_growth_channels: int = 32,
        unet_base_channels: int = 48,
        unet_channel_mults: Tuple[int, ...] = (1, 4, 6),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_levels: Iterable[int] = (3,),
        dropout: float = 0.0,
        num_heads: int = 4,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.mask_ch = mask_channels

        self.mask_encoder = MaskEncoderF(mask_channels, fusion_channels)


        self.image_encoder = ImageEncoderG(
            in_channels=image_channels,
            feat_channels=fusion_channels,
            out_channels=fusion_channels,
            num_rrdb_blocks=rrdb_blocks,
            growth_channels=rrdb_growth_channels,
        )

        self.unet = SegDiffUNet(
            in_channels=fusion_channels,
            out_channels=mask_channels,
            base_channels=unet_base_channels,
            channel_mults=unet_channel_mults,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            attn_levels=attn_levels,
            dropout=dropout,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x_t: torch.Tensor, img: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, mask_channels, H, W]   ノイズ付きマスク
        img: [B, image_channels, H, W]  条件画像
        t:   [B]                        時刻
        """
        mask_feat = self.mask_encoder(x_t)
        image_feat = self.image_encoder(img)
        fused = mask_feat + image_feat
        eps_pred = self.unet(fused, t)

        return eps_pred

class CSFMSourceGenerator(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        mask_channels: int = 19,
        fusion_channels: int = 64,
        rrdb_blocks: int = 8,
        rrdb_growth_channels: int = 32,
        source_base_channels: int = 64,
        source_channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_levels: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        num_heads: int = 4,
        use_checkpoint: bool = False,
        fixed_std: float | None = 1.0,
        logvar_min: float = -6.0,
        logvar_max: float = 3.0,
    ):
        super().__init__()
        self.mask_ch = mask_channels
        self.fixed_std = fixed_std
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.image_encoder = ImageEncoderG(
            in_channels=image_channels,
            feat_channels=fusion_channels,
            out_channels=fusion_channels,
            num_rrdb_blocks=rrdb_blocks,
            growth_channels=rrdb_growth_channels,
        )

        out_ch = mask_channels if fixed_std is not None else mask_channels * 2

        self.source_unet = SourceUNet(
            in_channels=fusion_channels,
            out_channels=out_ch,
            base_channels=source_base_channels,
            channel_mults=source_channel_mults,
            num_res_blocks=num_res_blocks,
            attn_levels=attn_levels,
            dropout=dropout,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, img: torch.Tensor):
        img_feat = self.image_encoder(img)
        out = self.source_unet(img_feat)

        if self.fixed_std is not None:
            mu = out
            logvar = torch.full_like(mu, math.log(self.fixed_std**2))
        else:
            mu, logvar = out.split(self.mask_ch, dim=1)
            logvar = logvar.clamp(self.logvar_min, self.logvar_max)

        eps = torch.randn_like(mu)
        z0 = mu + torch.exp(0.5 * logvar) * eps
        return z0, mu, logvar
