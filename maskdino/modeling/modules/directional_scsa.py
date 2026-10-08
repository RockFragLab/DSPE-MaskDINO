from collections import OrderedDict

import torch
from torch import nn


class SCSABlock(nn.Module):
    """Spatial-channel synergistic attention for one feature map."""

    def __init__(
        self,
        dim,
        heads=8,
        window_size=7,
        group_kernel_sizes=(3, 5, 7, 9),
        attn_drop=0.0,
        gate_layer="sigmoid",
    ):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"SCSA dim must be divisible by 4, got {dim}")
        if dim % heads != 0:
            raise ValueError(f"SCSA dim must be divisible by heads, got dim={dim}, heads={heads}")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.group_channels = dim // 4
        self.window_size = window_size

        group_channels = self.group_channels
        self.local_dwc = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_channels,
        )
        self.global_dwc_s = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_channels,
        )
        self.global_dwc_m = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_channels,
        )
        self.global_dwc_l = nn.Conv1d(
            group_channels,
            group_channels,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_channels,
        )

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.norm = nn.GroupNorm(1, dim)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def _downsample(self, x):
        if self.window_size <= 0:
            return nn.functional.adaptive_avg_pool2d(x, (1, 1))

        height, width = x.shape[-2:]
        if height < self.window_size or width < self.window_size:
            return nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return nn.functional.avg_pool2d(x, kernel_size=self.window_size, stride=self.window_size)

    def forward(self, x):
        b, c, height, width = x.shape

        x_h = x.mean(dim=3)
        h_chunks = torch.split(x_h, self.group_channels, dim=1)
        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(h_chunks[0]),
                        self.global_dwc_s(h_chunks[1]),
                        self.global_dwc_m(h_chunks[2]),
                        self.global_dwc_l(h_chunks[3]),
                    ),
                    dim=1,
                )
            )
        ).view(b, c, height, 1)

        x_w = x.mean(dim=2)
        w_chunks = torch.split(x_w, self.group_channels, dim=1)
        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(w_chunks[0]),
                        self.global_dwc_s(w_chunks[1]),
                        self.global_dwc_m(w_chunks[2]),
                        self.global_dwc_l(w_chunks[3]),
                    ),
                    dim=1,
                )
            )
        ).view(b, c, 1, width)

        x = x * x_h_attn * x_w_attn
        y = self.norm(self._downsample(x))
        _, _, y_height, y_width = y.shape

        q = self.q(y).reshape(b, self.heads, self.head_dim, y_height * y_width)
        k = self.k(y).reshape(b, self.heads, self.head_dim, y_height * y_width)
        v = self.v(y).reshape(b, self.heads, self.head_dim, y_height * y_width)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = attn.reshape(b, c, y_height, y_width)
        attn = self.ca_gate(attn.mean((2, 3), keepdim=True))
        return attn * x


class DirectionalEnhanceBlock(nn.Module):
    """Dynamic horizontal and vertical context enhancement."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.height_conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.width_conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )

    def forward(self, x):
        height_context = x.mean(dim=3)
        height_context = self.height_conv(height_context).unsqueeze(3).expand_as(x)
        width_context = x.mean(dim=2)
        width_context = self.width_conv(width_context).unsqueeze(2).expand_as(x)
        return self.fuse(torch.cat((height_context, width_context, x), dim=1))


class DirectionalSCSABlock(nn.Module):
    """Parallel directional enhancement and SCSA with a safe residual scale."""

    def __init__(
        self,
        dim,
        heads=8,
        window_size=7,
        direction_kernel_size=7,
        residual_init=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.direction = DirectionalEnhanceBlock(dim, kernel_size=direction_kernel_size)
        self.scsa = SCSABlock(dim, heads=heads, window_size=window_size, attn_drop=attn_drop)
        self.gamma = nn.Parameter(torch.full((1,), float(residual_init)))

    def forward(self, x):
        enhanced = self.direction(x) + self.scsa(x)
        return x + self.gamma * enhanced


class DirectionalSCSAFeatureEnhancer(nn.Module):
    """Apply DirectionalSCSABlock to projected pixel-decoder features."""

    def __init__(
        self,
        num_levels,
        dim,
        heads=8,
        window_size=7,
        direction_kernel_size=7,
        residual_init=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        blocks = OrderedDict()
        for level in range(num_levels):
            blocks[str(level)] = DirectionalSCSABlock(
                dim=dim,
                heads=heads,
                window_size=window_size,
                direction_kernel_size=direction_kernel_size,
                residual_init=residual_init,
                attn_drop=attn_drop,
            )
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, srcs):
        return [self.blocks[str(index)](src) for index, src in enumerate(srcs)]


def build_pixel_decoder_enhancer(cfg, num_levels, dim):
    if not cfg.MODEL.PIXEL_DECODER_ENHANCER.ENABLED:
        return nn.Identity()

    name = cfg.MODEL.PIXEL_DECODER_ENHANCER.NAME.lower()
    if name not in {"directionalscsa", "directional_scsa", "dssa"}:
        raise ValueError(f"Unsupported pixel decoder enhancer: {cfg.MODEL.PIXEL_DECODER_ENHANCER.NAME}")

    return DirectionalSCSAFeatureEnhancer(
        num_levels=num_levels,
        dim=dim,
        heads=cfg.MODEL.PIXEL_DECODER_ENHANCER.HEADS,
        window_size=cfg.MODEL.PIXEL_DECODER_ENHANCER.WINDOW_SIZE,
        direction_kernel_size=cfg.MODEL.PIXEL_DECODER_ENHANCER.DIRECTION_KERNEL_SIZE,
        residual_init=cfg.MODEL.PIXEL_DECODER_ENHANCER.RESIDUAL_INIT,
        attn_drop=cfg.MODEL.PIXEL_DECODER_ENHANCER.ATTN_DROP,
    )
