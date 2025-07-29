from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange
from model.vmamba.vmamba import VSSBlock, LayerNorm2d, Linear2d
from typing import Sequence, Type, Optional
import torch.nn.functional as F

class MultiScale(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
            for k in kernel_sizes
        ])

        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, len(kernel_sizes)*4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(len(kernel_sizes)*4, len(kernel_sizes),kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.channel_att = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, H, W = x.shape
        features = [conv(x) for conv in self.dw_convs]
        features = torch.stack(features, dim=1)  # [B, K, C, H, W]

        weights = self.weight_gen(x).view(B, len(self.kernel_sizes), 1, 1, 1)
        fused = (features * weights).sum(dim=1)

        channel_weight = self.channel_att(fused)
        spatial_weight = self.spatial_att(fused)
        refined = fused * channel_weight * spatial_weight

        return x + refined
class MSAC(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            drop: float = 0.,
            channels_first: bool = False,
            kernel_sizes: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if channels_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.dkf_conv =MultiScale(hidden_features, kernel_sizes)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.dkf_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EnhancedDWT(nn.Module):
    def __init__(self, wave_type='haar'):
        super(EnhancedDWT, self).__init__()
        self.wave_type = wave_type
        if wave_type == 'haar':
            self.h_lo = torch.tensor([0.7071067811865475, 0.7071067811865475]).view(1, 1, 1, 2)
            self.h_hi = torch.tensor([0.7071067811865475, -0.7071067811865475]).view(1, 1, 1, 2)
        elif wave_type == 'db2':
            self.h_lo = torch.tensor([-0.1294, 0.2241, 0.8365, 0.4830]).view(1, 1, 1, 4)
            self.h_hi = torch.tensor([-0.4830, 0.8365, -0.2241, -0.1294]).view(1, 1, 1, 4)
        elif wave_type == 'sym4':
            self.h_lo = torch.tensor([-0.0758, -0.0296, 0.4976, 0.8037, 0.2979, -0.0992, -0.0126, 0.0322]).view(1, 1, 1,                                                                                            8)
            self.h_hi = torch.tensor([-0.0322, -0.0126, 0.0992, 0.2979, -0.8037, 0.4976, 0.0296, -0.0758]).view(1, 1, 1,
                                                                                                                8)
        else:
            self.h_lo = torch.tensor([0.7071067811865475, 0.7071067811865475]).view(1, 1, 1, 2)
            self.h_hi = torch.tensor([0.7071067811865475, -0.7071067811865475]).view(1, 1, 1, 2)

        self.register_buffer('lo_filter', self.h_lo)
        self.register_buffer('hi_filter', self.h_hi)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h, pad_w = 0, 0
        if H % 2 != 0:
            pad_h = 1
        if W % 2 != 0:
            pad_w = 1

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H, W = x.shape

        lo_filter = self.lo_filter.repeat(C, 1, 1, 1).to(x.device)
        hi_filter = self.hi_filter.repeat(C, 1, 1, 1).to(x.device)

        lo_filter = lo_filter * self.alpha
        hi_filter = hi_filter * self.beta

        x_L = F.conv2d(x, lo_filter, stride=(1, 2), padding=(0, (lo_filter.shape[3] - 1) // 2), groups=C)
        x_H = F.conv2d(x, hi_filter, stride=(1, 2), padding=(0, (hi_filter.shape[3] - 1) // 2), groups=C)

        lo_filter_t = lo_filter.transpose(2, 3)
        hi_filter_t = hi_filter.transpose(2, 3)

        x_LL = F.conv2d(x_L, lo_filter_t, stride=(2, 1), padding=((lo_filter_t.shape[2] - 1) // 2, 0), groups=C)
        x_LH = F.conv2d(x_L, hi_filter_t, stride=(2, 1), padding=((hi_filter_t.shape[2] - 1) // 2, 0), groups=C)
        x_HL = F.conv2d(x_H, lo_filter_t, stride=(2, 1), padding=((lo_filter_t.shape[2] - 1) // 2, 0), groups=C)
        x_HH = F.conv2d(x_H, hi_filter_t, stride=(2, 1), padding=((hi_filter_t.shape[2] - 1) // 2, 0), groups=C)

        edge_weight = 1.2
        x_LH = x_LH * edge_weight
        x_HL = x_HL * edge_weight
        x_HH = x_HH * edge_weight

        return x_LL, x_LH, x_HL, x_HH

class EnhancedIDWT(nn.Module):
    def __init__(self, wave_type='haar'):
        super(EnhancedIDWT, self).__init__()
        self.wave_type = wave_type
        if wave_type == 'haar':
            self.g_lo = torch.tensor([0.7071067811865475, 0.7071067811865475]).view(1, 1, 1, 2)
            self.g_hi = torch.tensor([-0.7071067811865475, 0.7071067811865475]).view(1, 1, 1, 2)
        elif wave_type == 'db2':
            self.g_lo = torch.tensor([0.4830, 0.8365, 0.2241, -0.1294]).view(1, 1, 1, 4)
            self.g_hi = torch.tensor([-0.1294, -0.2241, 0.8365, -0.4830]).view(1, 1, 1, 4)
        elif wave_type == 'sym4':
            self.g_lo = torch.tensor([0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758]).view(1, 1, 1,
                                                                                                                8)
            self.g_hi = torch.tensor([-0.0758, 0.0296, -0.0992, -0.2979, 0.8037, -0.4976, -0.0126, -0.0322]).view(1, 1,
                                                                                                                  1, 8)
        else:
            self.g_lo = torch.tensor([0.7071067811865475, 0.7071067811865475]).view(1, 1, 1, 2)
            self.g_hi = torch.tensor([-0.7071067811865475, 0.7071067811865475]).view(1, 1, 1, 2)

        self.register_buffer('lo_filter', self.g_lo)
        self.register_buffer('hi_filter', self.g_hi)

        self.gamma = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))

    def forward(self, x_LL, x_LH, x_HL, x_HH):
        B, C, H, W = x_LL.shape

        lo_filter = self.lo_filter.repeat(C, 1, 1, 1).to(x_LL.device)
        hi_filter = self.hi_filter.repeat(C, 1, 1, 1).to(x_LL.device)

        lo_filter = lo_filter * self.gamma
        hi_filter = hi_filter * self.delta

        x_LL_up = F.interpolate(x_LL, scale_factor=2, mode='nearest')
        x_LH_up = F.interpolate(x_LH, scale_factor=2, mode='nearest')
        x_HL_up = F.interpolate(x_HL, scale_factor=2, mode='nearest')
        x_HH_up = F.interpolate(x_HH, scale_factor=2, mode='nearest')

        lo_filter_t = lo_filter.transpose(2, 3)
        hi_filter_t = hi_filter.transpose(2, 3)

        x_L = F.conv_transpose2d(x_LL_up, lo_filter_t, stride=(1, 1), padding=((lo_filter_t.shape[2] - 1) // 2, 0),
                                 groups=C)
        x_L = x_L + F.conv_transpose2d(x_LH_up, hi_filter_t, stride=(1, 1),
                                       padding=((hi_filter_t.shape[2] - 1) // 2, 0), groups=C)

        x_H = F.conv_transpose2d(x_HL_up, lo_filter_t, stride=(1, 1), padding=((lo_filter_t.shape[2] - 1) // 2, 0),
                                 groups=C)
        x_H = x_H + F.conv_transpose2d(x_HH_up, hi_filter_t, stride=(1, 1),
                                       padding=((hi_filter_t.shape[2] - 1) // 2, 0), groups=C)

        x = F.conv_transpose2d(x_L, lo_filter, stride=(1, 1), padding=(0, (lo_filter.shape[3] - 1) // 2), groups=C)
        x = x + F.conv_transpose2d(x_H, hi_filter, stride=(1, 1), padding=(0, (hi_filter.shape[3] - 1) // 2), groups=C)

        return x

class WBA(nn.Module):
    def __init__(self, dim, wave_dim=None, wave_type='db2'):
        super(WBA, self).__init__()
        wave_dim = wave_dim or dim // 2

        self.dwt = EnhancedDWT(wave_type=wave_type)
        self.idwt = EnhancedIDWT(wave_type=wave_type)

        self.low_freq_process = nn.Sequential(
            nn.Conv2d(dim, wave_dim, kernel_size=1), 
            nn.Conv2d(wave_dim, wave_dim, kernel_size=3, padding=1, groups=wave_dim),  
            nn.Conv2d(wave_dim, wave_dim, kernel_size=1), 
            LayerNorm2d(wave_dim),
            nn.GELU()
        )

        self.high_freq_process = nn.Sequential(
            nn.Conv2d(dim, wave_dim, kernel_size=1), 
            nn.Conv2d(wave_dim, wave_dim, kernel_size=3, padding=1, groups=wave_dim),  
            LayerNorm2d(wave_dim),
            nn.GELU()
        )

        self.band_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(wave_dim * 4, wave_dim * 4, kernel_size=1, groups=dim), 
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(wave_dim, dim, kernel_size=1)

        self.skip = nn.Identity() if dim == dim else nn.Conv2d(dim, dim, kernel_size=1)

        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim), 
            nn.Sigmoid()
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, groups=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = self.skip(x)
        B, C, H, W = x.shape

        LL, LH, HL, HH = self.dwt(x)

        LL_processed = self.low_freq_process(LL)

        LH_processed = self.high_freq_process(LH)
        HL_processed = self.high_freq_process(HL)
        HH_processed = self.high_freq_process(HH)

        all_bands = torch.cat([LL_processed, LH_processed, HL_processed, HH_processed], dim=1)
        band_weights = self.band_attention(all_bands)
        band_weights = torch.split(band_weights, LL_processed.size(1), dim=1)

        LL_processed = LL_processed * band_weights[0]
        LH_processed = LH_processed * band_weights[1]
        HL_processed = HL_processed * band_weights[2]
        HH_processed = HH_processed * band_weights[3]

        reconstructed = self.idwt(LL_processed, LH_processed, HL_processed, HH_processed)
        reconstructed = reconstructed[:, :, :H, :W]

        out = self.fusion(reconstructed)

        edge_weights = self.edge_enhancement(out)
        out = out * (1 + edge_weights)

        channel_weight = self.channel_att(out)
        out = out * channel_weight

        return out + identity


class FWVSS(nn.Sequential):
    def __init__(
            self,
            dim: int,
            depth: int,
            drop_path: Sequence[float] | float = 0.0,
            use_checkpoint: bool = False,
            norm_layer: Type[nn.Module] = LayerNorm2d,
            channel_first: bool = True,
            ssm_d_state: int = 1,
            ssm_ratio: float = 1.0,
            ssm_dt_rank: str = "auto",
            ssm_act_layer: Type[nn.Module] = nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias: bool = False,
            ssm_drop_rate: float = 0.0,
            ssm_init: str = "v0",
            forward_type: str = "v05_noz",
            mlp_ratio: float = 4.0,
            mlp_act_layer: Type[nn.Module] = nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MSAC
            ))
        super(FWVSS, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))

    def wba(self, x, wavelet_recon):
        return wavelet_recon(x)
class FAPE(nn.Module):
    """Frequency Adaptive Pixel Expansion) """

    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(FAPE, self).__init__()
        self.dim = dim

        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=5, padding=2, groups=dim * 2, bias=True)
        )

        self.freq_adapt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, groups=8),
            nn.Sigmoid()
        )

        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.expand(x)

        freq_weight = self.freq_adapt(feat)
        feat = feat * freq_weight

        feat = rearrange(feat, pattern="b c h w -> b h w c")
        B, H, W, C = feat.shape

        feat = rearrange(feat, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        feat = feat.view(B, -1, C // 4)
        feat = self.norm(feat)
        feat = feat.reshape(B, H * 2, W * 2, C // 4)

        feat = rearrange(feat, pattern="b h w c -> b c h w")
        return feat

class OutputExpander (nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(OutputExpander, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)


class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int,
            drop_path: Sequence[float] | float,
            use_wavelet: bool = True,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = FAPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = FWVSS(dim=out_channels, depth=depth, drop_path=drop_path)
        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.wavelet_recon = WBA(out_channels)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out)
        if self.use_wavelet:
            out = self.vss_layer.wba(out, self.wavelet_recon)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        num_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(
                UpBlock(
                    in_channels=dims[i - 1],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path=dpr[sum(depths[: i - 1]): sum(depths[: i])],
                    use_wavelet=True, 
                ))
        self.out_layers = nn.Sequential(OutputExpander(dims[-1], num_classes))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        out = features[0]
        features = features[1:]
        for i, layer in enumerate(self.layers):
            out = layer(out, features[i])
        return self.out_layers[0](out)
