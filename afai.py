import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

class FrequencyDecomposer(nn.Module):
    """平衡的频域分解器 - 将特征分解为低频和高频成分"""

    def __init__(self, dim: int):
        super().__init__()

        # 适中大小的卷积核低通滤波器
        self.low_pass = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        # 平衡的拉普拉斯算子
        self.laplacian_kernel = nn.Parameter(
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float().view(1, 1, 3, 3) / 8.0,
            requires_grad=True  # 允许核参数学习，但初始化较为保守
        )

        # 平衡的频率增强因子，从两个版本中找到中间值
        self.freq_scale = nn.Parameter(torch.tensor([1.0, 1.0]))  # 开始时平等处理高低频

        # 添加数据集自适应层，用于动态调整频率增强
        self.adaptive_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 低频分量 - 使用卷积
        low_freq = self.low_pass(x)

        # 高频分量 - 使用拉普拉斯算子
        batch, channels = x.shape[:2]
        weight = self.laplacian_kernel.to(x.device).repeat(channels, 1, 1, 1)
        high_freq = F.conv2d(x, weight, padding=1, groups=channels)

        # 计算自适应频率缩放 - 修改为可广播的形状
        adaptive_weights = self.adaptive_scale(x).squeeze(-1).squeeze(-1).view(batch, 2)

        # 应用频率增强因子和自适应缩放 - 使用可广播的形式
        low_freq = low_freq * (self.freq_scale[0] + adaptive_weights[:, 0].view(batch, 1, 1, 1))
        high_freq = high_freq * (self.freq_scale[1] + adaptive_weights[:, 1].view(batch, 1, 1, 1))

        return low_freq, high_freq


class FrequencyInteractionModule(nn.Module):
    """平衡的频率交互模块"""

    def __init__(self, dim: int):
        super().__init__()

        # 频域分解器
        self.freq_decomposer = FrequencyDecomposer(dim)

        # 平衡的频率交互矩阵初始值
        self.freq_interaction = nn.Parameter(torch.tensor([[0.7, 0.3], [0.3, 0.7]]))
        self.softmax = nn.Softmax(dim=1)

        # 中等复杂度的通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 4, 8), kernel_size=1),  # 轻量级瓶颈层
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // 4, 8), dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 精简的空间注意力 - 仅用于突出重要特征
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 效率优化的融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 频域分解
        low_freq, high_freq = self.freq_decomposer(x)

        # 频率交互
        interaction_weights = self.softmax(self.freq_interaction)

        # 应用频率交互
        low_enhanced = low_freq + high_freq * interaction_weights[0, 1]
        high_enhanced = high_freq + low_freq * interaction_weights[1, 0]

        # 应用通道注意力
        low_att = self.channel_att(low_enhanced)
        high_att = self.channel_att(high_enhanced)

        low_enhanced = low_enhanced * low_att
        high_enhanced = high_enhanced * high_att

        # 融合频率成分
        freq_fused = self.fusion(torch.cat([low_enhanced, high_enhanced], dim=1))

        # 可选的空间注意力 - 仅用于微调
        spatial_weight = self.spatial_att(freq_fused)
        freq_fused = freq_fused * spatial_weight

        # 残差连接
        return freq_fused + x


class CrossScaleAttention(nn.Module):
    """平衡的跨尺度注意力模块"""

    def __init__(self, c_list: List[int]):
        super().__init__()

        # 平衡的投影层 - 部分使用分组卷积减少参数
        self.projections = nn.ModuleList()
        for i, c_target in enumerate(c_list):
            proj_layers = nn.ModuleList()
            for j, c_source in enumerate(c_list):
                if i != j:
                    # 对于较大通道数使用分组卷积
                    groups = 1
                    if c_source > 64 and c_target > 64:
                        groups = 4

                    proj_layers.append(nn.Sequential(
                        nn.Conv2d(c_source, c_target, kernel_size=1, groups=groups),
                        nn.BatchNorm2d(c_target)
                    ))
                else:
                    proj_layers.append(None)
            self.projections.append(proj_layers)

        # 计算每个尺度的其他特征总通道数
        self.cross_channels = []
        for i, c_target in enumerate(c_list):
            other_channels = sum([c_target for j, c in enumerate(c_list) if i != j])
            self.cross_channels.append(other_channels)

        # 平衡的注意力门控
        self.gates = nn.ModuleList()
        for i, c in enumerate(c_list):
            bottleneck = max(c // 4, 8)  # 平衡的瓶颈大小
            self.gates.append(nn.Sequential(
                nn.Conv2d(self.cross_channels[i], bottleneck, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck, c, kernel_size=1),
                nn.Sigmoid()
            ))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        enhanced_features = []

        for i, feat in enumerate(features):
            # 收集其他尺度特征
            other_feats = []
            for j, other_feat in enumerate(features):
                if i != j:
                    # 使用自适应插值 - 根据输入大小选择合适的模式
                    h, w = feat.shape[2:]
                    if h <= 16 or w <= 16:  # 对于小尺寸特征图使用最近邻插值
                        aligned = F.interpolate(other_feat, size=(h, w), mode='nearest')
                    else:  # 对于较大尺寸特征图使用双线性插值
                        aligned = F.interpolate(other_feat, size=(h, w), mode='bilinear', align_corners=False)

                    projected = self.projections[i][j](aligned)
                    other_feats.append(projected)

            # 融合跨尺度信息
            if other_feats:
                cross_info = torch.cat(other_feats, dim=1) if len(other_feats) > 1 else other_feats[0]
                gate = self.gates[i](cross_info)
                # 平衡的融合方式
                enhanced = feat + feat * gate
                enhanced_features.append(enhanced)
            else:
                enhanced_features.append(feat)

        return enhanced_features


class AFAI(nn.Module):
    """平衡的自适应频域交互桥接"""

    def __init__(self, c_list: List[int]):
        super().__init__()

        # 跨尺度注意力
        self.cross_scale_att = CrossScaleAttention(c_list)

        # 频率交互模块
        self.freq_modules = nn.ModuleList([
            FrequencyInteractionModule(dim) for dim in c_list
        ])

        # 轻量级归一化层 - 使用组归一化，平衡效率和性能
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(16, dim), dim) for dim in c_list
        ])

    def forward(self, *features) -> Tuple[torch.Tensor, ...]:
        features = list(features)

        # 特征归一化 - 帮助稳定不同数据集上的训练
        normalized_features = []
        for i, feat in enumerate(features):
            normalized = self.norms[i](feat)
            normalized_features.append(normalized)

        # 跨尺度注意力
        enhanced_features = self.cross_scale_att(normalized_features)

        # 频率交互
        final_features = []
        for i, feat in enumerate(enhanced_features):
            freq_enhanced = self.freq_modules[i](feat)
            # 添加残差连接到原始输入，保留重要信息
            final_features.append(freq_enhanced + features[i])

        return tuple(final_features)