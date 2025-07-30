class FrequencyDecomposer(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.low_pass = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        self.laplacian_kernel = nn.Parameter(
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float().view(1, 1, 3, 3) / 8.0,
            requires_grad=True  
        )

        self.freq_scale = nn.Parameter(torch.tensor([1.0, 1.0])) 
        self.adaptive_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low_freq = self.low_pass(x)

        batch, channels = x.shape[:2]
        weight = self.laplacian_kernel.to(x.device).repeat(channels, 1, 1, 1)
        high_freq = F.conv2d(x, weight, padding=1, groups=channels)

        adaptive_weights = self.adaptive_scale(x).squeeze(-1).squeeze(-1).view(batch, 2)

        low_freq = low_freq * (self.freq_scale[0] + adaptive_weights[:, 0].view(batch, 1, 1, 1))
        high_freq = high_freq * (self.freq_scale[1] + adaptive_weights[:, 1].view(batch, 1, 1, 1))

        return low_freq, high_freq


class FrequencyInteractionModule(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.freq_decomposer = FrequencyDecomposer(dim)

        self.freq_interaction = nn.Parameter(torch.tensor([[0.7, 0.3], [0.3, 0.7]]))
        self.softmax = nn.Softmax(dim=1)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 4, 8), kernel_size=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // 4, 8), dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.LayerNorm([dim, 1, 1])  # 或 LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, high_freq = self.freq_decomposer(x)

        interaction_weights = self.softmax(self.freq_interaction)

        low_enhanced = low_freq + high_freq * interaction_weights[0, 1]
        high_enhanced = high_freq + low_freq * interaction_weights[1, 0]

        low_att = self.channel_att(low_enhanced)
        high_att = self.channel_att(high_enhanced)

        low_enhanced = low_enhanced * low_att
        high_enhanced = high_enhanced * high_att

        freq_fused = self.fusion(torch.cat([low_enhanced, high_enhanced], dim=1))

        spatial_weight = self.spatial_att(freq_fused)
        freq_fused = freq_fused * spatial_weight

        return freq_fused + x


class CrossScaleAttention(nn.Module):

    def __init__(self, c_list: List[int]):
        super().__init__()

        self.projections = nn.ModuleList()
        for i, c_target in enumerate(c_list):
            proj_layers = nn.ModuleList()
            for j, c_source in enumerate(c_list):
                if i != j:
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

        self.cross_channels = []
        for i, c_target in enumerate(c_list):
            other_channels = sum([c_target for j, c in enumerate(c_list) if i != j])
            self.cross_channels.append(other_channels)

        self.gates = nn.ModuleList()
        for i, c in enumerate(c_list):
            bottleneck = max(c // 4, 8) 
            self.gates.append(nn.Sequential(
                nn.Conv2d(self.cross_channels[i], bottleneck, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck, c, kernel_size=1),
                nn.Sigmoid()
            ))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        enhanced_features = []

        for i, feat in enumerate(features):
            other_feats = []
            for j, other_feat in enumerate(features):
                if i != j:
                    h, w = feat.shape[2:]
                    if h <= 16 or w <= 16:  
                        aligned = F.interpolate(other_feat, size=(h, w), mode='nearest')
                    else: 
                        aligned = F.interpolate(other_feat, size=(h, w), mode='bilinear', align_corners=False)

                    projected = self.projections[i][j](aligned)
                    other_feats.append(projected)

            if other_feats:
                cross_info = torch.cat(other_feats, dim=1) if len(other_feats) > 1 else other_feats[0]
                gate = self.gates[i](cross_info)
                enhanced = feat + feat * gate
                enhanced_features.append(enhanced)
            else:
                enhanced_features.append(feat)

        return enhanced_features


class AFIB(nn.Module):

    def __init__(self, c_list: List[int]):
        super().__init__()

        self.cross_scale_att = CrossScaleAttention(c_list)

        self.freq_modules = nn.ModuleList([
            FrequencyInteractionModule(dim) for dim in c_list
        ])

        self.norms = nn.ModuleList([
            nn.GroupNorm(min(16, dim), dim) for dim in c_list
        ])

    def forward(self, *features) -> Tuple[torch.Tensor, ...]:
        features = list(features)

        normalized_features = []
        for i, feat in enumerate(features):
            normalized = self.norms[i](feat)
            normalized_features.append(normalized)

        enhanced_features = self.cross_scale_att(normalized_features)

        final_features = []
        for i, feat in enumerate(enhanced_features):
            freq_enhanced = self.freq_modules[i](feat)
            final_features.append(freq_enhanced + features[i])

        return tuple(final_features)
