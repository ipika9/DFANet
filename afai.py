class FrequencyDecomposer(nn.Module):

    def __init__(self, dim: int, groups: int = 8):
        super().__init__()

        self.low_pass = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=groups, bias=False)
        self.high_pass = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=groups, bias=False)

        with torch.no_grad():

            nn.init.normal_(self.low_pass.weight, mean=0.1, std=0.01)

            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float() / 4.0
            self.high_pass.weight.data.copy_(
                laplacian.view(1, 1, 3, 3).repeat(dim, dim // groups, 1, 1)
            )

        self.freq_scale = nn.Parameter(torch.tensor([1.0, 0.5]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        low_freq = self.low_pass(x) * self.freq_scale[0]
        high_freq = self.high_pass(x) * self.freq_scale[1]

        return low_freq, high_freq


class FrequencyAttention(nn.Module):

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim * 2 // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2 // reduction, dim * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.freq_interaction = nn.Parameter(torch.ones(2, 2) / 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, freqs: List[torch.Tensor]) -> List[torch.Tensor]:
        concat_freqs = torch.cat(freqs, dim=1)

        channel_weights = self.channel_gate(concat_freqs)
        channel_chunks = torch.chunk(channel_weights, 2, dim=1)

        interaction_weights = self.softmax(self.freq_interaction)

        enhanced_freqs = []
        for i, freq in enumerate(freqs):

            freq = freq * channel_chunks[i]

            for j, other_freq in enumerate(freqs):
                if i != j:
                    freq = freq + other_freq * interaction_weights[i, j]

            enhanced_freqs.append(freq)

        return enhanced_freqs


class FrequencyFusion(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=4),
            nn.Sigmoid()
        )

    def forward(self, freqs: List[torch.Tensor], identity: torch.Tensor) -> torch.Tensor:
        concat_freqs = torch.cat(freqs, dim=1)

        fused = self.fusion(concat_freqs)

        gate_value = self.gate(fused)
        output = identity * (1 - gate_value) + fused * gate_value

        return output


class AFIB(nn.Module):

    def __init__(self, c_list: List[int], reduction: int = 16):
        super().__init__()

        self.freq_decomposers = nn.ModuleList([
            FrequencyDecomposer(dim) for dim in c_list
        ])

        self.freq_attentions = nn.ModuleList([
            FrequencyAttention(dim, reduction) for dim in c_list
        ])

        self.freq_fusions = nn.ModuleList([
            FrequencyFusion(dim) for dim in c_list
        ])

    def forward(self, *features) -> Tuple[torch.Tensor, ...]:
        features = list(features)

        freq_components = []
        for i, feature in enumerate(features):
            freq_components.append(
                self.freq_decomposers[i](feature)
            )

        cross_scale_features = []
        for i in range(len(features)):
            aligned_features = []
            for j, feat in enumerate(features):
                if i != j:
                    aligned = F.interpolate(feat, size=features[i].shape[2:], mode='bilinear')
                    aligned_features.append(aligned)

            cross_info = torch.cat(aligned_features, dim=1)
            cross_conv = nn.Conv2d(cross_info.shape[1], features[i].shape[1],
                                   kernel_size=1).to(features[i].device)
            cross_scale_features.append(features[i] + cross_conv(cross_info))

        enhanced_components = []
        for i, comps in enumerate(freq_components):
            enhanced_components.append(
                self.freq_attentions[i](list(comps))
            )

        final_features = []
        for i, (comps, feature) in enumerate(zip(enhanced_components, features)):
            final_features.append(
                self.freq_fusions[i](comps, feature)
            )

        return tuple(final_features)
