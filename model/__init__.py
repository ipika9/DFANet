from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
from typing import Any
from model.encoder import Encoder
from model.decoder import Decoder
import torch.nn.functional as F
from afai import AFAI
class CFAINET(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"
    ) -> None:
        super().__init__()
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.bridge = AFAI(
            c_list=[self.dims[0], self.dims[1], self.dims[2]],
            # split_att='fc'
        )
        self.decoder = Decoder(
            dims=[self.dims[-1]] + self.dims[:3][::-1],
            num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder_outputs = self.encoder(x)
        project1, project2, project3, deep = encoder_outputs

        ep1, ep2, ep3 = self.bridge(project1, project2, project3)

        output = [
            deep,
            F.interpolate(ep3, project3.shape[2:], mode='bilinear'),
            F.interpolate(ep2, project2.shape[2:], mode='bilinear'),
            ep1
        ]

        return self.decoder(output)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()

def build_model(**kwargs: Any) -> DFANet:
    return DFANet(**kwargs)
