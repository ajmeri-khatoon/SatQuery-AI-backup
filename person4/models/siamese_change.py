"""Small CPU-capable Siamese dense change detector.

The network follows a common bi-temporal segmentation pattern: a shared
encoder extracts features from both dates, latent differences are decoded to
the registered input grid, and a one-channel logit predicts change per pixel.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SiameseChangeModel(nn.Module):
    """Shared-weight encoder and difference decoder for registered image pairs."""

    def __init__(self, input_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        if input_channels < 1 or base_channels < 1:
            raise ValueError("input_channels and base_channels must be positive")
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.encoder1 = ConvBlock(input_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.decoder2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def _encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level1 = self.encoder1(inputs)
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        return level1, level2, level3

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.ndim != 4 or after.ndim != 4:
            raise ValueError("before and after inputs must be NCHW tensors")
        if before.shape != after.shape:
            raise ValueError("before and after tensors must have identical shapes")
        if before.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels")
        before1, before2, before3 = self._encode(before)
        after1, after2, after3 = self._encode(after)
        difference3 = torch.abs(after3 - before3)
        decoded2 = functional.interpolate(
            difference3, size=before2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded2 = self.decoder2(torch.cat((decoded2, torch.abs(after2 - before2)), dim=1))
        decoded1 = functional.interpolate(
            decoded2, size=before1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded1 = self.decoder1(torch.cat((decoded1, torch.abs(after1 - before1)), dim=1))
        return self.head(functional.interpolate(
            decoded1, size=before.shape[-2:], mode="bilinear", align_corners=False
        ))