"""Minimal BIT-LEVIR model compatible with the authors' public checkpoint."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models import resnet18  # type: ignore[import-untyped]


class _Residual(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.fn = layer

    def forward(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.fn(inputs, **kwargs) + inputs


class _PreNorm(nn.Module):
    def __init__(self, dimension: int, layer: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.fn = layer

    def forward(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.fn(self.norm(inputs), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(0),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _Attention(nn.Module):
    def __init__(self, dimension: int, heads: int, head_dimension: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dimension ** -0.5
        inner_dimension = heads * head_dimension
        self.to_qkv = nn.Linear(dimension, inner_dimension * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dimension, dimension), nn.Dropout(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = inputs.shape
        query, key, value = self.to_qkv(inputs).chunk(3, dim=-1)
        query = query.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        key = key.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        value = value.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(output)


class _Transformer(nn.Module):
    def __init__(self, dimension: int, depth: int, head_dimension: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _Residual(_PreNorm(dimension, _Attention(dimension, 8, head_dimension))),
                _Residual(_PreNorm(dimension, _FeedForward(dimension, dimension * 2))),
            ])
            for _ in range(depth)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer_pair in self.layers:
            pair = cast(Any, layer_pair)
            attention = pair[0]
            feed_forward = pair[1]
            inputs = attention(inputs)
            inputs = feed_forward(inputs)
        return inputs


class _CrossAttention(nn.Module):
    def __init__(self, dimension: int, head_dimension: int) -> None:
        super().__init__()
        self.heads = 8
        self.scale = dimension ** -0.5
        inner_dimension = self.heads * head_dimension
        self.to_q = nn.Linear(dimension, inner_dimension, bias=False)
        self.to_k = nn.Linear(dimension, inner_dimension, bias=False)
        self.to_v = nn.Linear(dimension, inner_dimension, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dimension, dimension), nn.Dropout(0))

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = inputs.shape
        memory_tokens = memory.shape[1]
        query = self.to_q(inputs).reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        key = self.to_k(memory).reshape(batch, memory_tokens, self.heads, -1).transpose(1, 2)
        value = self.to_v(memory).reshape(batch, memory_tokens, self.heads, -1).transpose(1, 2)
        weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(output)


class _CrossPreNorm(nn.Module):
    def __init__(self, dimension: int, layer: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.fn = layer

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(inputs), self.norm(memory))


class _ResidualCross(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.fn = layer

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, memory) + inputs


class _TransformerDecoder(nn.Module):
    def __init__(self, dimension: int, depth: int, head_dimension: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _ResidualCross(
                    _CrossPreNorm(dimension, _CrossAttention(dimension, head_dimension))
                ),
                _Residual(_PreNorm(dimension, _FeedForward(dimension, dimension * 2))),
            ])
            for _ in range(depth)
        ])

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer_pair in self.layers:
            pair = cast(Any, layer_pair)
            attention = pair[0]
            feed_forward = pair[1]
            inputs = attention(inputs, memory)
            inputs = feed_forward(inputs)
        return inputs


class BITLevirModel(nn.Module):
    """BIT base transformer used by the public LEVIR-CD checkpoint."""

    input_size = 256
    input_channels = 3
    output_channels = 2

    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet18(weights=None)
        # The original repository's BasicBlock resets unsupported dilation to
        # one while retaining the dilated stage's stride of one.
        self.resnet.layer3[0].conv1.stride = (1, 1)
        self.resnet.layer3[0].downsample[0].stride = (1, 1)
        self.conv_pred = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.conv_a = nn.Conv2d(32, 4, kernel_size=1, padding=0, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, 8, 32))
        self.transformer = _Transformer(32, depth=1, head_dimension=64)
        self.transformer_decoder = _TransformerDecoder(32, depth=8, head_dimension=8)
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, stride=1),
        )

    def _features(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(inputs)))
        inputs = self.resnet.maxpool(inputs)
        inputs = self.resnet.layer1(inputs)
        inputs = self.resnet.layer2(inputs)
        inputs = self.resnet.layer3(inputs)
        return self.conv_pred(self.upsamplex2(inputs))

    def _tokens(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        attention = torch.softmax(self.conv_a(features).reshape(batch, 4, -1), dim=-1)
        flattened = features.reshape(batch, channels, -1)
        return torch.einsum("bln,bcn->blc", attention, flattened)

    def _decode(self, features: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.transformer_decoder(tokens, memory)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        before_features = self._features(before)
        after_features = self._features(after)
        tokens = self.transformer(
            torch.cat((self._tokens(before_features), self._tokens(after_features)), dim=1)
        )
        before_tokens, after_tokens = tokens.chunk(2, dim=1)
        before_features = self._decode(before_features, before_tokens)
        after_features = self._decode(after_features, after_tokens)
        difference = torch.abs(before_features - after_features)
        return self.classifier(self.upsamplex4(difference))