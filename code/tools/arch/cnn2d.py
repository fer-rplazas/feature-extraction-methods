import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, ks: int = 3, stride: int = 1):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, ks, padding=(ks - 1) // 2, stride=stride),
            nn.SiLU(),
            nn.BatchNorm2d(n_out),
        )

    def forward(self, x):
        return self.layers(x)


def _conv_block(ni: int, nf: int, stride: int):
    """Bottleneck layer."""
    return nn.Sequential(
        ConvLayer(ni, nf // 4, 1),
        ConvLayer(nf // 4, nf // 4, stride=stride),
        ConvLayer(nf // 4, nf, 1),
    )


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else ConvLayer(ni, nf, ks=1)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return torch.relu(self.convs(x) + self.idconv(self.pool(x)))


def _resnet_stem(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i + 1], 3, stride=2 if i == 0 else 1)
        for i in range(len(sizes) - 1)
    ] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]


class ResNet2d(nn.Module):
    """ "ResNet2d implementation.

    Adapted from `fastai's implementation <https://github.com/fastai/fastbook/blob/master/14_resnet.ipynb)>`,
    which in turn implements `https://arxiv.org/abs/1812.01187`
    """

    def __init__(self, n_in: int, n_out: int, layers: list[int]):
        super().__init__()
        stem = _resnet_stem(n_in, 32, 32, 64)
        self.block_szs = [64, 64, 128, 256, 512]
        blocks = [
            self._make_layer(idx, n_layers) for idx, n_layers in enumerate(layers)
        ]
        self.layers = nn.Sequential(
            *stem,
            *blocks,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.block_szs[len(layers)], n_out),
            nn.Flatten(),
        )

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_szs[idx : idx + 2]
        return nn.Sequential(
            *[
                ResBlock(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        return self.layers(x)


def resnet(n_channels: int, n_out: int = 1):
    return ResNet2d(n_channels, n_out, [3, 3, 5, 5])
