"""Generator and discriminator networks of the MiniGAN paper.

The building blocks follow MP-SENet (https://github.com/yxlu-0102/MP-SENet, MIT
license). See THIRD_PARTY_NOTICES.md.

Attribute names and ``nn.Sequential`` layouts are part of the checkpoint format: changing
them makes the weights in ``checkpoints/`` impossible to load.
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

N_FREQ_BINS = 512 // 2 + 1  # n_fft // 2 + 1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionModule(nn.Module):
    """Multi-head self-attention.

    As in MP-SENet, inputs are [B, N, C] while nn.MultiheadAttention defaults to sequence-first,
    so attention runs along the first dimension. The published weights were trained this way.
    """

    def __init__(self, dim, n_head=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(
                inner_dim,
                inner_dim,
                kernel_size=kernel_size,
                padding=get_padding(kernel_size),
                groups=inner_dim,
            ),
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ccm(x)


class Conformer(nn.Module):
    def __init__(self, dim, n_head=4, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0.2, attn_dropout=0.2, ccm_dropout=0.0):
        super().__init__()
        self.ffm1 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConvolutionModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForward(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        return self.post_norm(x)


class TimeFrequencyConformerBlock(nn.Module):
    """One time conformer and one frequency conformer, in either order."""

    def __init__(self, dense_channel=64, frequency_first=False):
        super().__init__()
        self.frequency_first = frequency_first
        self.time_conformer = Conformer(dense_channel)
        self.freq_conformer = Conformer(dense_channel)

    def _time(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_conformer(x) + x
        return x.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()

    def _frequency(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        x = self.freq_conformer(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        if self.frequency_first:
            return self._time(self._frequency(x))
        return self._frequency(self._time(x))


class DenseBlock(nn.Module):
    """Dilated dense net with residual connections (dilations 1, 2, 4, 8)."""

    def __init__(self, kernel_size=(3, 3), depth=4, dense_channel=64):
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            self.dense_block.append(nn.Sequential(
                nn.Conv2d(dense_channel * (i + 1), dense_channel, kernel_size, dilation=(dilation, 1),
                          padding=get_padding_2d(kernel_size, (dilation, 1))),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel),
            ))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, in_channel=2, dense_channel=32):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.dense_block = DenseBlock(depth=4, dense_channel=dense_channel)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        return self.dense_conv_2(x)


class DenseMaskDecoder(nn.Module):
    def __init__(self, out_channel=1, dense_channel=32):
        super().__init__()
        self.dense_block = DenseBlock(depth=4, dense_channel=dense_channel)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(N_FREQ_BINS)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)


class DensePhaseDecoder(nn.Module):
    def __init__(self, out_channel=1, dense_channel=32):
        super().__init__()
        self.dense_block = DenseBlock(depth=4, dense_channel=dense_channel)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


def _conv_block(in_channels, out_channels, kernel_size, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, dilation=(dilation, 1),
                  padding=get_padding_2d(kernel_size, (dilation, 1))),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.PReLU(out_channels),
    )


class PlainEncoder(nn.Module):
    """Five convolution blocks that double the channels, with no frequency downsampling."""

    def __init__(self):
        super().__init__()
        self.dense_conv_1 = _conv_block(2, 4, (1, 1), 1)
        self.dense_conv_2 = _conv_block(4, 8, (3, 3), 1)
        self.dense_conv_3 = _conv_block(8, 16, (3, 3), 2)
        self.dense_conv_4 = _conv_block(16, 32, (3, 3), 4)
        self.dense_conv_5 = _conv_block(32, 64, (3, 3), 8)

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        return self.dense_conv_5(x)


class _PlainDecoderBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_conv_2 = _conv_block(64, 32, (3, 3), 8)
        self.dense_conv_3 = _conv_block(32, 16, (3, 3), 4)
        self.dense_conv_4 = _conv_block(16, 8, (3, 3), 2)
        self.dense_conv_5 = _conv_block(8, 4, (3, 3), 1)

    def body(self, x):
        x = self.dense_conv_2(x)
        x = self.dense_conv_3(x)
        x = self.dense_conv_4(x)
        return self.dense_conv_5(x)


class PlainMaskDecoder(_PlainDecoderBody):
    def __init__(self):
        super().__init__()
        self.mask_conv = nn.Sequential(nn.Conv2d(4, 1, (1, 1)))
        self.lsigmoid = LearnableSigmoid2d(N_FREQ_BINS)

    def forward(self, x):
        x = self.mask_conv(self.body(x))
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)


class PlainPhaseDecoder(_PlainDecoderBody):
    def __init__(self):
        super().__init__()
        self.phase_conv_r = nn.Conv2d(4, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(4, 1, (1, 1))

    def forward(self, x):
        x = self.body(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


class _MagnitudePhaseGenerator(nn.Module):
    def latent(self, x):
        raise NotImplementedError

    def forward(self, noisy_mag, noisy_pha):
        """Map a compressed magnitude [B, 1, F, T] and a phase [B, F, T] to their enhanced [B, F, T] versions."""
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_mag = noisy_mag.permute(0, 1, 3, 2)
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        x = self.latent(x)
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        return denoised_mag, denoised_pha


class MiniGANGenerator(_MagnitudePhaseGenerator):
    """MiniGAN, and through its arguments MiniGAN-48 and MiniGAN-FT."""

    def __init__(self, dense_channel=32, frequency_first=False):
        super().__init__()
        self.dense_encoder = DenseEncoder(in_channel=2, dense_channel=dense_channel)
        self.conformer = TimeFrequencyConformerBlock(dense_channel, frequency_first=frequency_first)
        self.mask_decoder = DenseMaskDecoder(out_channel=1, dense_channel=dense_channel)
        self.phase_decoder = DensePhaseDecoder(out_channel=1, dense_channel=dense_channel)

    def latent(self, x):
        return self.conformer(x)


class MiniGANEDGenerator(_MagnitudePhaseGenerator):
    """MiniGAN-ED (called LitGAN in the thesis): plain encoder-decoder, 64 latent channels."""

    def __init__(self):
        super().__init__()
        self.dense_encoder = PlainEncoder()
        self.conformer = TimeFrequencyConformerBlock(64)
        self.mask_decoder = PlainMaskDecoder()
        self.phase_decoder = PlainPhaseDecoder()

    def latent(self, x):
        return self.conformer(x)


class MidGANGenerator(_MagnitudePhaseGenerator):
    """MidGAN (thesis only): MiniGAN-ED with two conformer blocks."""

    def __init__(self):
        super().__init__()
        self.dense_encoder = PlainEncoder()
        self.conformer1 = TimeFrequencyConformerBlock(64)
        self.conformer2 = TimeFrequencyConformerBlock(64)
        self.mask_decoder = PlainMaskDecoder()
        self.phase_decoder = PlainPhaseDecoder()

    def latent(self, x):
        return self.conformer2(self.conformer1(x))


class Discriminator(nn.Module):
    """Metric discriminator shared by all variants; it sees magnitudes only."""

    def __init__(self, dim=16, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (3, 3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim * 2, (3, 3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            nn.utils.spectral_norm(nn.Conv2d(dim * 2, dim * 4, (3, 3), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim * 4, dim * 2)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 2),
            nn.utils.spectral_norm(nn.Linear(dim * 2, 1)),
            LearnableSigmoid1d(1),
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
