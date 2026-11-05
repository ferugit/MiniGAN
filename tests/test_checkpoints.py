"""Every published checkpoint must load strictly and match the sizes in the paper and thesis."""

import pytest
import torch

from minigan import dsp
from minigan.variants import CHECKPOINT_DIR, VARIANTS, load_discriminator, load_generator

# Generator parameters: thesis table 4.3, and 0.801 M total for MiniGAN-48 in the paper's table 1.
GENERATOR_PARAMETERS = {
    "MiniGAN": 347_273,
    "MiniGAN-FT": 347_273,
    "MiniGAN-NDA": 347_273,
    "MiniGAN-WN": 347_273,
    "MiniGAN-48": pytest.approx(801_000 - 25_810, abs=1_000),
    "MiniGAN-ED": 299_592,
    "LitGAN": 299_592,
    "MidGAN": 524_488,
}
DISCRIMINATOR_PARAMETERS = 25_810


def count(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", VARIANTS)
def test_generator_size(name):
    assert count(load_generator(name)) == GENERATOR_PARAMETERS[name]


@pytest.mark.parametrize("name", VARIANTS)
def test_discriminator_size(name):
    assert count(load_discriminator(name)) == DISCRIMINATOR_PARAMETERS


@pytest.mark.parametrize("name", VARIANTS)
def test_enhance_keeps_length(name):
    torch.manual_seed(0)
    noisy = torch.randn(dsp.SAMPLE_RATE // 2 + 17) * 0.1
    estimate = dsp.enhance(load_generator(name), noisy)
    assert estimate.shape == noisy.shape
    assert torch.isfinite(estimate).all()


def test_ed_and_litgan_are_the_same_model():
    for file in ("generator.pt", "discriminator.pt"):
        assert (CHECKPOINT_DIR / "MiniGAN-ED" / file).read_bytes() == (CHECKPOINT_DIR / "LitGAN" / file).read_bytes()
