"""The model variants of the paper and how each one is built and trained."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .models import Discriminator, MidGANGenerator, MiniGANEDGenerator, MiniGANGenerator

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass(frozen=True)
class Variant:
    build_generator: Callable[[], torch.nn.Module]
    white_noise: bool = False
    revecho: bool = True
    bandmask: bool = True
    # Stabilisers the paper only uses for "extremely unstable" experiments.
    generator_lr: float = 1e-3
    kaiming_init: bool = False
    grad_clip: float = 0.0


def _minigan():
    return MiniGANGenerator()


VARIANTS = {
    "MiniGAN": Variant(_minigan),
    "MiniGAN-FT": Variant(lambda: MiniGANGenerator(frequency_first=True)),
    "MiniGAN-48": Variant(lambda: MiniGANGenerator(dense_channel=48)),
    "MiniGAN-NDA": Variant(_minigan, revecho=False, bandmask=False),
    "MiniGAN-WN": Variant(_minigan, white_noise=True, generator_lr=5e-4, kaiming_init=True, grad_clip=0.5),
    "MiniGAN-ED": Variant(MiniGANEDGenerator),
    "LitGAN": Variant(MiniGANEDGenerator),
    "MidGAN": Variant(MidGANGenerator),
}


def load_generator(name, checkpoint=None, device="cpu"):
    """Build a variant's generator and load its weights, by default the ones in the repository."""
    generator = VARIANTS[name].build_generator()
    path = checkpoint or CHECKPOINT_DIR / name / "generator.pt"
    generator.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return generator.to(device).eval()


def load_discriminator(name, checkpoint=None, device="cpu"):
    discriminator = Discriminator()
    path = checkpoint or CHECKPOINT_DIR / name / "discriminator.pt"
    discriminator.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return discriminator.to(device).eval()
