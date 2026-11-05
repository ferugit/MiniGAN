"""Training losses, equations (1)-(11) of the paper."""

import numpy as np
import torch
from pesq import pesq

from .dsp import N_FFT, SAMPLE_RATE

# Order: GAN, time, magnitude, instantaneous phase, group delay, instantaneous frequency.
LOSS_WEIGHTS = (0.05, 0.1, 0.2, 0.2, 0.1, 0.1)


def _anti_wrapping(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def _difference_matrix(size, device):
    return (torch.triu(torch.ones(size, size), diagonal=1)
            - torch.triu(torch.ones(size, size), diagonal=2)
            - torch.eye(size)).to(device)


def phase_losses(clean_phase, est_phase):
    device = est_phase.device
    gd_matrix = _difference_matrix(N_FFT // 2 + 1, device)
    iaf_matrix = _difference_matrix(clean_phase.size(-1), device)
    instantaneous = torch.mean(_anti_wrapping(clean_phase - est_phase))
    group_delay = torch.mean(_anti_wrapping(
        torch.matmul(clean_phase.permute(0, 2, 1), gd_matrix) - torch.matmul(est_phase.permute(0, 2, 1), gd_matrix)
    ))
    angular_frequency = torch.mean(_anti_wrapping(
        torch.matmul(clean_phase, iaf_matrix) - torch.matmul(est_phase, iaf_matrix)
    ))
    return instantaneous, group_delay, angular_frequency


def generator_loss(out, discriminator, weights=LOSS_WEIGHTS):
    gan = torch.mean((discriminator(out["clean_mag"], out["est_mag"]).flatten() - 1.0) ** 2)
    time = torch.mean(torch.abs(out["est_audio"] - out["clean"]))
    magnitude = torch.mean((out["est_mag"] - out["clean_mag"]) ** 2)
    instantaneous, group_delay, angular_frequency = phase_losses(out["clean_phase"], out["est_phase"])
    return (weights[0] * gan + weights[1] * time + weights[2] * magnitude
            + weights[3] * instantaneous + weights[4] * group_delay + weights[5] * angular_frequency)


def discriminator_loss(out, discriminator):
    enhanced = discriminator(out["clean_mag"], out["est_mag"].detach()).flatten()
    clean = discriminator(out["clean_mag"], out["clean_mag"]).flatten()
    return torch.mean((clean - 1.0) ** 2) + torch.mean(enhanced ** 2)


def batch_pesq(out):
    """Mean wide-band PESQ of a batch, or None if PESQ fails, as it can on silent segments."""
    enhanced = out["est_audio"].detach().cpu().numpy()
    clean = out["clean"].cpu().numpy()
    try:
        return float(np.mean([pesq(SAMPLE_RATE, c, e, "wb") for c, e in zip(clean, enhanced)]))
    except Exception:
        return None
