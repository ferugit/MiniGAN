"""STFT analysis and synthesis around the generator."""

import math

import torch

N_FFT = 512
HOP = 100
COMPRESSION = 0.3
SAMPLE_RATE = 16000


def stft_window(device):
    return torch.hamming_window(N_FFT, device=device)


def compressed_stft(audio):
    """Return the power-compressed real and imaginary spectra of ``audio`` [B, T], each [B, 1, F, T]."""
    spec = torch.stft(audio, N_FFT, HOP, window=stft_window(audio.device), onesided=True, return_complex=True)
    mag = torch.abs(spec) ** COMPRESSION
    phase = torch.angle(spec)
    return (mag * torch.cos(phase)).unsqueeze(1), (mag * torch.sin(phase)).unsqueeze(1)


def magnitude_and_phase(real, imag):
    """Return magnitude [B, 1, F, T] and phase [B, F, T] of a compressed spectrum."""
    return torch.sqrt(real ** 2 + imag ** 2), torch.atan2(imag.squeeze(1), real.squeeze(1))


def synthesize(mag, phase, length=None):
    """Turn a compressed magnitude and a phase, both [B, F, T], back into audio [B, T]."""
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    spec = torch.complex(real, imag)
    uncompressed = torch.abs(spec) ** (1.0 / COMPRESSION)
    angle = torch.angle(spec)
    spec = torch.complex(uncompressed * torch.cos(angle), uncompressed * torch.sin(angle))
    return torch.istft(spec, N_FFT, HOP, window=stft_window(mag.device), onesided=True, length=length)


def rms_gain(audio):
    """Gain [B, 1] that normalises each signal in ``audio`` [B, T] to unit RMS."""
    return torch.sqrt(audio.size(-1) / torch.sum(audio ** 2.0, dim=-1, keepdim=True))


@torch.no_grad()
def enhance(generator, noisy):
    """Enhance a whole utterance, given as a 1-D tensor, without slicing it."""
    noisy = noisy.unsqueeze(0)
    length = noisy.size(-1)
    gain = rms_gain(noisy)
    normalized = noisy * gain
    # Round the length up to a whole number of hops by repeating the start of the signal.
    padding = math.ceil(length / HOP) * HOP - length
    normalized = torch.cat([normalized, normalized[:, :padding]], dim=-1)
    mag, phase = magnitude_and_phase(*compressed_stft(normalized))
    est_mag, est_phase = generator(mag, phase)
    return (synthesize(est_mag, est_phase) / gain).flatten()[:length]


def enhance_with_fallback(generator, noisy):
    """Enhance on the current device, moving to the CPU if the GPU runs out of memory.

    Memory grows quickly with the length of the utterance, and slicing it would change the result.
    """
    device = noisy.device
    try:
        return enhance(generator, noisy)
    except torch.cuda.OutOfMemoryError:
        pass
    torch.cuda.empty_cache()
    estimate = enhance(generator.cpu(), noisy.cpu())
    generator.to(device)
    return estimate.to(device)
