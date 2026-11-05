# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

"""Data augmentation adapted from https://github.com/facebookresearch/denoiser

Adapted by Wiliam Fernando López Gavilánez to be applied to single audio inputs.
"""

import math
import random

import numpy as np
from scipy.signal import convolve


def _hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_frequencies(n_mels, fmin, fmax):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels))


class LowPassFilters:
    """Bank of sinc low pass filters, with cutoffs expressed as f / sample_rate."""

    def __init__(self, cutoffs, width=None):
        if width is None:
            width = int(2 / min(cutoffs))
        window = np.hamming(2 * width + 1)
        t = np.arange(-width, width + 1, dtype=np.float32)
        self.filters = [2 * cutoff * np.sinc(2 * cutoff * t) * window for cutoff in cutoffs]

    def __call__(self, signal):
        return [convolve(signal, f, mode="same") for f in self.filters]


class RevEcho:
    """Reverb made of a succession of attenuated echoes of the input.

    The delay of the first echo is roughly twice the radius of the room and is controlled by
    ``first_delay``. Echoes keep being added with further attenuation until the amplitude ratio
    between the last and the first echo is 1e-3, which is reached after RT60 seconds.

    Args:
        proba: probability of applying the reverb.
        initial: the amplitude of the first echo is sampled from [0, initial].
        rt60: range to sample RT60 from, in seconds.
        first_delay: range to sample the first echo delay from, in seconds.
        repeat: number of echo trains with different jitter to add.
        jitter: each echo delay is resampled within first_delay +- jitter.
        keep_clean: fraction of the reverb of the speech added back to the speech. The rest goes
            to the noise.
        sample_rate: sample rate of the input signals.
    """

    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, sample_rate=16000):
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def _reverb(self, source, initial, first_delay, rt60):
        length = source.shape[-1]
        reverb = np.zeros_like(source).astype(np.float64)
        for _ in range(self.repeat):
            frac = 1  # fraction of the first echo amplitude that is still present
            echo = initial * source
            while frac > 1e-3:
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(1 + int(jitter * first_delay * self.sample_rate), length)
                echo = np.pad(echo[:-delay], (delay, 0), "constant")
                reverb += echo
                # With d the attenuation, d ** (rt60 / first_delay) = 1e-3.
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                attenuation = 10 ** (-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def __call__(self, clean, noise):
        if random.random() >= self.proba:
            return clean, noise

        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)

        reverb_noise = self._reverb(noise, initial, first_delay, rt60)
        reverb_clean = self._reverb(clean, initial, first_delay, rt60)

        noise += reverb_noise
        clean = clean.astype(np.float64)
        clean += self.keep_clean * reverb_clean
        noise += (1 - self.keep_clean) * reverb_clean
        return clean, noise


class BandMask:
    """Mask a band of frequencies of the waveform, similar to SpecAugment."""

    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16000):
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate

    def __call__(self, signal):
        bandwidth = int(abs(self.maxwidth) * self.bands)
        mels = mel_frequencies(self.bands, 40, self.sample_rate / 2) / self.sample_rate
        low = random.randrange(self.bands)
        high = random.randrange(low, min(self.bands, low + bandwidth))
        low_passed, mid_low_passed = LowPassFilters([mels[low], mels[high]])(signal)
        return signal - mid_low_passed + low_passed


class WhiteNoise:
    """Additive white Gaussian noise."""

    def __init__(self, min_snr_db=0, max_snr_db=20, p=0.5):
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def __call__(self, x):
        if self.p < random.random():
            return x
        snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)
        awgn = np.random.randn(*x.shape)
        desired_rms = np.sqrt(np.mean(x ** 2) / math.pow(10, snr_db / 10))
        x = x + desired_rms / np.std(awgn) * awgn
        return x / np.max(np.abs(x))
