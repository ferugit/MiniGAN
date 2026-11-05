"""Training data: clean VoiceBank speech mixed on the fly with noises from the same database."""

import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from .augment import BandMask, RevEcho, WhiteNoise


def load_partition(partition_dir, split, **kwargs):
    """Load the ``train`` or ``dev`` split written by ``scripts/prepare_data.py``."""
    df = pd.read_csv(os.path.join(partition_dir, split + ".tsv"), sep="\t")
    speech = df[df["Sound_Type"] == "speech"]["Sample_Path"].tolist()
    noises = df[df["Sound_Type"] != "speech"]["Sample_Path"].tolist()
    return AudioDataset(speech, noises, **kwargs)


def _normalize(audio, eps=0.001):
    """Peak normalization."""
    return (audio.astype(np.float32) / float(np.amax(np.abs(audio)))) + eps


def _read_wav(path):
    _, audio = wavfile.read(path)
    return _normalize(audio)


def _pad_randomly(audio, length):
    padding = length - len(audio)
    left, right = np.random.multinomial(padding, np.ones(2) / 2, size=1)[0]
    return np.pad(audio, (left, right), "constant", constant_values=(0, 0))


class AudioDataset(torch.utils.data.Dataset):
    """Yield (noisy, clean) windows, in the order of section 3.2 of the paper."""

    def __init__(self, speech_paths, noise_paths, window_size=32000, snr_range=(5, 30),
                 white_noise=False, revecho=False, bandmask=False):
        self.speech_paths = speech_paths
        self.noise_paths = noise_paths
        self.window_size = window_size
        self.snr_range = snr_range
        self.white_noise = WhiteNoise() if white_noise else None
        self.revecho = RevEcho() if revecho else None
        self.bandmask = BandMask() if bandmask else None

    def __len__(self):
        return len(self.speech_paths)

    def __getitem__(self, index):
        clean, noisy = self._prepare(_read_wav(self.speech_paths[index]))
        return _normalize(noisy, eps=0), _normalize(clean, eps=0)

    def _prepare(self, audio):
        if len(audio) >= self.window_size:
            start = random.randint(0, len(audio) - self.window_size)
            audio = audio[start:start + self.window_size]
        else:
            audio = _pad_randomly(audio, self.window_size)

        audio = _normalize(audio) if self.white_noise else audio
        noisy = self.white_noise(audio) if self.white_noise else _normalize(audio)
        noisy = self._add_background_noise(noisy)
        if self.bandmask:
            noisy = self.bandmask(noisy)
        return audio, noisy

    def _add_background_noise(self, speech):
        noise = _read_wav(random.choice(self.noise_paths))
        if len(noise) < len(speech):
            noise = _pad_randomly(noise, len(speech))
        start = random.randint(0, len(noise) - len(speech))
        noise = noise[start:start + len(speech)].astype(np.float64)

        snr_db = np.random.uniform(*self.snr_range)
        speech_rms = np.sqrt(np.mean(speech ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        noise *= np.sqrt(speech_rms ** 2 / math.pow(10, snr_db / 10)) / noise_rms

        if self.revecho:
            speech, noise = self.revecho(speech, noise)
        return speech + noise
