"""Enhance a 16 kHz mono wav file.

    python -m scripts.enhance --variant MiniGAN --input noisy.wav --output enhanced.wav
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from minigan.dsp import SAMPLE_RATE, enhance_with_fallback
from minigan.variants import VARIANTS, load_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="MiniGAN")
    parser.add_argument("--checkpoint", type=Path, help="generator weights; defaults to the ones in checkpoints/")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    noisy, sample_rate = sf.read(args.input, dtype="float32")
    if sample_rate != SAMPLE_RATE or noisy.ndim != 1:
        raise SystemExit(f"expected {SAMPLE_RATE} Hz mono audio, got {sample_rate} Hz with shape {noisy.shape}. "
                         f"Convert it first: ffmpeg -i in.wav -ar {SAMPLE_RATE} -ac 1 out.wav")
    generator = load_generator(args.variant, args.checkpoint, args.device)
    estimate = enhance_with_fallback(generator, torch.from_numpy(noisy).to(args.device))
    sf.write(args.output, estimate.cpu().numpy(), SAMPLE_RATE)


if __name__ == "__main__":
    main()
