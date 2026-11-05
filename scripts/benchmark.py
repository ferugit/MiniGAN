"""Measure parameters, model size, FLOPS and real-time factor (RTF) of a variant on CPU.

    python -m scripts.benchmark --variant MiniGAN --noisy_dir data/noisy_testset_wav

The RTF depends on the machine, so expect it to differ from the paper. FLOPS are what thop
reports per second of audio, which is how the paper's table was produced.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from thop import profile

from minigan import dsp
from minigan.variants import VARIANTS, load_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="MiniGAN")
    parser.add_argument("--noisy_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, help="use only the first files, for a quick estimate")
    parser.add_argument("--threads", type=int, help="CPU threads; defaults to the torch default")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    generator = load_generator(args.variant)
    profiled = copy.deepcopy(generator)  # thop leaves counters on the modules it measures
    rtf, flops = [], []
    for path in sorted(args.noisy_dir.glob("*.wav"))[:args.max_files]:
        noisy, sample_rate = sf.read(path, dtype="float32")
        noisy = torch.from_numpy(noisy)
        seconds = len(noisy) / sample_rate

        start = time.perf_counter()
        dsp.enhance(generator, noisy)
        rtf.append((time.perf_counter() - start) / seconds)

        mag, phase = dsp.magnitude_and_phase(*dsp.compressed_stft(noisy.unsqueeze(0) * dsp.rms_gain(noisy.unsqueeze(0))))
        operations, _ = profile(profiled, (mag, phase), verbose=False)
        flops.append(operations / seconds)

    parameters = sum(p.numel() for p in generator.parameters())
    print(f"{args.variant}: generator parameters {parameters:,}  size {parameters * 4 / 1e6:.2f} MB  "
          f"FLOPS {np.mean(flops) / 1e6:.0f} M  RTF {np.mean(rtf):.3f} over {len(rtf)} files")


if __name__ == "__main__":
    main()
