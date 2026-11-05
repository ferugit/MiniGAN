"""Compute the objective metrics of a variant on the VoiceBank+DEMAND test set.

    python -m scripts.evaluate --variant MiniGAN \
        --noisy_dir data/noisy_testset_wav --clean_dir data/clean_testset_wav
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from joblib import Parallel, delayed

from minigan.dsp import SAMPLE_RATE, enhance_with_fallback
from minigan.metrics import compute_metrics
from minigan.variants import VARIANTS, load_generator

METRICS = ["PESQ", "CSIG", "CBAK", "COVL", "SSNR", "STOI"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="MiniGAN")
    parser.add_argument("--checkpoint", type=Path, help="generator weights; defaults to the ones in checkpoints/")
    parser.add_argument("--noisy_dir", type=Path, required=True)
    parser.add_argument("--clean_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, help="also write the enhanced audio to this folder")
    parser.add_argument("--output_json", type=Path, help="also write the mean metrics to this file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel jobs for the metrics")
    args = parser.parse_args()

    generator = load_generator(args.variant, args.checkpoint, args.device)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    paths = sorted(args.noisy_dir.glob("*.wav"))
    for index, path in enumerate(paths, start=1):
        noisy, sample_rate = sf.read(path, dtype="float32")
        clean, clean_rate = sf.read(args.clean_dir / path.name)
        if sample_rate != SAMPLE_RATE or clean_rate != SAMPLE_RATE:
            raise SystemExit(f"{path.name} is not at {SAMPLE_RATE} Hz; run scripts/prepare_data.py first")
        estimate = enhance_with_fallback(generator, torch.from_numpy(noisy).to(args.device)).cpu().numpy()
        if args.save_dir:
            sf.write(args.save_dir / path.name, estimate, SAMPLE_RATE)
        pairs.append((clean, estimate))
        if index % 100 == 0:
            print(f"enhanced {index}/{len(paths)}", flush=True)

    scores = Parallel(n_jobs=args.jobs)(delayed(compute_metrics)(c, e, SAMPLE_RATE, 0) for c, e in pairs)
    means = dict(zip(METRICS, map(float, np.mean(scores, axis=0))))
    print(args.variant, f"({len(paths)} files):", "  ".join(f"{k} {v:.2f}" for k, v in means.items()))
    if args.output_json:
        args.output_json.write_text(json.dumps({"variant": args.variant, "files": len(paths), **means}, indent=2))


if __name__ == "__main__":
    main()
