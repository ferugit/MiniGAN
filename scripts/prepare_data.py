"""Prepare VoiceBank+DEMAND for training and evaluation.

Expects the unzipped database (https://doi.org/10.7488/ds/2117) in ``--valentini_dir``:

    clean_trainset_56spk_wav/  noisy_trainset_56spk_wav/  log_trainset_56spk.txt
    clean_testset_wav/         noisy_testset_wav/

and writes to ``--output_dir``:

    <the same folders of wavs, at 16 kHz>
    noises_trainset_56spk_wav/  noises extracted as noisy - clean
    partition/train.tsv         partition/dev.tsv
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

WAV_FOLDERS = ["clean_trainset_56spk_wav", "noisy_trainset_56spk_wav", "clean_testset_wav", "noisy_testset_wav"]


def downsample(valentini_dir, output_dir, jobs):
    def convert(source, target):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
                        "-ar", "16000", "-ac", "1", "-bitexact", str(target)], check=True)

    with ThreadPoolExecutor(jobs) as pool:
        futures = []
        for folder in WAV_FOLDERS:
            (output_dir / folder).mkdir(parents=True, exist_ok=True)
            for source in sorted((valentini_dir / folder).glob("*.wav")):
                target = output_dir / folder / source.name
                if not target.exists():
                    futures.append(pool.submit(convert, source, target))
        for future in futures:
            future.result()


def extract_noises(output_dir, log_file):
    """Write noisy - clean for the training set and return their (path, noise type) rows."""
    noise_types = {}
    for line in log_file.read_text().splitlines():
        sample_id, noise_type, _ = line.split(" ")
        noise_types[sample_id] = noise_type

    noises_dir = output_dir / "noises_trainset_56spk_wav"
    noises_dir.mkdir(exist_ok=True)
    rows = []
    for clean_path in sorted((output_dir / "clean_trainset_56spk_wav").glob("*.wav")):
        clean, sample_rate = sf.read(clean_path, dtype="float32")
        noisy, _ = sf.read(output_dir / "noisy_trainset_56spk_wav" / clean_path.name, dtype="float32")
        noise = noisy - clean
        if np.mean(noise) == 0.0:
            continue
        noise_path = noises_dir / clean_path.name
        sf.write(noise_path, noise, sample_rate, subtype="FLOAT")
        rows.append({"Sample_Path": str(noise_path.resolve()), "Speaker_ID": "Unknown",
                     "Sound_Type": noise_types[clean_path.stem]})
    return pd.DataFrame(rows)


def make_partition(output_dir, noise_df, seed):
    """Hold out 10% of the speakers and 10% of each noise type for validation."""
    paths = sorted((output_dir / "clean_trainset_56spk_wav").glob("*.wav"))
    speech_df = pd.DataFrame({
        "Sample_Path": [str(p.resolve()) for p in paths],
        "Speaker_ID": [p.stem.split("_")[0] for p in paths],
        "Sound_Type": "speech",
    })
    train, dev = next(GroupShuffleSplit(test_size=0.1, n_splits=2, random_state=seed)
                      .split(speech_df, groups=speech_df["Speaker_ID"]))

    counts = noise_df["Sound_Type"].value_counts()
    rare = noise_df["Sound_Type"].isin(counts[counts == 1].index)
    splittable = noise_df[~rare]
    noise_train, noise_dev = next(StratifiedShuffleSplit(test_size=0.1, n_splits=2, random_state=seed)
                                  .split(splittable, splittable["Sound_Type"]))

    partition_dir = output_dir / "partition"
    partition_dir.mkdir(exist_ok=True)
    splits = {
        "train": [speech_df.iloc[train], splittable.iloc[noise_train], noise_df[rare]],
        "dev": [speech_df.iloc[dev], splittable.iloc[noise_dev]],
    }
    for name, frames in splits.items():
        pd.concat(frames, ignore_index=True).to_csv(partition_dir / f"{name}.tsv", sep="\t", index=False)
        print(f"{name}: {len(frames[0])} utterances, {sum(len(f) for f in frames[1:])} noises")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--valentini_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to downsample the database to 16 kHz")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downsample(args.valentini_dir, args.output_dir, args.jobs)
    noise_df = extract_noises(args.output_dir, args.valentini_dir / "log_trainset_56spk.txt")
    make_partition(args.output_dir, noise_df, args.seed)


if __name__ == "__main__":
    main()
