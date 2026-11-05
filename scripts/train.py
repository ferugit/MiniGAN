"""Train one variant with the settings of the paper.

    python -m scripts.train --variant MiniGAN --partition_dir data/partition
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from minigan import dsp
from minigan.data import load_partition
from minigan.losses import batch_pesq, discriminator_loss, generator_loss
from minigan.models import Discriminator
from minigan.variants import VARIANTS

log = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="MiniGAN")
    parser.add_argument("--partition_dir", type=Path, required=True, help="folder with train.tsv and dev.tsv")
    parser.add_argument("--output_dir", type=Path, help="defaults to runs/<variant>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=75, help="epochs without improvement before stopping")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--decay_epochs", type=int, default=30, help="halve the learning rates every this many epochs")
    parser.add_argument("--discriminator_lr", type=float, default=1e-3)
    parser.add_argument("--time_window", type=float, default=2.0, help="training window in seconds")
    parser.add_argument("--min_snr", type=int, default=5, help="minimum background noise SNR in dB")
    parser.add_argument("--max_snr", type=int, default=30, help="maximum background noise SNR in dB")
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=500, help="steps between training logs")
    parser.add_argument("--detect_anomaly", action="store_true", help="stop on NaN gradients; much slower")
    parser.add_argument("--max_batches", type=int, help="stop every epoch after this many batches, for smoke tests")
    return parser.parse_args()


def kaiming_init(model):
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.kaiming_uniform_(param.unsqueeze(0), a=0, mode="fan_in", nonlinearity="leaky_relu")


def forward(generator, noisy, clean):
    """Run the generator and return everything the losses need."""
    gain = dsp.rms_gain(noisy)
    noisy_mag, noisy_phase = dsp.magnitude_and_phase(*dsp.compressed_stft(noisy * gain))
    clean_mag, clean_phase = dsp.magnitude_and_phase(*dsp.compressed_stft(clean * gain))
    est_mag, est_phase = generator(noisy_mag, noisy_phase)
    est_audio = dsp.synthesize(est_mag, est_phase, length=clean.size(1)) / gain
    return {
        "clean": clean,
        "clean_mag": clean_mag.squeeze(1),
        "clean_phase": clean_phase,
        "est_audio": est_audio,
        "est_mag": est_mag,
        "est_phase": est_phase,
    }


class Trainer:
    def __init__(self, args, variant, train_loader, dev_loader):
        self.args = args
        self.device = args.device
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.grad_clip = variant.grad_clip
        self.generator = variant.build_generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        if variant.kaiming_init:
            kaiming_init(self.generator)
            kaiming_init(self.discriminator)
        self.optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=variant.generator_lr)
        self.optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=args.discriminator_lr)
        self.history = []

    def _clip(self, model):
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)

    def train_step(self, noisy, clean, step, last_step):
        out = forward(self.generator, noisy, clean)
        update = step % self.args.accumulation_steps == 0 or step == last_step

        # The adversarial term also backpropagates into the discriminator and those gradients
        # stay until optimizer_d.step(). This is how the published models were trained, so it is
        # kept.
        loss_g = generator_loss(out, self.discriminator)
        loss_g.backward()
        self._clip(self.generator)
        if update:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()

        loss_d = discriminator_loss(out, self.discriminator)
        loss_d.backward()
        self._clip(self.discriminator)
        if update:
            self.optimizer_d.step()
            self.optimizer_d.zero_grad()
        return loss_g.item(), loss_d.item()

    @torch.no_grad()
    def validate(self):
        self.generator.eval()
        self.discriminator.eval()
        loss_g, loss_d, pesq_total, steps = 0.0, 0.0, 0.0, 0
        for noisy, clean in self.dev_loader:
            out = forward(self.generator, noisy.to(self.device), clean.to(self.device))
            loss_g += generator_loss(out, self.discriminator).item()
            loss_d += discriminator_loss(out, self.discriminator).item()
            pesq_total += batch_pesq(out) or 0.0
            steps += 1
            if steps == self.args.max_batches:
                break
        return loss_g / steps, loss_d / steps, pesq_total / steps

    def save(self, suffix=""):
        output_dir = self.args.output_dir
        torch.save(self.generator.state_dict(), output_dir / f"generator{suffix}.pt")
        torch.save(self.discriminator.state_dict(), output_dir / f"discriminator{suffix}.pt")

    def train(self):
        args = self.args
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=args.decay_epochs, gamma=0.5)
        scheduler_d = torch.optim.lr_scheduler.StepLR(self.optimizer_d, step_size=args.decay_epochs, gamma=0.5)
        best_loss, epochs_without_improvement = np.inf, 0
        last_step = min(len(self.train_loader), args.max_batches or len(self.train_loader))

        for epoch in range(args.epochs):
            self.generator.train()
            self.discriminator.train()
            total_g, total_d = 0.0, 0.0
            for step, (noisy, clean) in enumerate(self.train_loader, start=1):
                loss_g, loss_d = self.train_step(noisy.to(self.device), clean.to(self.device), step, last_step)
                total_g += loss_g
                total_d += loss_d
                if step % args.log_interval == 0:
                    log.info("epoch %d step %d/%d: g=%.4f d=%.4f", epoch, step, last_step, loss_g, loss_d)
                if step == last_step:
                    break

            val_g, val_d, pesq = self.validate()
            log.info("epoch %d validation: g=%.4f d=%.4f pesq=%.3f", epoch, val_g, val_d, pesq)
            self.history.append({"epoch": epoch, "train_g": total_g / last_step, "train_d": total_d / last_step,
                                 "val_g": val_g, "val_d": val_d, "val_pesq": pesq})
            (args.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))

            if best_loss <= val_g:
                epochs_without_improvement += 1
                if epochs_without_improvement == args.patience:
                    log.info("out of patience")
                    break
            elif pesq != 0.0:
                best_loss, epochs_without_improvement = val_g, 0
                self.save()

            scheduler_g.step()
            scheduler_d.step()
        self.save(suffix="_last")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output_dir = args.output_dir or Path("runs") / args.variant
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    variant = VARIANTS[args.variant]
    dataset_args = dict(window_size=int(args.time_window * dsp.SAMPLE_RATE), snr_range=(args.min_snr, args.max_snr),
                        white_noise=variant.white_noise, revecho=variant.revecho, bandmask=variant.bandmask)
    train_set = load_partition(args.partition_dir, "train", **dataset_args)
    dev_set = load_partition(args.partition_dir, "dev", **dataset_args)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    dev_loader = DataLoader(dev_set, shuffle=False, **loader_args)
    log.info("%s: %d train and %d dev utterances on %s", args.variant, len(train_set), len(dev_set), args.device)
    Trainer(args, variant, train_loader, dev_loader).train()


if __name__ == "__main__":
    main()
