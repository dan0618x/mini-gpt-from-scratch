"""Stress test: train with a high learning rate and compare warm-up / gradient clipping."""

import os
import time

import matplotlib.pyplot as plt
import torch

from mini_gpt.config import ASSETS_DIR, RANDOM_SEED
from mini_gpt.model import make_model
from mini_gpt.training import estimate_loss, train
from train import load_dataset

PEAK_LR = 1e-2          # deliberately higher than the 3e-3 used for the main run
COMPARE_STEPS = 600
LAST_STEPS = 50         # final steps averaged for the summary
CONFIGS = [("none", False, False), ("warmup", True, False), ("clip", False, True), ("warmup+clip", True, True)]
LOSS_PLOT = os.path.join(ASSETS_DIR, "warmup_clip_loss.png")
NORM_PLOT = os.path.join(ASSETS_DIR, "warmup_clip_grad_norm.png")


def run_config(vocab_size, train_tensor, val_tensor, use_warmup, use_clip):
    torch.manual_seed(RANDOM_SEED)
    model = make_model(vocab_size)
    start_time = time.perf_counter()
    history = train(model, train_tensor, val_tensor, COMPARE_STEPS, PEAK_LR, use_warmup, use_clip)
    return history, estimate_loss(model, val_tensor), time.perf_counter() - start_time


def plot(results, key, ylabel, path, log_scale):
    for name, history in results.items():
        plt.plot(history[key], label=name)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path)
    plt.close()


def main():
    char_to_id, train_tensor, val_tensor = load_dataset()
    results = {}
    print("| config | max batch loss (step) | max grad norm (step) | last-50 avg loss | val loss | time (s) |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, use_warmup, use_clip in CONFIGS:
        history, val_loss, seconds = run_config(len(char_to_id), train_tensor, val_tensor, use_warmup, use_clip)
        results[name] = history
        losses, norms = history["batch_loss"], history["grad_norm"]
        last_avg = sum(losses[-LAST_STEPS:]) / LAST_STEPS
        print(f"| {name} | {max(losses):.2f} ({losses.index(max(losses))}) | {max(norms):.2f} ({norms.index(max(norms))}) "
              f"| {last_avg:.4f} | {val_loss:.4f} | {seconds:.1f} |")
    plot(results, "batch_loss", "train batch loss", LOSS_PLOT, False)
    plot(results, "grad_norm", "grad norm (before clipping)", NORM_PLOT, True)
    print(f"saved {LOSS_PLOT} and {NORM_PLOT}")


if __name__ == "__main__":
    main()
