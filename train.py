"""Train the mini GPT on tiny-shakespeare, save a checkpoint and the loss curve."""

import math
import os
import time

import matplotlib.pyplot as plt
import torch

from mini_gpt.config import ASSETS_DIR, CHECKPOINT_PATH, DATA_PATH, N_STEPS, RANDOM_SEED
from mini_gpt.data import build_vocab, encode, load_text, split_data
from mini_gpt.model import make_model
from mini_gpt.training import estimate_loss, train

CURVE_PATH = os.path.join(ASSETS_DIR, "train_val_curve.png")


def load_dataset():
    text = load_text(DATA_PATH)
    char_to_id, id_to_char = build_vocab(text)
    train_tensor, val_tensor = split_data(torch.tensor(encode(text, char_to_id)))
    return char_to_id, train_tensor, val_tensor


def save_curve(history, final_step, final_train, final_val):
    steps = history["eval_steps"] + [final_step]
    plt.plot(steps, history["train"] + [final_train], marker="o", label="train")
    plt.plot(steps, history["val"] + [final_val], marker="o", label="val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.savefig(CURVE_PATH)
    plt.close()


def main():
    char_to_id, train_tensor, val_tensor = load_dataset()
    vocab_size = len(char_to_id)
    torch.manual_seed(RANDOM_SEED)
    model = make_model(vocab_size)
    print(f"vocab {vocab_size} | train tokens {len(train_tensor)} | val tokens {len(val_tensor)}")
    print(f"initial val loss {estimate_loss(model, val_tensor):.4f} (uniform guess = ln(vocab) = {math.log(vocab_size):.4f})")

    start_time = time.perf_counter()
    history = train(model, train_tensor, val_tensor, N_STEPS)
    elapsed = time.perf_counter() - start_time
    final_train = estimate_loss(model, train_tensor)
    final_val = estimate_loss(model, val_tensor)
    print(f"{N_STEPS} steps in {elapsed:.1f} s | final train {final_train:.4f} | final val {final_val:.4f}")

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({"model": model, "chars": sorted(char_to_id)}, CHECKPOINT_PATH)
    save_curve(history, N_STEPS, final_train, final_val)
    print(f"saved {CHECKPOINT_PATH} and {CURVE_PATH}")


if __name__ == "__main__":
    main()
