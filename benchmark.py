"""Measure what the KV cache buys: generation time, memory, and prefill vs decode cost.

Timing only depends on tensor shapes, so a randomly initialised model with a
longer position table is used; the generated text itself is meaningless.
"""

import os
import time

import matplotlib.pyplot as plt
import torch

from mini_gpt.config import ASSETS_DIR, D_MODEL, DATA_PATH, N_HEADS, N_LAYERS, RANDOM_SEED
from mini_gpt.data import build_vocab, load_text
from mini_gpt.kv_cache import (forward_step, kv_cache_bytes, kv_cache_formula, prefill,
                               prefill_step_by_step, scores_bytes)
from mini_gpt.model import make_model
from mini_gpt.sampling import generate_cached, generate_no_cache

LONG_MAX_LEN = 1024                         # position-table rows for the benchmark model
PROMPT_TOKENS = 7                           # length of the generation prompt
GENERATION_LENGTHS = [25, 100, 200, 400, 800]
PROMPT_LENGTHS = [50, 200, 800]
DECODE_STEPS = 20                           # tokens decoded after prefill to average per-token cost
BYTES_PER_FLOAT32 = 4
BYTES_PER_MB = 1024 ** 2
MS_PER_SECOND = 1000
SAME_ATOL = 1e-5
PLOT_PATH = os.path.join(ASSETS_DIR, "generation_time.png")


def timed(function, *args):
    start_time = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start_time


def benchmark_generation(model):
    prompt = list(range(PROMPT_TOKENS))
    no_cache_times, cache_times = [], []
    print("| generated tokens | no cache (s) | KV cache (s) | speed-up | same output |")
    print("|---:|---:|---:|---:|:---:|")
    for n_new in GENERATION_LENGTHS:
        ids_plain, seconds_plain = timed(generate_no_cache, model, prompt, n_new)
        ids_cached, seconds_cached = timed(generate_cached, model, prompt, n_new)
        no_cache_times.append(seconds_plain)
        cache_times.append(seconds_cached)
        print(f"| {n_new} | {seconds_plain:.3f} | {seconds_cached:.3f} | {seconds_plain / seconds_cached:.1f}x | {ids_plain == ids_cached} |")
    plt.plot(GENERATION_LENGTHS, no_cache_times, marker="o", label="no cache")
    plt.plot(GENERATION_LENGTHS, cache_times, marker="o", label="KV cache")
    plt.xlabel("generated tokens")
    plt.ylabel("seconds")
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()


def benchmark_memory(model):
    n_tokens = PROMPT_TOKENS + GENERATION_LENGTHS[-1]
    with torch.no_grad():
        _, caches = prefill(model, torch.arange(n_tokens) % model["w_head"].shape[1])
    measured = kv_cache_bytes(caches)
    formula = kv_cache_formula(N_LAYERS, n_tokens, D_MODEL, BYTES_PER_FLOAT32)
    full_scores = scores_bytes(N_HEADS, n_tokens, n_tokens, BYTES_PER_FLOAT32) / BYTES_PER_MB
    cached_scores = scores_bytes(N_HEADS, 1, n_tokens, BYTES_PER_FLOAT32) / BYTES_PER_MB
    print(f"\nKV cache at {n_tokens} tokens: measured {measured} B, formula {formula} B, "
          f"{measured // n_tokens} B per token")
    print(f"attention score table per layer: full recompute {full_scores:.3f} MB vs cached step {cached_scores:.4f} MB")


def caches_match(caches_a, caches_b):
    all_same = True
    for cache_a, cache_b in zip(caches_a, caches_b):
        for name in ["k", "v"]:
            all_same = all_same and torch.allclose(cache_a[name], cache_b[name], atol=SAME_ATOL)
    return all_same


def decode_ms_per_token(model, logits, caches, start_position):
    start_time = time.perf_counter()
    for step in range(DECODE_STEPS):
        logits = forward_step(model, logits[-1].argmax().item(), start_position + step, caches)
    return (time.perf_counter() - start_time) * MS_PER_SECOND / DECODE_STEPS


def benchmark_prefill(model):
    vocab_size = model["w_head"].shape[1]
    print("\n| prompt tokens | batched prefill (ms) | token-by-token prefill (ms) | decode per token (ms) | caches match |")
    print("|---:|---:|---:|---:|:---:|")
    with torch.no_grad():
        prefill(model, torch.arange(PROMPT_TOKENS))  # warm-up, not timed
        for prompt_len in PROMPT_LENGTHS:
            id_list = [index % vocab_size for index in range(prompt_len)]
            (logits_full, caches_full), seconds_full = timed(prefill, model, torch.tensor(id_list))
            (logits_step, caches_step), seconds_step = timed(prefill_step_by_step, model, id_list)
            same = caches_match(caches_full, caches_step) and torch.allclose(logits_full[-1], logits_step[-1], atol=SAME_ATOL)
            decode_ms = decode_ms_per_token(model, logits_full, caches_full, prompt_len)
            print(f"| {prompt_len} | {seconds_full * MS_PER_SECOND:.2f} | {seconds_step * MS_PER_SECOND:.2f} | {decode_ms:.3f} | {same} |")


def main():
    vocab_size = len(build_vocab(load_text(DATA_PATH))[0])
    torch.manual_seed(RANDOM_SEED)
    model = make_model(vocab_size, max_len=LONG_MAX_LEN)
    benchmark_generation(model)
    benchmark_memory(model)
    benchmark_prefill(model)
    print(f"\nsaved {PLOT_PATH}")


if __name__ == "__main__":
    main()
