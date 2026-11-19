"""Decoding strategies: greedy, temperature sampling, top-k and top-p."""

import torch

from mini_gpt.config import BLOCK_SIZE
from mini_gpt.kv_cache import forward_step, new_caches
from mini_gpt.model import forward, softmax_rows


def greedy_next(last_logits):
    return last_logits.argmax().item()


def sample_next(last_logits, temperature):
    """Lower temperature sharpens the distribution, higher flattens it."""
    probs = softmax_rows(last_logits / temperature)
    return torch.multinomial(probs, 1).item()


def top_k_filter(last_logits, k):
    """Keep the k highest logits, set the rest to -inf."""
    threshold = torch.topk(last_logits, k).values[-1]
    filtered = last_logits.clone()
    filtered[last_logits < threshold] = float("-inf")
    return filtered


def top_p_filter(last_logits, p):
    """Keep the smallest set of tokens whose cumulative probability reaches p."""
    probs = softmax_rows(last_logits)
    sorted_probs = torch.sort(probs, descending=True).values
    cumulative = torch.cumsum(sorted_probs, dim=0)
    keep_count = (cumulative - sorted_probs < p).sum().item()
    threshold = sorted_probs[keep_count - 1]
    filtered = last_logits.clone()
    filtered[probs < threshold] = float("-inf")
    return filtered


def make_picker(strategy, temperature, top_k, top_p):
    """Return a function that maps the last-position logits to the next token id."""
    if strategy == "greedy":
        return greedy_next
    if strategy == "sample":
        return lambda logits: sample_next(logits, temperature)
    if strategy == "top_k":
        return lambda logits: sample_next(top_k_filter(logits, top_k), temperature)
    if strategy == "top_p":
        return lambda logits: sample_next(top_p_filter(logits, top_p), temperature)
    raise ValueError(f"unknown strategy: {strategy}")


def generate(model, start_ids, n_new, picker):
    """Sliding-window generation: the model always sees the last BLOCK_SIZE tokens."""
    ids = list(start_ids)
    with torch.no_grad():
        for _ in range(n_new):
            context = torch.tensor(ids[-BLOCK_SIZE:])
            logits = forward(model, context, True)
            ids.append(picker(logits[-1]))
    return ids


def generate_no_cache(model, start_ids, n_new):
    """Greedy baseline that re-runs the full sequence every step (no window, no cache)."""
    ids = list(start_ids)
    with torch.no_grad():
        for _ in range(n_new):
            logits = forward(model, torch.tensor(ids), True)
            ids.append(logits[-1].argmax().item())
    return ids


def generate_cached(model, start_ids, n_new):
    """Greedy generation with a KV cache: prompt token by token, then one new token per step.

    Positions are absolute, so total length must stay within the position table.
    """
    ids = list(start_ids)
    caches = new_caches(model)
    with torch.no_grad():
        for position, token_id in enumerate(ids):
            logits = forward_step(model, token_id, position, caches)
        for _ in range(n_new):
            next_id = logits[-1].argmax().item()
            ids.append(next_id)
            logits = forward_step(model, next_id, len(ids) - 1, caches)
    return ids
