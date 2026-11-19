"""KV cache: keep every layer's keys and values so each new token costs O(1) projections.

With a causal mask, earlier tokens never attend to later ones, so their K/V
do not change when a token is appended. They can be stored and reused.
The cache stores the pre-split (tokens, D_MODEL) tensors; heads are split on use.
"""

import torch

from mini_gpt.config import D_HEAD, D_MODEL, N_HEADS
from mini_gpt.model import attention, feed_forward, multi_head_attention, rms_norm


def split_heads(x):
    """(tokens, D_MODEL) -> (N_HEADS, tokens, D_HEAD)"""
    return x.reshape(x.shape[0], N_HEADS, D_HEAD).transpose(0, 1)


def merge_heads(heads):
    """(N_HEADS, tokens, D_HEAD) -> (tokens, D_MODEL)"""
    return heads.transpose(0, 1).reshape(heads.shape[1], D_MODEL)


def new_cache():
    return {"k": torch.zeros(0, D_MODEL), "v": torch.zeros(0, D_MODEL)}


def new_caches(model):
    caches = []
    for _ in model["layers"]:
        caches.append(new_cache())
    return caches


def cached_multi_head(x_new, cache, layer):
    """Append the new token's K/V, then attend from its query to the whole cache.

    No mask is needed: the cache only holds past tokens plus the token itself.
    """
    query_new = x_new @ layer["w_q_full"]
    cache["k"] = torch.cat([cache["k"], x_new @ layer["w_k_full"]], dim=0)
    cache["v"] = torch.cat([cache["v"], x_new @ layer["w_v_full"]], dim=0)
    heads = attention(split_heads(query_new), split_heads(cache["k"]), split_heads(cache["v"]), False)
    return merge_heads(heads) @ layer["w_o"]


def forward_step(model, new_id, position, caches):
    """Decode one token at absolute `position`; returns logits of shape (1, vocab)."""
    embedded = model["embedding_table"][torch.tensor([new_id])]
    x = embedded + model["position_table"][position:position + 1]
    for layer_index, layer in enumerate(model["layers"]):
        x = x + cached_multi_head(rms_norm(x), caches[layer_index], layer)
        x = x + feed_forward(rms_norm(x), layer)
    return rms_norm(x) @ model["w_head"]


def prefill(model, token_ids):
    """Run the whole prompt in one pass and fill every layer's cache at once."""
    caches = new_caches(model)
    x = model["embedding_table"][token_ids] + model["position_table"][torch.arange(token_ids.shape[-1])]
    for layer_index, layer in enumerate(model["layers"]):
        normed = rms_norm(x)
        caches[layer_index]["k"] = normed @ layer["w_k_full"]
        caches[layer_index]["v"] = normed @ layer["w_v_full"]
        x = x + multi_head_attention(normed, True, layer)
        x = x + feed_forward(rms_norm(x), layer)
    return rms_norm(x) @ model["w_head"], caches


def prefill_step_by_step(model, id_list):
    """Reference prefill that feeds the prompt one token at a time."""
    caches = new_caches(model)
    logits = None
    for position, token_id in enumerate(id_list):
        logits = forward_step(model, token_id, position, caches)
    return logits, caches


def kv_cache_bytes(caches):
    total = 0
    for cache in caches:
        total = total + cache["k"].numel() * cache["k"].element_size()
        total = total + cache["v"].numel() * cache["v"].element_size()
    return total


def kv_cache_formula(n_layers, n_tokens, kv_width, bytes_per_value):
    """2 (K and V) x layers x tokens x width x bytes."""
    return 2 * n_layers * n_tokens * kv_width * bytes_per_value


def scores_bytes(n_heads, n_queries, n_keys, bytes_per_value):
    """Size of one layer's attention score table (heads, queries, keys)."""
    return n_heads * n_queries * n_keys * bytes_per_value
