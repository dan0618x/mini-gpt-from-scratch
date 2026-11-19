"""Decoder-only transformer written with plain tensor operations.

Weights live in a nested dict instead of nn.Module so every matrix
multiplication is visible in the forward pass.
"""

import torch
import torch.nn.functional as F

from mini_gpt.config import BLOCK_SIZE, D_FF, D_HEAD, D_MODEL, EPS, INIT_STD, N_HEADS, N_LAYERS


def softmax_rows(x):
    # Subtract the row max first so exp() cannot overflow.
    shifted = x - x.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(shifted)
    return exp_scores / exp_scores.sum(dim=-1, keepdim=True)


def attention(query, key, value, is_causal):
    """Scaled dot-product attention; the causal mask hides future positions."""
    seq_len = query.shape[-2]
    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / d_k ** 0.5
    if is_causal:
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        scores = scores + causal_mask
    return softmax_rows(scores) @ value


def rms_norm(x):
    mean_square = (x ** 2).mean(dim=-1, keepdim=True)
    return x / ((mean_square + EPS) ** 0.5)


def make_layer_weights(init_std):
    return {
        "w_q_full": torch.randn(D_MODEL, D_MODEL) * init_std,
        "w_k_full": torch.randn(D_MODEL, D_MODEL) * init_std,
        "w_v_full": torch.randn(D_MODEL, D_MODEL) * init_std,
        "w_o": torch.randn(D_MODEL, D_MODEL) * init_std,
        "w_up": torch.randn(D_MODEL, D_FF) * init_std,
        "w_down": torch.randn(D_FF, D_MODEL) * init_std,
    }


def make_model(vocab_size, max_len=BLOCK_SIZE, init_std=INIT_STD):
    """Token table, position table, N_LAYERS blocks and the output head."""
    model = {
        "embedding_table": torch.randn(vocab_size, D_MODEL) * init_std,
        "position_table": torch.randn(max_len, D_MODEL) * init_std,
        "w_head": torch.randn(D_MODEL, vocab_size) * init_std,
        "layers": [],
    }
    for _ in range(N_LAYERS):
        model["layers"].append(make_layer_weights(init_std))
    return model


def model_parameters(model):
    """Collect every weight tensor and switch on gradient tracking."""
    params = [model["embedding_table"], model["position_table"], model["w_head"]]
    for layer in model["layers"]:
        params.extend(layer.values())
    for param in params:
        param.requires_grad_(True)
    return params


def multi_head_attention(x, is_causal, layer):
    """(..., seq, D_MODEL) -> same shape; all heads computed in one batched call."""
    lead_shape = x.shape[:-1]
    query_all = (x @ layer["w_q_full"]).reshape(*lead_shape, N_HEADS, D_HEAD).transpose(-3, -2)
    key_all = (x @ layer["w_k_full"]).reshape(*lead_shape, N_HEADS, D_HEAD).transpose(-3, -2)
    value_all = (x @ layer["w_v_full"]).reshape(*lead_shape, N_HEADS, D_HEAD).transpose(-3, -2)
    heads_all = attention(query_all, key_all, value_all, is_causal)
    merged = heads_all.transpose(-3, -2).reshape(*lead_shape, D_MODEL)
    return merged @ layer["w_o"]


def feed_forward(x, layer):
    return F.gelu(x @ layer["w_up"]) @ layer["w_down"]


def transformer_block(x, layer, is_causal):
    """Pre-norm residual block: attention then feed-forward."""
    x = x + multi_head_attention(rms_norm(x), is_causal, layer)
    x = x + feed_forward(rms_norm(x), layer)
    return x


def forward(model, token_ids, is_causal=True):
    """token_ids (..., seq) -> logits (..., seq, vocab)."""
    embedded = model["embedding_table"][token_ids]
    position = model["position_table"][torch.arange(token_ids.shape[-1])]
    x = embedded + position
    for layer in model["layers"]:
        x = transformer_block(x, layer, is_causal)
    return rms_norm(x) @ model["w_head"]


def compute_loss(model, x, y):
    logits = forward(model, x, True)
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
