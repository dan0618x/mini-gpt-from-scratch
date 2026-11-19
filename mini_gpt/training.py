"""Training loop with periodic train / validation evaluation, warm-up and clipping."""

import torch

from mini_gpt.config import EVAL_EVERY, EVAL_ITERS, LEARNING_RATE, MAX_GRAD_NORM, WARMUP_STEPS, WEIGHT_DECAY
from mini_gpt.data import get_batch
from mini_gpt.model import compute_loss, model_parameters


def estimate_loss(model, split_tensor):
    """Average loss over EVAL_ITERS random batches, without updating weights."""
    total = 0.0
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(split_tensor)
            total = total + compute_loss(model, xb, yb).item()
    return total / EVAL_ITERS


def warmup_lr(step, peak_lr, warmup_steps=WARMUP_STEPS):
    """Linear warm-up from peak_lr / warmup_steps up to peak_lr, then constant."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    return peak_lr


def grad_norm_of(params):
    return sum((param.grad ** 2).sum() for param in params).item() ** 0.5


def train_step(model, params, optimizer, train_tensor, use_clip):
    """One update; returns (batch loss, gradient norm before clipping)."""
    xb, yb = get_batch(train_tensor)
    loss = compute_loss(model, xb, yb)
    optimizer.zero_grad()
    loss.backward()
    if use_clip:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM).item()
    else:
        grad_norm = grad_norm_of(params)
    optimizer.step()
    return loss.item(), grad_norm


def train(model, train_tensor, val_tensor, n_steps, peak_lr=LEARNING_RATE, use_warmup=False, use_clip=False):
    """Train in place and return a history dict for plotting."""
    params = model_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=peak_lr, weight_decay=WEIGHT_DECAY)
    history = {"eval_steps": [], "train": [], "val": [], "batch_loss": [], "grad_norm": []}
    for step in range(n_steps):
        if step % EVAL_EVERY == 0:
            history["eval_steps"].append(step)
            history["train"].append(estimate_loss(model, train_tensor))
            history["val"].append(estimate_loss(model, val_tensor))
        if use_warmup:
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr(step, peak_lr)
        batch_loss, grad_norm = train_step(model, params, optimizer, train_tensor, use_clip)
        history["batch_loss"].append(batch_loss)
        history["grad_norm"].append(grad_norm)
    return history
