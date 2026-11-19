"""Correctness checks for attention masking, the KV cache and the sampling filters.

Run with: python -m unittest discover tests
"""

import math
import unittest

import torch

from mini_gpt.config import BLOCK_SIZE, D_MODEL, N_LAYERS
from mini_gpt.kv_cache import kv_cache_bytes, kv_cache_formula, prefill, prefill_step_by_step
from mini_gpt.model import compute_loss, forward, make_model
from mini_gpt.sampling import generate_cached, generate_no_cache, top_k_filter, top_p_filter

VOCAB_SIZE = 65
SEQ_LEN = 12
SEED = 0
ATOL = 1e-5
LOSS_TOLERANCE = 0.05
BYTES_PER_FLOAT32 = 4
N_GENERATE = 10
TOP_K = 3
TOP_P = 0.8


def random_ids(length):
    return torch.randint(VOCAB_SIZE, (length,))


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.model = make_model(VOCAB_SIZE)

    def test_causal_mask_hides_future_tokens(self):
        ids = random_ids(SEQ_LEN)
        changed = ids.clone()
        changed[-1] = (changed[-1] + 1) % VOCAB_SIZE
        with torch.no_grad():
            before = forward(self.model, ids)
            after = forward(self.model, changed)
        self.assertTrue(torch.allclose(before[:-1], after[:-1], atol=ATOL))

    def test_initial_loss_is_close_to_uniform_guess(self):
        x = random_ids(BLOCK_SIZE).unsqueeze(0)
        y = random_ids(BLOCK_SIZE).unsqueeze(0)
        with torch.no_grad():
            loss = compute_loss(self.model, x, y).item()
        self.assertAlmostEqual(loss, math.log(VOCAB_SIZE), delta=LOSS_TOLERANCE)


class KVCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.model = make_model(VOCAB_SIZE)

    def test_cached_generation_matches_full_recompute(self):
        prompt = random_ids(SEQ_LEN).tolist()
        self.assertEqual(generate_no_cache(self.model, prompt, N_GENERATE),
                         generate_cached(self.model, prompt, N_GENERATE))

    def test_batched_prefill_matches_token_by_token(self):
        ids = random_ids(SEQ_LEN)
        with torch.no_grad():
            logits_full, caches_full = prefill(self.model, ids)
            logits_step, caches_step = prefill_step_by_step(self.model, ids.tolist())
        self.assertTrue(torch.allclose(logits_full[-1], logits_step[-1], atol=ATOL))
        for cache_full, cache_step in zip(caches_full, caches_step):
            self.assertTrue(torch.allclose(cache_full["k"], cache_step["k"], atol=ATOL))
            self.assertTrue(torch.allclose(cache_full["v"], cache_step["v"], atol=ATOL))

    def test_cache_size_matches_formula(self):
        with torch.no_grad():
            _, caches = prefill(self.model, random_ids(SEQ_LEN))
        expected = kv_cache_formula(N_LAYERS, SEQ_LEN, D_MODEL, BYTES_PER_FLOAT32)
        self.assertEqual(kv_cache_bytes(caches), expected)


class SamplingFilterTest(unittest.TestCase):
    def test_top_k_keeps_exactly_k_tokens(self):
        logits = torch.tensor([3.0, 1.0, 4.0, 1.5, 5.0, 9.0])
        kept = torch.isfinite(top_k_filter(logits, TOP_K)).sum().item()
        self.assertEqual(kept, TOP_K)

    def test_top_p_keeps_smallest_set_reaching_p(self):
        probs = torch.tensor([0.5, 0.25, 0.15, 0.1])
        filtered = top_p_filter(torch.log(probs), TOP_P)
        self.assertEqual(torch.isfinite(filtered).tolist(), [True, True, True, False])


if __name__ == "__main__":
    unittest.main()
