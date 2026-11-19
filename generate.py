"""Generate text from a trained checkpoint with a chosen decoding strategy."""

import argparse

import torch

from mini_gpt.config import CHECKPOINT_PATH, RANDOM_SEED
from mini_gpt.data import decode, encode
from mini_gpt.sampling import generate, make_picker

DEFAULT_PROMPT = "ROMEO:\n"
DEFAULT_LENGTH = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 10
DEFAULT_TOP_P = 0.9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--strategy", choices=["greedy", "sample", "top_k", "top_p"], default="top_p")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")
    chars = checkpoint["chars"]
    char_to_id = {char: index for index, char in enumerate(chars)}
    id_to_char = dict(enumerate(chars))

    torch.manual_seed(args.seed)
    picker = make_picker(args.strategy, args.temperature, args.top_k, args.top_p)
    ids = generate(checkpoint["model"], encode(args.prompt, char_to_id), args.length, picker)
    print(decode(ids, id_to_char))


if __name__ == "__main__":
    main()
