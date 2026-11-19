"""Character-level tokenizer and random batch sampling."""

import torch

from mini_gpt.config import BATCH_SIZE, BLOCK_SIZE, TRAIN_RATIO


def load_text(path):
    with open(path, "r", encoding="utf-8") as text_file:
        return text_file.read()


def build_vocab(text):
    """Map every distinct character to an integer id and back."""
    chars = sorted(set(text))
    char_to_id = {}
    id_to_char = {}
    for index, char in enumerate(chars):
        char_to_id[char] = index
        id_to_char[index] = char
    return char_to_id, id_to_char


def encode(text, char_to_id):
    id_list = []
    for char in text:
        id_list.append(char_to_id[char])
    return id_list


def decode(ids, id_to_char):
    text = ""
    for index in ids:
        text = text + id_to_char[index]
    return text


def split_data(token_tensor):
    """Contiguous split so validation text is never seen during training."""
    split_index = int(len(token_tensor) * TRAIN_RATIO)
    return token_tensor[:split_index], token_tensor[split_index:]


def get_batch(split_tensor, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """Return inputs x and next-token targets y, both shaped (batch, block)."""
    starts = torch.randint(len(split_tensor) - block_size, (batch_size,))
    x_list = []
    y_list = []
    for start in starts:
        x_list.append(split_tensor[start:start + block_size])
        y_list.append(split_tensor[start + 1:start + block_size + 1])
    return torch.stack(x_list), torch.stack(y_list)
