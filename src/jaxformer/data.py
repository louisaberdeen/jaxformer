"""Data pipeline: FineWeb-EDU download, BPE tokenizer, chunking, train/val/test split.

The tokenizer is trained once and saved to `cfg.tokenizer_path`;
on later runs (and once the driver has `colab download`ed it) it is reused rather than
retrained.
"""

from __future__ import annotations

import os
import random

import jax.numpy as jnp

from .config import TrainConfig


def download_fineweb(cfg: TrainConfig) -> list[str]:
    """Download one pre-shuffled FineWeb-EDU parquet shard and return its texts."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    shard_path = hf_hub_download(repo_id=cfg.hf_repo, filename=cfg.hf_filename, repo_type="dataset")
    df = pd.read_parquet(shard_path)
    texts = df["text"].tolist()
    if cfg.max_docs is not None:
        texts = texts[:cfg.max_docs]
    print(len(texts), "documents,", sum(len(t) for t in texts), "chars")
    return texts


def train_tokenizer(texts: list[str], cfg: TrainConfig):
    """Train a ByteLevel BPE tokenizer and save it to `cfg.tokenizer_path`."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.tokenizer_vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(cfg.tokenizer_path)
    return tokenizer


def load_tokenizer(cfg: TrainConfig):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(cfg.tokenizer_path)


def get_or_train_tokenizer(texts: list[str], cfg: TrainConfig):
    if os.path.exists(cfg.tokenizer_path):
        print("reusing tokenizer", cfg.tokenizer_path)
        return load_tokenizer(cfg)
    return train_tokenizer(texts, cfg)


def chunk_texts(texts: list[str], tokenizer, seq_len: int) -> list[list[int]]:
    """Encode all texts and pack into fixed (seq_len+1) chunks, carrying the leftover
    tail of one document into the next. The +1 is the shift target."""
    data: list[list[int]] = []
    tail: list[int] = []
    for text in texts:
        ids = tail + tokenizer.encode(text).ids
        tail = []
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i + seq_len + 1]
            if len(chunk) < seq_len + 1:
                tail = chunk
            else:
                data.append(chunk)
    return data


def split_data(data: list[list[int]], cfg: TrainConfig):
    """Shuffle and split into (train, valid, test) jnp arrays."""
    random.seed(cfg.seed)
    data = list(data)
    random.shuffle(data)

    n = len(data)
    test_slice = int(n * cfg.test_frac)
    valid_slice = int(n * cfg.valid_frac)

    test = jnp.array(data[:test_slice])
    valid = jnp.array(data[test_slice:test_slice + valid_slice])
    train = jnp.array(data[test_slice + valid_slice:])
    print(len(train), "train,", len(test), "test,", len(valid), "valid chunks")
    return train, valid, test


def build_dataset(cfg: TrainConfig):
    """End-to-end: download -> tokenizer -> chunk -> split. Returns (tokenizer, (train, valid, test))."""
    texts = download_fineweb(cfg)
    tokenizer = get_or_train_tokenizer(texts, cfg)
    data = chunk_texts(texts, tokenizer, cfg.rope_max_seq)
    return tokenizer, split_data(data, cfg)
