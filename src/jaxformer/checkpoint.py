"""Pickle checkpointing.

A pickled checkpoint stores only the params -- it skips
`ModelParams.random_init`, which is what normally hands back the RoPE tables. So the
load path MUST re-derive `COS`/`SIN` from the loaded params and install them via
`set_rope_tables`, or `apply_rope` broadcasts against a stale/wrong-width table.
"""

from __future__ import annotations

import os
import pickle

import jax

from .config import TrainConfig
from .model import ModelParams, rope_tables, set_rope_tables


def exists(cfg: TrainConfig) -> bool:
    return os.path.exists(cfg.ckpt_path)


def save_params(params: ModelParams, cfg: TrainConfig) -> str:
    with open(cfg.ckpt_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    return cfg.ckpt_path


def load_params(cfg: TrainConfig) -> ModelParams:
    with open(cfg.ckpt_path, "rb") as f:
        params: ModelParams = pickle.load(f)
    # re-derive the RoPE tables from the loaded geometry and install them.
    n_heads = params.blocks[0].n_heads
    set_rope_tables(*rope_tables(cfg.rope_max_seq, params.embed_dim // n_heads))
    return params
