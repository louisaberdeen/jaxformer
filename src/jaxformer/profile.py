"""Profiling run: one warmup-excluded profiled `train_epoch_step`.

The warmup compile is deliberately outside the trace; per-batch HBM and FLOPs are
per-batch so a few batches profile the same as a full epoch.
"""

from __future__ import annotations

import shutil

import jax

from .config import TrainConfig
from .model import ModelParams, set_attention_backend, set_rope_tables
from .optim import AdamWState, cosine_lr_schedule
from .train import batchify, train_epoch_step


def run_profile(cfg: TrainConfig, trace_dir: str = "/content/jax-trace") -> str:
    from .data import build_dataset

    set_attention_backend(cfg.attention)
    tokenizer, (train, _valid, _test) = build_dataset(cfg)
    cfg = cfg.with_overrides(vocab_size=tokenizer.get_vocab_size())

    key = jax.random.key(cfg.seed)
    params, tables = ModelParams.random_init(
        key, vocab_size=cfg.vocab_size, n_blocks=cfg.n_blocks, embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads, max_seq=cfg.rope_max_seq)
    set_rope_tables(*tables)
    optim_state = AdamWState.init(params)

    n = cfg.n_train_batches or 8
    batches, masks = batchify(train, cfg.batch_size)
    batches, masks = batches[:n], masks[:n]
    lr = cosine_lr_schedule(1, cfg.peak, cfg.warmup, cfg.end, cfg.total_steps)

    print("warmup compile...")
    params, optim_state, loss = train_epoch_step(params, optim_state, batches, masks, lr)
    loss.block_until_ready()

    print("profiling...")
    jax.profiler.start_trace(trace_dir)
    params, optim_state, loss = train_epoch_step(params, optim_state, batches, masks, lr)
    loss.block_until_ready()
    jax.profiler.stop_trace()

    archive = shutil.make_archive(trace_dir, "gztar", trace_dir)
    print("trace ->", archive)
    return archive
