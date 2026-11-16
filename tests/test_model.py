"""CPU-safe smoke tests plus a TPU-only flash/reference parity check."""

import jax
import jax.numpy as jnp
import pytest

from jaxformer.config import TrainConfig
from jaxformer.model import (
    ModelParams,
    compute_flash_attention,
    compute_multi_head_attention,
    model_forward,
    set_attention_backend,
    set_rope_tables,
)


def _has_tpu() -> bool:
    return any(d.platform == "tpu" for d in jax.devices())


def _tiny_model(cfg):
    key = jax.random.key(0)
    params, tables = ModelParams.random_init(
        key, vocab_size=cfg.vocab_size, n_blocks=cfg.n_blocks, embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads, max_seq=cfg.rope_max_seq)
    set_rope_tables(*tables)
    return params


def test_forward_shape_reference_cpu():
    """The reference backend runs anywhere; forward produces (batch, seq, vocab)."""
    set_attention_backend("reference")
    cfg = TrainConfig(vocab_size=13, n_blocks=2, embed_dim=32, hidden_dim=64,
                      n_heads=2, rope_max_seq=16)
    params = _tiny_model(cfg)
    X = jax.random.randint(jax.random.key(1), (2, cfg.rope_max_seq), 0, cfg.vocab_size)
    logits = model_forward(params, X)
    assert logits.shape == (2, cfg.rope_max_seq, cfg.vocab_size)
    assert jnp.isfinite(logits).all()


@pytest.mark.skipif(not _has_tpu(), reason="flash attention is Pallas/TPU only")
def test_flash_matches_reference():
    """Pallas flash attention matches the naive causal reference."""
    key = jax.random.key(0)
    Q, K, V = jax.random.normal(key, (3, 2, 8, 64, 16))
    assert jnp.allclose(compute_flash_attention(Q, K, V),
                        compute_multi_head_attention(Q, K, V), atol=1e-2)
