"""Model: params pytrees, RoPE, attention, LayerNorm, the transformer stack.

Two module-level knobs:

* ``COS`` / ``SIN`` — the RoPE tables. They MUST stay module-level globals that the
  jitted ``transformer_block`` closes over, so that when jit traces (on the first call,
  *after* the tables are set) they bake in as XLA constants — no sine/cosine ops in the
  compiled HLO. Set them with :func:`set_rope_tables`.
* ``_ATTENTION`` — "flash" (Pallas/TPU) or "reference" (pure-jnp, CPU-safe). Read at
  trace time so the branch is resolved statically. Set with :func:`set_attention_backend`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

# --- RoPE tables (module globals; see docstring) --------------
COS: jax.Array | None = None
SIN: jax.Array | None = None


def set_rope_tables(cos: jax.Array, sin: jax.Array) -> None:
    """Install the RoPE tables the jitted blocks close over."""
    global COS, SIN
    COS, SIN = cos, sin


# --- attention backend selector ------------------------------------------
_ATTENTION = "flash"


def set_attention_backend(name: str) -> None:
    """Choose "flash" (Pallas/TPU) or "reference" (CPU-safe) attention."""
    global _ATTENTION
    if name not in ("flash", "reference"):
        raise ValueError(f"unknown attention backend: {name!r}")
    _ATTENTION = name


# ==========================================================================
# Block params
# ==========================================================================
@dataclass
class BlockParams:
    W_q: jax.Array
    W_k: jax.Array
    W_v: jax.Array
    W_1: jax.Array
    W_2: jax.Array
    gamma1: jax.Array
    beta1: jax.Array
    gamma2: jax.Array
    beta2: jax.Array
    n_heads: int = 1
    embed_dim: int = 256
    hidden_dim: int = 1024

    @classmethod
    def random_init(cls, key: jax.Array, embed_dim: int, hidden_dim: int, n_heads: int) -> "BlockParams":
        def init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            scale = 1.0 / shape[0]
            return jax.random.normal(key, shape) * scale

        shapes = [
            (embed_dim, embed_dim),   # W_q
            (embed_dim, embed_dim),   # W_k
            (embed_dim, embed_dim),   # W_v
            (embed_dim, hidden_dim),  # W_1
            (hidden_dim, embed_dim),  # W_2
        ]
        keys = jax.random.split(key, len(shapes))
        W_q, W_k, W_v, W_1, W_2 = [init(k, s) for k, s in zip(keys, shapes)]
        return cls(
            W_q, W_k, W_v, W_1, W_2,
            gamma1=jnp.ones(embed_dim), beta1=jnp.zeros(embed_dim),
            gamma2=jnp.ones(embed_dim), beta2=jnp.zeros(embed_dim),
            n_heads=n_heads, embed_dim=embed_dim, hidden_dim=hidden_dim,
        )


jax.tree_util.register_dataclass(
    BlockParams,
    data_fields=["W_q", "W_k", "W_v", "W_1", "W_2",
                 "gamma1", "beta1", "gamma2", "beta2"],
    meta_fields=["n_heads", "embed_dim", "hidden_dim"],
)


# ==========================================================================
# Attention pieces
# ==========================================================================
@jax.jit
def compute_qkv(X: jax.Array, W_q: jax.Array, W_k: jax.Array, W_v: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    Q = jnp.dot(X, W_q)
    K = jnp.dot(X, W_k)
    V = jnp.dot(X, W_v)
    return Q, K, V


@functools.partial(jax.jit, static_argnames=["n_heads"])
def split_heads(x: jax.Array, n_heads: int) -> jax.Array:
    *batch, seq, embed_dim = x.shape
    head_dim = embed_dim // n_heads
    x = x.reshape(*batch, seq, n_heads, head_dim)
    return jnp.swapaxes(x, -3, -2)


def merge_heads(x: jax.Array) -> jax.Array:
    x = jnp.swapaxes(x, -3, -2)
    return x.reshape(*x.shape[:-2], -1)


@jax.jit
def compute_multi_head_attention(Q: jax.Array, K: jax.Array, V: jax.Array) -> jax.Array:
    """Reference causal attention. Q, K, V: (batch, heads, seq, head_dim) -- already
    split, already roped. Same contract as compute_flash_attention, so swappable."""
    d_k = Q.shape[-1]
    QK = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(d_k)
    causal_mask = -jnp.inf * jnp.ones(QK.shape)
    causal_mask = jnp.triu(causal_mask, k=1)
    attention_weights = jax.nn.softmax(QK + causal_mask)
    attention_output = jnp.matmul(attention_weights, V)
    return merge_heads(attention_output)


@jax.jit
def compute_flash_attention(Q: jax.Array, K: jax.Array, V: jax.Array) -> jax.Array:
    # Pallas TPU flash attention; sm_scale is required (no implicit 1/sqrt(d)).
    # Imported lazily so the module still imports on a CPU-only box.
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    return merge_heads(fa.flash_attention(Q, K, V, causal=True,
                                          sm_scale=1.0 / math.sqrt(Q.shape[-1])))


def rope_tables(seq_len: int, dim: int, base: float = 10000.0) -> tuple[jax.Array, jax.Array]:
    """`dim` is the *per-head* width (head_dim), not embed_dim: RoPE is applied after
    the head split, so every head gets the same full spread of frequencies."""
    inv_freq = base ** (-jnp.arange(0, dim, 2) / dim)     # (dim/2,)
    angles = jnp.outer(jnp.arange(seq_len), inv_freq)      # (seq, dim/2)
    return jnp.cos(angles), jnp.sin(angles)


@jax.jit
def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    # x: (..., seq, head_dim), pairs adjacent dims (0,1),(2,3),...
    # cos/sin: (seq, head_dim/2) -- broadcasts over any leading batch/head axes
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = jnp.stack([x1 * cos - x2 * sin,
                     x1 * sin + x2 * cos], axis=-1)
    return out.reshape(x.shape)


# ==========================================================================
# Block / model
# ==========================================================================
@jax.jit
def layer_norm(X: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    mean = jnp.mean(X, axis=-1, keepdims=True)
    var = jnp.var(X, axis=-1, keepdims=True)
    return gamma * (X - mean) / jnp.sqrt(var + 1e-5) + beta


@jax.jit
def relu(X: jax.Array) -> jax.Array:
    return jnp.maximum(0, X)


@jax.jit
def feed_forward(X: jax.Array, W_1: jax.Array, W_2: jax.Array) -> jax.Array:
    return jnp.dot(relu(jnp.dot(X, W_1)), W_2)


@jax.jit
def transformer_block(X: jax.Array, params: BlockParams) -> jax.Array:
    Q, K, V = compute_qkv(layer_norm(X, params.gamma1, params.beta1),
                          params.W_q, params.W_k, params.W_v)
    seq = X.shape[-2]
    n_heads = params.n_heads

    Q = split_heads(Q, n_heads)
    K = split_heads(K, n_heads)
    V = split_heads(V, n_heads)

    Q = apply_rope(Q, COS[:seq], SIN[:seq])
    K = apply_rope(K, COS[:seq], SIN[:seq])

    # `_ATTENTION` is a plain Python string read at trace time: the branch is
    # resolved statically, so only the chosen kernel appears in the HLO.
    if _ATTENTION == "flash":
        X = X + compute_flash_attention(Q, K, V)
    else:
        X = X + compute_multi_head_attention(Q, K, V)

    X = X + feed_forward(layer_norm(X, params.gamma2, params.beta2), params.W_1, params.W_2)
    return X


@dataclass
class ModelParams:
    W_embed: jax.Array
    blocks: list[BlockParams]
    W_decode: jax.Array
    vocab_size: int
    n_blocks: int
    embed_dim: int
    hidden_dim: int

    @classmethod
    def random_init(cls, key: jax.Array, vocab_size: int, n_blocks: int, embed_dim: int,
                    hidden_dim: int, n_heads: int, max_seq: int) -> tuple["ModelParams", tuple[jax.Array, jax.Array]]:
        k_embed, k_decode, k_blocks = jax.random.split(key, 3)
        W_embed = jax.random.normal(k_embed, (vocab_size, embed_dim)) / jnp.sqrt(embed_dim)
        W_decode = jax.random.normal(k_decode, (embed_dim, vocab_size)) / jnp.sqrt(embed_dim)
        blocks = [BlockParams.random_init(k, embed_dim, hidden_dim, n_heads)
                  for k in jax.random.split(k_blocks, n_blocks)]
        return (cls(W_embed, blocks, W_decode, vocab_size, n_blocks, embed_dim, hidden_dim),
                rope_tables(max_seq, embed_dim // n_heads))


jax.tree_util.register_dataclass(
    ModelParams,
    data_fields=["W_embed", "blocks", "W_decode"],
    meta_fields=["vocab_size", "n_blocks", "embed_dim", "hidden_dim"],
)


@jax.jit
def model_forward(model_params: ModelParams, X: jax.Array) -> jax.Array:
    X = model_params.W_embed[X]
    for block in model_params.blocks:
        X = transformer_block(X, block)
    return X @ model_params.W_decode
