"""AdamW (hand-written, decoupled weight decay) and the cosine LR schedule.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import ModelParams


@dataclass
class AdamWState:
    m: ModelParams
    v: ModelParams
    t: jax.Array
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-9
    wd: float = 0.01

    @classmethod
    def init(cls, params: ModelParams, **hparams: float) -> "AdamWState":
        return cls(
            m=jax.tree.map(jnp.zeros_like, params),
            v=jax.tree.map(jnp.zeros_like, params),
            t=jnp.array(0, jnp.int32),
            **hparams,
        )


jax.tree_util.register_dataclass(
    AdamWState,
    data_fields=["m", "v", "t"],
    meta_fields=["b1", "b2", "eps", "wd"],
)


@jax.jit
def AdamW_optim_step(params: ModelParams, grad: ModelParams, optim_state: AdamWState,
                     lr: jax.Array) -> tuple[ModelParams, AdamWState]:
    _m, _v, _t = optim_state.m, optim_state.v, optim_state.t
    b1, b2, eps, wd = optim_state.b1, optim_state.b2, optim_state.eps, optim_state.wd

    t = _t + 1
    m = jax.tree.map(lambda _m, g: b1 * _m + (1 - b1) * g, _m, grad)
    v = jax.tree.map(lambda _v, g: b2 * _v + (1 - b2) * g ** 2, _v, grad)

    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    mhat = jax.tree.map(lambda m: m / bc1, m)
    vhat = jax.tree.map(lambda v: v / bc2, v)

    def weight_update(params, mhat, vhat, eps, wd, lr):
        # decoupled weight decay: wd*params added at the param, not folded into g
        return params - lr * (mhat / (jnp.sqrt(vhat) + eps) + wd * params)

    f = lambda p, m, v: weight_update(p, m, v, eps, wd, lr)
    params = jax.tree.map(f, params, mhat, vhat)
    return params, dataclasses.replace(optim_state, m=m, v=v, t=t)


@jax.jit
def cosine_lr_schedule(t: jax.Array | int, peak: float, warmup: int, end: float,
                       total: int) -> jax.Array:
    warm = peak * t / warmup
    prog = jnp.clip((t - warmup) / (total - warmup), 0.0, 1.0)
    cos = 0.5 * (1.0 + jnp.cos(jnp.pi * prog))
    decay = end + (peak - end) * cos
    return jnp.where(t < warmup, warm, decay)
