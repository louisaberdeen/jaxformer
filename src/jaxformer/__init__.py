"""jaxformer: a from-scratch decoder-only transformer in pure JAX.

Everything -- attention, RoPE, LayerNorm, AdamW, cosine LR -- is hand-written; params
are plain dataclasses registered as pytrees.
"""

from .config import TrainConfig
from .model import (
    BlockParams,
    ModelParams,
    model_forward,
    rope_tables,
    set_attention_backend,
    set_rope_tables,
)
from .optim import AdamW_optim_step, AdamWState, cosine_lr_schedule

__all__ = [
    "TrainConfig",
    "BlockParams",
    "ModelParams",
    "model_forward",
    "rope_tables",
    "set_attention_backend",
    "set_rope_tables",
    "AdamWState",
    "AdamW_optim_step",
    "cosine_lr_schedule",
]

__version__ = "0.1.0"
