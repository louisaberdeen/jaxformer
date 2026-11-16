"""Loss / train / valid steps (fused with `lax.scan`) and the MLflow epoch loop.

Everything is driven by a `TrainConfig`; on Colab the driver `colab download`s outputs.
"""

from __future__ import annotations

import datetime
import time

import jax
import jax.numpy as jnp

from .config import TrainConfig
from .model import ModelParams, model_forward, set_attention_backend, set_rope_tables
from .optim import AdamW_optim_step, AdamWState, cosine_lr_schedule


# ==========================================================================
# Loss + steps
# ==========================================================================
@jax.jit
def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits, -1)
    return -jnp.mean(jnp.take_along_axis(logp, targets[..., None], -1))


@jax.jit
def masked_cross_entropy_loss(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    logp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(logp, targets[..., None], -1).squeeze(-1)  # target index across log probs
    nll = nll * mask
    count = mask.sum() * targets.shape[-1]
    return nll.sum() / count, count


@jax.jit
def masked_loss(params: ModelParams, X: jax.Array, targets: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = model_forward(params, X)
    loss, count = masked_cross_entropy_loss(logits, targets, mask)
    return (loss, count)


@jax.jit
def train_step(params, X, Y, mask, optim_state, lr):
    (loss_val, count), grads = jax.value_and_grad(masked_loss, has_aux=True)(params, X, Y, mask)
    params, optim_state = AdamW_optim_step(params, grads, optim_state, lr)
    return (params, optim_state), (loss_val, count)


@jax.jit
def train_batch_step(carry, xs):
    params, optim_state, lr = carry
    batches, mask = xs
    X, Y = batches[:, :-1], batches[:, 1:]
    (params, optim_state), out = train_step(params, X, Y, mask, optim_state, lr)
    return (params, optim_state, lr), out


@jax.jit
def train_epoch_step(params, optim_state, batches, masks, lr):
    (params, optim_state, lr), (batch_losses, batch_counts) = jax.lax.scan(
        train_batch_step, (params, optim_state, lr), (batches, masks))
    epoch_loss = jnp.sum(batch_losses * batch_counts) / jnp.sum(batch_counts)  # exact
    return params, optim_state, epoch_loss


@jax.jit
def valid_batch_step(params, xs):
    batches, mask = xs
    X, Y = batches[:, :-1], batches[:, 1:]
    loss_val, count = masked_loss(params, X, Y, mask)
    return params, (loss_val, count)


@jax.jit
def valid_epoch_step(params, batches, masks):
    params, (batch_losses, batch_counts) = jax.lax.scan(valid_batch_step, params, (batches, masks))
    epoch_loss = jnp.sum(batch_losses * batch_counts) / jnp.sum(batch_counts)
    return epoch_loss


# ==========================================================================
# Training loop
# ==========================================================================
def run_training(cfg: TrainConfig) -> str:
    """Full pretraining run driven by `cfg`. Returns the checkpoint path.

    Imports of heavy/optional deps (mlflow, tqdm, data pipeline) are local so the
    step functions above can be imported and tested without them.
    """
    import mlflow
    from tqdm import tqdm

    from . import checkpoint
    from .data import build_dataset

    set_attention_backend(cfg.attention)

    tokenizer, (train, valid, _test) = build_dataset(cfg)
    cfg = cfg.with_overrides(vocab_size=tokenizer.get_vocab_size())

    key = jax.random.key(cfg.seed)
    if cfg.resume and checkpoint.exists(cfg):
        params = checkpoint.load_params(cfg)          # re-derives + installs RoPE tables
        print("resumed from", cfg.ckpt_path)
    else:
        params, tables = ModelParams.random_init(
            key, vocab_size=cfg.vocab_size, n_blocks=cfg.n_blocks, embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads, max_seq=cfg.rope_max_seq)
        set_rope_tables(*tables)
    optim_state = AdamWState.init(params)

    train_batches, train_masks = batchify(train, cfg.batch_size)
    valid_batches, valid_masks = batchify(valid, cfg.batch_size)
    if cfg.n_train_batches is not None:               # profiling / smoke subset
        train_batches = train_batches[:cfg.n_train_batches]
        train_masks = train_masks[:cfg.n_train_batches]
    print("train batches", train_batches.shape, "valid batches", valid_batches.shape)

    mlflow.set_tracking_uri(f"sqlite:///{cfg.mlflow_db_path}")
    mlflow.config.enable_async_logging()
    mlflow.set_experiment(cfg.experiment_name())

    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dev = jax.devices()[0]
    n_examples = train_batches.shape[0] * train_batches.shape[1]
    buf: list[tuple[int, jax.Array, jax.Array]] = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "seq_len": cfg.rope_max_seq,
            "vocab_size": cfg.vocab_size, "n_blocks": cfg.n_blocks, "embed_dim": cfg.embed_dim,
            "hidden_dim": cfg.hidden_dim, "n_heads": cfg.n_heads,
            "lr_peak": cfg.peak, "lr_end": cfg.end, "warmup": cfg.warmup,
            "attention": cfg.attention,
        })

        t_window = time.perf_counter()   # start of the current 25-epoch logging window
        for i in tqdm(range(cfg.epochs), total=cfg.epochs):
            lr = cosine_lr_schedule(i, cfg.peak, cfg.warmup, cfg.end, cfg.total_steps)
            params, optim_state, epoch_loss = train_epoch_step(
                params, optim_state, train_batches, train_masks, lr)
            valid_loss = valid_epoch_step(params, valid_batches, valid_masks)

            buf.append((i, epoch_loss, valid_loss))
            if i % 25 == 0:
                steps, train_losses, valid_losses = zip(*buf); buf = []
                # first real sync point: drains every epoch queued since the last drain
                train_losses = jax.device_get(jnp.stack(train_losses))
                valid_losses = jax.device_get(jnp.stack(valid_losses))
                for s, tl, vl in zip(steps, train_losses, valid_losses):
                    mlflow.log_metrics({"train_loss": float(tl), "valid_loss": float(vl)}, step=s)
                mem = dev.memory_stats() or {}          # None on CPU, dict on TPU
                dt = time.perf_counter() - t_window     # wall time for the whole window...
                t_window = time.perf_counter()
                metrics = {
                    "lr": float(lr),
                    "examples_per_sec": n_examples * len(steps) / dt,   # ...divide by its epochs
                    "tokens_per_sec": n_examples * len(steps) * cfg.rope_max_seq / dt,
                }
                if "bytes_in_use" in mem:
                    metrics["tpu_mem_used_gb"] = mem["bytes_in_use"] / 1e9
                    metrics["tpu_mem_peak_gb"] = mem["peak_bytes_in_use"] / 1e9
                mlflow.log_metrics(metrics, step=i)

        ckpt_path = checkpoint.save_params(params, cfg)
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    print("checkpoint ->", ckpt_path)
    return ckpt_path


def batchify(arr: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array]:
    """Pad `arr` (n, seq) up to a whole number of batches and build the row mask,
    then reshape to (n_batches, batch_size, ...)."""
    n = len(arr)
    if n % batch_size != 0:
        tail = -n % batch_size
        padded = jnp.pad(arr, ((0, tail), (0, 0)))
        mask = jnp.ones(len(padded)).at[len(padded) - tail:].set(0)
    else:
        padded = arr
        mask = jnp.ones(n)
    n_batches = len(padded) // batch_size
    batches = padded[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    masks = mask[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    return batches, masks
