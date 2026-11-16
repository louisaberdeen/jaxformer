# jaxformer

A from-scratch decoder-only transformer in **pure JAX** (no Flax/Optax/Haiku). Attention,
RoPE, LayerNorm, AdamW and the cosine LR schedule are all hand-written, and params are plain
dataclasses registered as pytrees. It trains on FineWeb-EDU on a Colab TPU, driven entirely
from a local terminal by the [`colab` CLI](https://pypi.org/project/google-colab-cli/).

Companion code for the post on [louisaberdeen.dev](https://louisaberdeen.dev).

## Setup

```bash
uv sync                                                   # local (CPU) env

uv tool install google-colab-cli --python 3.13            # the colab CLI
gcloud auth application-default login --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab --auth=adc whoami                                   # check identity + scopes
```

All four scopes are needed: a 403 from `colab.pa.googleapis.com` means `colaboratory` is missing.

## Run locally (CPU)

```bash
uv run pytest
uv run jaxformer train --attention reference --n-blocks 2 --embed-dim 64 \
  --hidden-dim 128 --n-heads 2 --rope-max-seq 64 --batch-size 4 \
  --max-docs 40 --epochs 3 --out-dir ./artifacts
```

You need `--attention reference` off-TPU, because `flash` is a Pallas TPU-only kernel.

## Train on a Colab TPU

```bash
EPOCHS=2 MAX_DOCS=200 ./scripts/train_colab.sh    # cheap end-to-end smoke test
./scripts/train_colab.sh                          # full run on a v6e-1 (returns at once)
./scripts/watch_train.sh                          # live tail of the VM's log (Ctrl-C = stop watching)
FOLLOW=1 ./scripts/view_mlflow.sh                 # MLflow UI on localhost:5000, re-pulled every 30s
./scripts/pull_train.sh                           # once the log shows "checkpoint ->"
RESUME=1 ./scripts/train_colab.sh                 # continue from artifacts/model_params.pkl
```

Training runs **detached** on the VM, so it survives your terminal closing. The driver
**leaves the VM running** and prints `colab stop -s <name>`. Run that when you're done,
because an idle VM keeps billing.

## Profile

```bash
./scripts/profile_colab.sh     # one warmup-excluded profiled step -> artifacts/jax-trace.tar.gz
./scripts/view_trace.sh        # TensorBoard on localhost:6006 (Profile tab)
```

## Knobs

Set these as environment variables on the scripts: `EPOCHS`, `MAX_DOCS`, `N_TRAIN_BATCHES`,
`RESUME`, `ATTENTION_BACKEND` (`flash`|`reference`), `SESSION` (VM name),
`ACCEL` (default `--tpu v6e1`; `"--gpu T4"`, or `""` for CPU), and `PIN_JAX` (default `0.6.2`).
The model hyperparameters are on `TrainConfig` in `src/jaxformer/config.py`.

## How the Colab flow works

1. `uv build` produces a wheel.
2. The driver reuses the named VM, or runs `colab new` to create one.
3. It uploads the wheel and pip-installs it `--no-deps`, so the VM's `jax[tpu]` stays in place.
4. It pins `jax[tpu]==0.6.2`. The VM's stock jaxlib 0.7.2 emits Mosaic v8, but its libtpu
   only reads v7 or older, which breaks the flash-attention Pallas kernel.
5. `colab exec` starts `scripts/run.py` as a background process. `colab exec` doesn't forward
   argv, so config is passed as env vars set in the persistent kernel.
6. Outputs (`mlflow.db`, checkpoint, tokenizer, trace) come back to `artifacts/` with `colab download`.

## Layout

```
src/jaxformer/
  config.py      TrainConfig: every hyperparameter on one dataclass
  model.py       params pytrees, RoPE, attention (flash + reference), the block stack
  optim.py       AdamW + cosine LR schedule
  train.py       lax.scan-fused train/valid steps + the MLflow epoch loop
  data.py        FineWeb download, BPE tokenizer, chunking, split
  checkpoint.py  pickle save/load (re-derives the RoPE tables on load)
  profile.py     one warmup-excluded profiled step
  cli.py         `jaxformer train | profile`
scripts/         colab CLI drivers + run.py (the on-VM entry point)
tests/           CPU smoke test + TPU-only flash/reference parity check
```
