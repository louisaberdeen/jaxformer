"""Central configuration for a training / profiling run.

Every hyperparameter lives on one dataclass, so a run is fully specified by a `TrainConfig` instance and nothing
has to be edited in the source to change a hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional


@dataclass
class TrainConfig:
    # --- model -----------------------------------------------------------
    # vocab_size is filled in at runtime from the trained tokenizer.
    vocab_size: Optional[int] = None
    n_blocks: int = 10
    embed_dim: int = 1024
    hidden_dim: int = 4 * 1024
    n_heads: int = 8
    rope_max_seq: int = 512

    # which attention kernel `transformer_block` uses. "flash" is Pallas/TPU
    # only; "reference" is the pure-jnp causal path that also runs on CPU.
    attention: Literal["flash", "reference"] = "flash"

    # --- data ------------------------------------------------------------
    hf_repo: str = "karpathy/fineweb-edu-100b-shuffle"
    hf_filename: str = "shard_00000.parquet"
    max_docs: Optional[int] = None          # cap docs for quick local runs
    tokenizer_vocab_size: int = 4096
    test_frac: float = 0.1
    valid_frac: float = 0.1
    batch_size: int = 64
    seed: int = 42

    # --- optimisation / schedule ----------------------------------------
    epochs: int = 1000
    peak: float = 3e-4
    warmup: int = 50
    end: float = 3e-7
    # `total` for the cosine decay; defaults to `epochs` when left None.
    total: Optional[int] = None
    # if set, use only this many training batches per epoch (profiling / smoke).
    n_train_batches: Optional[int] = None

    # --- io --------------------------------------------------------------
    # On a Colab VM these default under /content; the driver `colab download`s
    # them back before `colab stop`.
    out_dir: str = "/content"
    tokenizer_name: str = "tokenizer-4096.json"
    ckpt_name: str = "model_params.pkl"
    mlflow_db_name: str = "mlflow.db"
    experiment: Optional[str] = None        # None -> derived from model shape
    resume: bool = False                    # load ckpt/tokenizer from out_dir if present

    def with_overrides(self, **kw) -> "TrainConfig":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})

    @property
    def total_steps(self) -> int:
        return self.total if self.total is not None else self.epochs

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.n_heads

    # --- resolved paths --------------------------------------------------
    @property
    def tokenizer_path(self) -> str:
        return f"{self.out_dir}/{self.tokenizer_name}"

    @property
    def ckpt_path(self) -> str:
        return f"{self.out_dir}/{self.ckpt_name}"

    @property
    def mlflow_db_path(self) -> str:
        return f"{self.out_dir}/{self.mlflow_db_name}"

    def experiment_name(self) -> str:
        if self.experiment:
            return self.experiment
        return f"fineweb-edu-{self.n_blocks}L-{self.embed_dim}d-{self.n_heads}h"
