"""On-VM entry point. `colab exec -s <name> -f scripts/run.py` runs this on the Colab
kernel after the jaxformer wheel has been installed.

`colab exec` does not forward argv, so overrides come from environment variables (the
driver scripts set them with a prior `colab exec`). `MODE` selects train vs profile.
All artifacts land under /content; the local driver `colab download`s them before you
stop the VM.
"""

import os

from jaxformer.config import TrainConfig


def _int(name):
    v = os.environ.get(name)
    return int(v) if v else None


cfg = TrainConfig().with_overrides(
    out_dir="/content",
    attention=os.environ.get("ATTENTION", "flash"),
    epochs=_int("EPOCHS"),
    n_train_batches=_int("N_TRAIN_BATCHES"),
    max_docs=_int("MAX_DOCS"),
    resume=os.environ.get("RESUME") == "1",
)

if __name__ == "__main__":
    if os.environ.get("MODE") == "profile":
        from jaxformer.profile import run_profile
        run_profile(cfg, os.environ.get("TRACE_DIR", "/content/jax-trace"))
    else:
        from jaxformer.train import run_training
        run_training(cfg)
