"""Command line entry point: `jaxformer train | profile`.

Every `TrainConfig` field is overridable from the command line, so a run is fully
specified without editing source. `run.py` (executed on the Colab VM) can call
`main()` too, or build a `TrainConfig` directly.
"""

from __future__ import annotations

import argparse

from .config import TrainConfig


def _add_config_args(p: argparse.ArgumentParser) -> None:
    # only the knobs worth flipping from the shell; others use TrainConfig defaults
    p.add_argument("--attention", choices=["flash", "reference"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--n-blocks", type=int, dest="n_blocks")
    p.add_argument("--embed-dim", type=int, dest="embed_dim")
    p.add_argument("--hidden-dim", type=int, dest="hidden_dim")
    p.add_argument("--n-heads", type=int, dest="n_heads")
    p.add_argument("--rope-max-seq", type=int, dest="rope_max_seq")
    p.add_argument("--peak", type=float)
    p.add_argument("--warmup", type=int)
    p.add_argument("--end", type=float)
    p.add_argument("--n-train-batches", type=int, dest="n_train_batches")
    p.add_argument("--max-docs", type=int, dest="max_docs")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--resume", action="store_true", default=None)


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    fields = {f: getattr(args, f, None) for f in (
        "attention", "epochs", "batch_size", "n_blocks", "embed_dim", "hidden_dim",
        "n_heads", "rope_max_seq", "peak", "warmup", "end", "n_train_batches",
        "max_docs", "out_dir", "resume")}
    return TrainConfig().with_overrides(**fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jaxformer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="pretrain on FineWeb-EDU")
    _add_config_args(p_train)

    p_prof = sub.add_parser("profile", help="one warmup-excluded profiled step")
    _add_config_args(p_prof)
    p_prof.add_argument("--trace-dir", default="/content/jax-trace")

    args = parser.parse_args(argv)
    cfg = _config_from_args(args)

    if args.cmd == "train":
        from .train import run_training
        run_training(cfg)
    elif args.cmd == "profile":
        from .profile import run_profile
        run_profile(cfg, args.trace_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
