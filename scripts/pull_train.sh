#!/usr/bin/env bash
# Pull a training run's artifacts from the VM into artifacts/ — run when watch_train.sh
# shows "checkpoint ->". The VM is left running (stop it yourself with `colab stop -s <S>`).
#
# Usage:  SESSION=train ./scripts/pull_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="${SESSION:-train}"
cd "$ROOT"
mkdir -p artifacts

echo "[pull] downloading artifacts from '$S'"
# full dest FILE paths — `colab download` errors "Is a directory" on a dir dest.
colab download -s "$S" /content/model_params.pkl    artifacts/model_params.pkl    || true
colab download -s "$S" /content/tokenizer-4096.json artifacts/tokenizer-4096.json || true
colab download -s "$S" /content/mlflow.db           artifacts/mlflow.db           || true
echo "[pull] done -> artifacts/"
