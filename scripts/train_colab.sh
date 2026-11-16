#!/usr/bin/env bash
# Build the jaxformer wheel and launch training DETACHED on a Colab TPU VM: training runs
# as a background process on the VM (logging to /content/train.log), so this returns right
# away and survives your terminal/connection dropping. Watch it with watch_train.sh, see
# live metrics with `FOLLOW=1 view_mlflow.sh`, and pull results with pull_train.sh when it
# finishes. The VM is LEFT RUNNING; the `colab stop` command is printed to copy-paste.
#
# One-time prerequisites (interactive, cannot be automated):
#   uv tool install google-colab-cli --python 3.13
#   gcloud auth application-default login --scopes=openid,\
#     https://www.googleapis.com/auth/cloud-platform,\
#     https://www.googleapis.com/auth/userinfo.email,\
#     https://www.googleapis.com/auth/colaboratory
#   colab whoami   # verify scopes/identity
#
# Usage:  ./scripts/train_colab.sh                       # full run
#         EPOCHS=2 MAX_DOCS=200 ./scripts/train_colab.sh # cheap end-to-end smoke
#         RESUME=1 ./scripts/train_colab.sh              # resume from artifacts/model_params.pkl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/_colab_common.sh"

S="${SESSION:-train}"
ACCEL="${ACCEL:---tpu v6e1}"                 # set ACCEL="" for CPU, or "--gpu T4"
cd "$ROOT"
mkdir -p artifacts

colab_build_wheel
colab_provision "$ACCEL"
colab_install_wheel
colab_pin_jax

echo "[driver] uploading resumable artifacts"
[ -f artifacts/model_params.pkl ]    && colab upload -s "$S" artifacts/model_params.pkl    /content/model_params.pkl
[ -f artifacts/tokenizer-4096.json ] && colab upload -s "$S" artifacts/tokenizer-4096.json /content/tokenizer-4096.json

echo "[driver] uploading run.py + launching training detached"
colab upload -s "$S" scripts/run.py /content/run.py
# Set the config env, then Popen run.py in the BACKGROUND (start_new_session so it isn't
# killed when this exec's websocket closes; the kernel persists, so its child survives).
# `-u` (unbuffered) so /content/train.log fills live for watch_train.sh to tail.
{
  echo "import os, sys, subprocess"
  echo "os.environ['MODE'] = 'train'"
  echo "os.environ['ATTENTION'] = '${ATTENTION_BACKEND:-flash}'"
  [ -n "${EPOCHS:-}" ]          && echo "os.environ['EPOCHS'] = '${EPOCHS}'"
  [ -n "${MAX_DOCS:-}" ]        && echo "os.environ['MAX_DOCS'] = '${MAX_DOCS}'"
  [ -n "${N_TRAIN_BATCHES:-}" ] && echo "os.environ['N_TRAIN_BATCHES'] = '${N_TRAIN_BATCHES}'"
  [ -n "${RESUME:-}" ]          && echo "os.environ['RESUME'] = '${RESUME}'"
  echo "log = open('/content/train.log', 'w')"
  echo "p = subprocess.Popen([sys.executable, '-u', '/content/run.py'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)"
  echo "open('/content/train.pid','w').write(str(p.pid)); print('launched pid', p.pid)"
} | colab exec -s "$S"

cat <<EOF

[driver] training launched DETACHED on '$S' -> /content/train.log
  watch output:  SESSION=$S ./scripts/watch_train.sh
  live metrics:  FOLLOW=1 SESSION=$S ./scripts/view_mlflow.sh
  pull results:  SESSION=$S ./scripts/pull_train.sh   (when the run finishes)
EOF
