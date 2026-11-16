#!/usr/bin/env bash
# Build the jaxformer wheel, run ONE warmup-excluded profiled step on a Colab TPU VM, and
# pull the profiler trace back to artifacts/jax-trace.tar.gz (view it with view_trace.sh).
# The VM is LEFT RUNNING at the end; the `colab stop` command is printed to copy-paste.
#
# Prereqs + colab CLI details: see the header of train_colab.sh.
#
# Usage:  ./scripts/profile_colab.sh                              # profile the default model
#         N_TRAIN_BATCHES=4 MAX_DOCS=200 ./scripts/profile_colab.sh  # quick trace
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/_colab_common.sh"

S="${SESSION:-profile}"
ACCEL="${ACCEL:---tpu v6e1}"
cd "$ROOT"
mkdir -p artifacts

colab_build_wheel
colab_provision "$ACCEL"
colab_install_wheel
colab_pin_jax

echo "[driver] forwarding profile config into the kernel"
{
  echo "import os"
  echo "os.environ['MODE'] = 'profile'"
  echo "os.environ['ATTENTION'] = '${ATTENTION_BACKEND:-flash}'"
  echo "os.environ['TRACE_DIR'] = '${TRACE_DIR:-/content/jax-trace}'"
  echo "os.environ['N_TRAIN_BATCHES'] = '${N_TRAIN_BATCHES:-8}'"
  # small default: a roofline profile needs a few batches, not the whole shard (a full
  # tokenizer build over ~250M chars just blows the exec timeout for no benefit).
  echo "os.environ['MAX_DOCS'] = '${MAX_DOCS:-400}'"
} | colab exec -s "$S"

echo "[driver] profiling"
colab exec -s "$S" --timeout "${TIMEOUT:-1800}" -f scripts/run.py

# run_profile can crash on the VM while `colab exec` still exits 0, so confirm the archive
# exists before downloading — otherwise a failed run looks "done". Sentinel + grep, so we
# don't depend on how colab exec maps a remote SystemExit to a shell exit code.
echo "[driver] checking the trace was produced"
if echo "import os; print('TRACE_OK' if os.path.exists('/content/jax-trace.tar.gz') else 'TRACE_MISSING')" \
     | colab exec -s "$S" --timeout 60 2>/dev/null | grep -q TRACE_OK; then
  echo "[driver] downloading trace"
  # NB: `colab download` needs a full dest FILE path — a dir errors "Is a directory".
  colab download -s "$S" /content/jax-trace.tar.gz artifacts/jax-trace.tar.gz
else
  echo "[driver] NO TRACE — run_profile failed on the VM (see output above)"; exit 1
fi

echo "[driver] done -> artifacts/jax-trace.tar.gz"
