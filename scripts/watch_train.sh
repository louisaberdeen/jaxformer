#!/usr/bin/env bash
# Live-tail a detached training run's log (/content/train.log on the VM) launched by
# train_colab.sh. Polls a tail every few seconds (no orphaned `tail -f` left on the VM).
# Ctrl-C stops watching — training keeps running on the VM. Reattach anytime by re-running.
#
# Usage:  SESSION=train ./scripts/watch_train.sh        # LINES=40 INTERVAL=5 to tune
set -euo pipefail

S="${SESSION:-train}"
N="${LINES:-40}"
INT="${INTERVAL:-5}"

echo "watching /content/train.log on '$S' (Ctrl-C to stop watching; training keeps running)"
while true; do
  printf '\033[2J\033[H'   # clear + home, so the tail redraws in place like `watch`
  echo "import subprocess as s; print(s.run(['tail','-n','$N','/content/train.log'],capture_output=True,text=True).stdout, end='')" \
    | colab exec -s "$S" --timeout 60 2>/dev/null || echo "(no log yet / VM unreachable)"
  sleep "$INT"
done
