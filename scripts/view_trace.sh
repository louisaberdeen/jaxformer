#!/usr/bin/env bash
# Extract the profiler trace pulled back by profile_colab.sh (artifacts/jax-trace.tar.gz)
# and open it in TensorBoard. The trace is already local — no Drive/rclone.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAR="$ROOT/artifacts/jax-trace.tar.gz"
LOGDIR="$HOME/jax-trace"
PORT=6006
URL="http://localhost:$PORT"

[ -f "$TAR" ] || { echo "no $TAR — run ./scripts/profile_colab.sh first"; exit 1; }

# fresh extract
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"
tar -xzf "$TAR" -C "$LOGDIR"

# (re)start TensorBoard: kill anything on the port, then launch fresh
if curl -sf -m 2 "$URL" >/dev/null 2>&1; then
  echo "→ restarting TensorBoard with refreshed trace..."
  fuser -k "$PORT/tcp" 2>/dev/null || true
  sleep 1
fi

echo "→ starting TensorBoard..."
nohup uv run --project "$ROOT" tensorboard --logdir "$LOGDIR" --port "$PORT" \
  > "$HOME/tensorboard.log" 2>&1 &

# wait for it to come up
for _ in $(seq 1 20); do
  curl -sf -m 2 "$URL" >/dev/null 2>&1 && break
  sleep 0.5
done

echo "→ TensorBoard ready:  $URL   (open the Profile tab → Trace Viewer / Overview)"
command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
