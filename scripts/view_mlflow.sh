#!/usr/bin/env bash
# Open the MLflow UI on the metrics db in artifacts/mlflow.db — no Drive/rclone.
# FOLLOW=1 keeps re-pulling the db from a running VM so the UI tracks a live run:
#   FOLLOW=1 SESSION=train ./scripts/view_mlflow.sh   # INTERVAL=30 to tune
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/artifacts/mlflow.db"
PORT=5000
URL="http://localhost:$PORT"
S="${SESSION:-train}"

if [ "${FOLLOW:-}" = "1" ]; then
  echo "→ FOLLOW: pulling mlflow.db from '$S' every ${INTERVAL:-30}s"
  colab download -s "$S" /content/mlflow.db "$DB" 2>/dev/null || true   # initial snapshot
fi

[ -f "$DB" ] || { echo "no $DB — run ./scripts/train_colab.sh first (or FOLLOW=1 SESSION=... to pull live)"; exit 1; }

# (re)start the UI: kill anything on the port, then launch fresh against the db
if curl -sf -m 2 "$URL/health" >/dev/null 2>&1; then
  echo "→ restarting UI..."
  fuser -k "$PORT/tcp" 2>/dev/null || true
  sleep 1
fi

echo "→ starting MLflow UI..."
nohup uv run --project "$ROOT" mlflow ui --backend-store-uri "sqlite:///$DB" --port "$PORT" \
  > "$HOME/mlflow_ui.log" 2>&1 &

# wait for it to come up
for _ in $(seq 1 20); do
  curl -sf -m 2 "$URL/health" >/dev/null 2>&1 && break
  sleep 0.5
done

echo "→ MLflow UI ready:  $URL"
command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true

if [ "${FOLLOW:-}" = "1" ]; then
  echo "→ following '$S' (Ctrl-C to stop; UI keeps serving the last snapshot)"
  # foreground loop so Ctrl-C ends it; the UI reads the sqlite file fresh each request, so
  # overwriting it in place surfaces new metrics (they flush every 25 epochs).
  while true; do
    sleep "${INTERVAL:-30}"
    colab download -s "$S" /content/mlflow.db "$DB" 2>/dev/null || true
  done
fi
