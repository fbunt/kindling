#!/usr/bin/env bash
# Run the kindling app under Podman (Option A): the app container spawns sibling
# worker containers via the host's rootless Podman socket.
#
# Usage:
#   scripts/run.sh [PARQUET]            # run (default: data/mtbs_pix_data.parquet)
#   scripts/run.sh --build [PARQUET]    # build both images first, then run
#
# Env overrides (all optional):
#   KINDLING_SANDBOX_MEM   per-worker memory cap (image default: 110g)
#   KINDLING_POOL_SIZE     warm workers to keep ready (default: 2)
#   GEMINI_API_KEY         Gemini key (otherwise log in via the web UI)
#   KINDLING_PORT          host port (default: 8000)
#   KINDLING_NAME          container name (default: kindling)
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (for build context + the default parquet)

BUILD=0
if [[ "${1:-}" == "--build" ]]; then BUILD=1; shift; fi

PARQUET_INPUT="${1:-data/mtbs_pix_data.parquet}"
NAME="${KINDLING_NAME:-kindling}"
PORT="${KINDLING_PORT:-8000}"
APP_IMAGE="${KINDLING_APP_IMAGE:-kindling-app:latest}"
WORKER_IMAGE="${KINDLING_SANDBOX_IMAGE:-kindling-worker:latest}"
SOCK="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/podman/podman.sock"

if [[ ! -e "$PARQUET_INPUT" ]]; then
  echo "error: parquet not found: $PARQUET_INPUT (relative to repo root)" >&2
  exit 1
fi
# Absolute, symlink-resolved host path — workers bind-mount this exact path.
PARQUET="$(readlink -f "$PARQUET_INPUT")"

# Ensure the rootless Podman socket exists (the app drives the host via it).
if [[ ! -S "$SOCK" ]]; then
  echo ">> enabling rootless podman socket"
  systemctl --user enable --now podman.socket
fi

if [[ "$BUILD" == 1 ]]; then
  echo ">> building worker + app images"
  podman build -t "$WORKER_IMAGE" -f Containerfile .
  podman build -t "$APP_IMAGE" -f Containerfile.app .
fi

# Both images must exist in the host store (workers run on the host runtime).
for img in "$WORKER_IMAGE" "$APP_IMAGE"; do
  if ! podman image exists "$img"; then
    echo "error: image '$img' not found — run with --build, or build it manually." >&2
    exit 1
  fi
done

podman rm -f "$NAME" >/dev/null 2>&1 || true

# If GEMINI_API_KEY isn't already in the environment, read it from .env (repo
# root). Targeted extraction (not `source`) so arbitrary .env content can't run.
if [[ -z "${GEMINI_API_KEY:-}" && -f .env ]]; then
  line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?GEMINI_API_KEY[[:space:]]*=' .env | tail -1 || true)"
  if [[ -n "$line" ]]; then
    val="${line#*=}"                                  # value after the first =
    val="${val%$'\r'}"                                # strip trailing CR
    val="${val#"${val%%[![:space:]]*}"}"             # ltrim
    val="${val%"${val##*[![:space:]]}"}"             # rtrim
    val="${val%\"}"; val="${val#\"}"                  # strip surrounding "
    val="${val%\'}"; val="${val#\'}"                  # strip surrounding '
    export GEMINI_API_KEY="$val"
    echo ">> loaded GEMINI_API_KEY from .env"
  fi
fi
[[ -z "${GEMINI_API_KEY:-}" ]] && echo ">> no GEMINI_API_KEY (set it or log in via the UI)"

# Forward optional tuning env only when set.
extra=()
[[ -n "${KINDLING_SANDBOX_MEM:-}" ]] && extra+=(-e "KINDLING_SANDBOX_MEM=${KINDLING_SANDBOX_MEM}")
[[ -n "${KINDLING_POOL_SIZE:-}" ]]   && extra+=(-e "KINDLING_POOL_SIZE=${KINDLING_POOL_SIZE}")
[[ -n "${GEMINI_API_KEY:-}" ]]       && extra+=(-e "GEMINI_API_KEY=${GEMINI_API_KEY}")

echo ">> starting $NAME  (parquet=$PARQUET)"
podman run -d --name "$NAME" -p "${PORT}:8000" \
  -v "${SOCK}:/run/podman/podman.sock" \
  -e CONTAINER_HOST=unix:///run/podman/podman.sock \
  -v "${PARQUET}:/data/dataset.parquet:ro" \
  -e KINDLING_WORKER_PARQUET_PATH="${PARQUET}" \
  -e KINDLING_SANDBOX_IMAGE="${WORKER_IMAGE}" \
  -e PYTHONUNBUFFERED=1 \
  "${extra[@]}" \
  --security-opt label=disable \
  "$APP_IMAGE" kindling /data/dataset.parquet --host 0.0.0.0

echo ">> kindling running at http://localhost:${PORT}"
echo ">> logs:  podman logs -f ${NAME}"
echo ">> stop:  podman rm -f ${NAME}"
