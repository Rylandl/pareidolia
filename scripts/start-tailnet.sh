#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

tailnet_ip="${RECTIFIER_HOST:-}"
if [[ -z "$tailnet_ip" ]]; then
  if ! command -v tailscale >/dev/null 2>&1; then
    echo "tailscale is not available; set RECTIFIER_HOST explicitly" >&2
    exit 1
  fi
  tailnet_ip="$(tailscale ip -4 | sed -n '1p')"
fi
if [[ -z "$tailnet_ip" ]]; then
  echo "could not determine a Tailscale IPv4 address" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  for candidate in "$project_dir/.tools/node/bin" /tmp/node-v24.18.0-linux-x64/bin; do
    if [[ -x "$candidate/npm" ]]; then
      export PATH="$candidate:$PATH"
      break
    fi
  done
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js 22.13 or newer is required" >&2
  exit 1
fi

api_port="${RECTIFIER_API_PORT:-8000}"
ui_port="${RECTIFIER_PORT:-3000}"
python_bin="${RECTIFIER_PYTHON:-python3}"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  python_bin="$project_dir/.venv/bin/python"
fi
if [[ -z "${CUDA_PATH:-}" && -f /usr/include/vector_types.h ]]; then
  export CUDA_PATH=/usr
fi
default_volume="$project_dir/data/pherc0358-z7168-y5888-x4608.npy"
volume_path="${RECTIFIER_VOLUME:-$default_volume}"
volume_args=()
if [[ -f "$volume_path" ]]; then
  volume_args=(--volume "$volume_path")
elif [[ -n "${RECTIFIER_VOLUME:-}" ]]; then
  echo "RECTIFIER_VOLUME does not exist: $volume_path" >&2
  exit 1
fi
npm run build

"$python_bin" backend/server.py --host "127.0.0.1" --port "$api_port" "${volume_args[@]}" &
backend_pid=$!
"$python_bin" backend/proxy.py \
  --host "$tailnet_ip" \
  --port "$ui_port" \
  --upstream "http://127.0.0.1:$ui_port" \
  --api-upstream "http://127.0.0.1:$api_port" &
proxy_pid=$!
trap 'kill "$backend_pid" "$proxy_pid" 2>/dev/null || true' EXIT INT TERM

echo "Rectifier Lab will be available at http://$tailnet_ip:$ui_port/"
npm run start -- --hostname 127.0.0.1 --port "$ui_port"
