#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
API_PID=""
FRONTEND_PID=""

cleanup() {
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

echo "==> SolarOps demo run"
echo "==> Verifying demo snapshot"
"${PYTHON_BIN}" src/pipeline/prepare_demo_snapshot.py --config config.yaml
test -s outputs/demo/demo_snapshot.json
test -s outputs/demo/demo_forecast.csv
test -s outputs/demo/demo_site_ranking.csv
test -s outputs/demo/demo_operator_actions.json
test -s outputs/demo/demo_source_status.json

if [[ ! -d frontend/node_modules ]]; then
  echo "==> Installing frontend dependencies"
  npm --prefix frontend install
fi

echo "==> Building frontend"
npm --prefix frontend run build

if curl -fsS "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
  echo "==> FastAPI appears to be running already on port ${API_PORT}."
  echo "    To force demo mode, stop that server and rerun this script."
else
  echo "==> Starting FastAPI in demo mode on port ${API_PORT}"
  SOLAROPS_DATA_MODE=demo "${PYTHON_BIN}" -m uvicorn src.api.app:app --host 127.0.0.1 --port "${API_PORT}" &
  API_PID="$!"
fi

echo "==> Starting frontend on port ${FRONTEND_PORT}"
npm --prefix frontend run dev -- --port "${FRONTEND_PORT}" &
FRONTEND_PID="$!"

echo
echo "SolarOps demo is running."
echo "Frontend URL: http://127.0.0.1:${FRONTEND_PORT}"
echo "Demo API:     http://127.0.0.1:${API_PORT}/api/overview"
echo "The UI should display: Demo Snapshot"
echo "Press Ctrl+C to stop processes started by this script."
wait "${FRONTEND_PID}"
