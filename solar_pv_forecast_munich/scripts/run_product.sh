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

echo "==> SolarOps product run"
echo "==> Validating environment"
command -v "${PYTHON_BIN}" >/dev/null
command -v node >/dev/null
command -v npm >/dev/null

"${PYTHON_BIN}" - <<'PY'
import importlib
required = ["pandas", "numpy", "requests", "yaml", "pvlib", "sklearn", "fastapi", "uvicorn", "lightgbm"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing or broken Python dependencies:\n" + "\n".join(missing))
print("Python dependencies OK, including LightGBM.")
PY

if [[ ! -d frontend/node_modules ]]; then
  echo "==> Installing frontend dependencies"
  npm --prefix frontend install
fi

echo "==> Updating live forecast pipeline"
"${PYTHON_BIN}" src/pipeline/run_live_forecast.py --config config.yaml

echo "==> Building frontend"
npm --prefix frontend run build

if curl -fsS "http://127.0.0.1:${API_PORT}/api/health" >/dev/null 2>&1; then
  echo "==> FastAPI already running on port ${API_PORT}"
else
  echo "==> Starting FastAPI on port ${API_PORT}"
  "${PYTHON_BIN}" -m uvicorn src.api.app:app --host 127.0.0.1 --port "${API_PORT}" &
  API_PID="$!"
fi

echo "==> Starting frontend on port ${FRONTEND_PORT}"
npm --prefix frontend run dev -- --port "${FRONTEND_PORT}" &
FRONTEND_PID="$!"

echo
echo "SolarOps product is running."
echo "Frontend URL: http://127.0.0.1:${FRONTEND_PORT}"
echo "API health:   http://127.0.0.1:${API_PORT}/api/health"
echo "Press Ctrl+C to stop processes started by this script."
wait "${FRONTEND_PID}"
