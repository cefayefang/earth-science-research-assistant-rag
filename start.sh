#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate

# kill any existing instances
pkill -f "uvicorn app.main" 2>/dev/null || true
pkill -f "ui.py" 2>/dev/null || true
sleep 1

echo "Starting API server on http://localhost:8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/ra_api.log 2>&1 &
API_PID=$!

echo "Waiting for API to be ready..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API ready."
        break
    fi
    sleep 1
done

echo "Starting UI on http://localhost:7860 ..."
python3 ui.py &
UI_PID=$!

echo ""
echo "============================================"
echo "  Earth Science Research Assistant"
echo "  UI:  http://localhost:7860"
echo "  API: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $API_PID $UI_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait $UI_PID
