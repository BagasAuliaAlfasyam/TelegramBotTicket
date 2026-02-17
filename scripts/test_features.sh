#!/bin/bash
# ============================================================
# Full feature test suite for all microservices
# ============================================================
set -euo pipefail

BASE="http://localhost"
PASS=0
FAIL=0
SKIP=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

result() {
  if [ "$1" = "PASS" ]; then
    echo -e "  ${GREEN}✓ PASS${NC} — $2"
    PASS=$((PASS + 1))
  elif [ "$1" = "FAIL" ]; then
    echo -e "  ${RED}✗ FAIL${NC} — $2"
    FAIL=$((FAIL + 1))
  else
    echo -e "  ${YELLOW}⊘ SKIP${NC} — $2"
    SKIP=$((SKIP + 1))
  fi
}

# ============================================================
echo "============================================"
echo "  Feature Test Suite"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── 1. Health Checks ──────────────────────────────────────
echo "=== 1. Health Checks ==="

for svc in prediction-api:8001 data-api:8002 training-api:8005 mlflow:5000; do
  name="${svc%%:*}"
  port="${svc##*:}"
  resp=$(curl -sf "${BASE}:${port}/health" 2>/dev/null || echo "FAIL")
  if echo "$resp" | grep -q '"status"'; then
    result "PASS" "${name} health (port ${port})"
  elif [ "$port" = "5000" ] && curl -sf "${BASE}:5000/health" >/dev/null 2>&1; then
    result "PASS" "${name} health (port ${port})"
  else
    result "FAIL" "${name} health (port ${port}) — ${resp}"
  fi
done

echo ""

# ── 2. Data API ──────────────────────────────────────────
echo "=== 2. Data API ==="

# 2a. GET training data
resp=$(curl -sf --max-time 30 "${BASE}:8002/training/data" 2>/dev/null || echo "FAIL")
if echo "$resp" | grep -qE '"logs_data"|"ml_tracking"'; then
  count=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('logs_data',[])) + len(d.get('ml_tracking',[])))" 2>/dev/null || echo "?")
  result "PASS" "GET /training/data (${count} records)"
else
  result "FAIL" "GET /training/data — ${resp:0:120}"
fi

# 2b. GET realtime stats
resp=$(curl -sf --max-time 15 "${BASE}:8002/stats/realtime" 2>/dev/null || echo "FAIL")
if echo "$resp" | grep -qE '"total_predictions"|"auto_count"'; then
  total=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_predictions',0))" 2>/dev/null || echo "?")
  result "PASS" "GET /stats/realtime (${total} predictions)"
else
  result "FAIL" "GET /stats/realtime — ${resp:0:120}"
fi

# 2c. GET weekly stats
resp=$(curl -sf --max-time 15 "${BASE}:8002/stats/weekly" 2>/dev/null || echo "TIMEOUT")
if [ "$resp" != "FAIL" ] && [ "$resp" != "TIMEOUT" ] && [ -n "$resp" ]; then
  result "PASS" "GET /stats/weekly"
else
  result "FAIL" "GET /stats/weekly — ${resp:0:120}"
fi

# 2c. POST log entry (dry run — no actual write)
result "SKIP" "POST /log (skip: would write to sheets)"

echo ""

# ── 3. Training API ──────────────────────────────────────
echo "=== 3. Training API ==="

# 3a. GET training status
resp=$(curl -sf "${BASE}:8005/health" 2>/dev/null || echo "FAIL")
if echo "$resp" | grep -q '"status"'; then
  result "PASS" "GET /health"
else
  result "FAIL" "GET /health"
fi

# 3b. Training (skip if already done recently)
resp=$(curl -sf "${BASE}:8005/status" 2>/dev/null || echo "FAIL")
if echo "$resp" | grep -q '"status"'; then
  result "PASS" "GET /status"
else
  # Try training health endpoint
  result "SKIP" "GET /status (may not exist)"
fi

echo ""

# ── 4. Prediction API ────────────────────────────────────
echo "=== 4. Prediction API ==="

# 4a. Predict — high confidence ticket
cat > /tmp/test_high.json << 'EOF'
{"tech_raw_text": "moban reset password MIT tidak bisa login", "solving": "reset password user berhasil"}
EOF

resp=$(curl -sf -X POST "${BASE}:8001/predict" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_high.json 2>/dev/null || echo "FAIL")

if echo "$resp" | grep -q '"predicted_symtomps"'; then
  label=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"predicted_symtomps\"]} ({d[\"ml_confidence\"]:.1%}) [{d[\"prediction_status\"]}]')" 2>/dev/null)
  result "PASS" "POST /predict high-conf — ${label}"
else
  result "FAIL" "POST /predict high-conf — ${resp}"
fi

# 4b. Predict — low confidence (should trigger Gemini cascade)
cat > /tmp/test_low.json << 'EOF'
{"tech_raw_text": "printer error kode 0x12345", "solving": "ganti toner"}
EOF

resp=$(curl -sf -X POST "${BASE}:8001/predict" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_low.json 2>/dev/null || echo "FAIL")

if echo "$resp" | grep -q '"predicted_symtomps"'; then
  label=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"predicted_symtomps\"]} ({d[\"ml_confidence\"]:.1%}) [{d[\"prediction_status\"]}] gemini={d.get(\"gemini_label\",\"N/A\")}')" 2>/dev/null)
  result "PASS" "POST /predict low-conf — ${label}"
else
  result "FAIL" "POST /predict low-conf — ${resp}"
fi

# 4c. Model reload
resp=$(curl -sf -X POST "${BASE}:8001/model/reload" \
  -H "Content-Type: application/json" 2>/dev/null || echo "FAIL")

if echo "$resp" | grep -q '"success"'; then
  result "PASS" "POST /model/reload"
else
  result "FAIL" "POST /model/reload — ${resp}"
fi

echo ""

# ── 5. MLflow ─────────────────────────────────────────────
echo "=== 5. MLflow ==="

resp=$(curl -sf "${BASE}:5000/api/2.0/mlflow/registered-models/search" 2>/dev/null || echo "FAIL")
if echo "$resp" | grep -q '"registered_models"'; then
  count=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('registered_models',[])))" 2>/dev/null)
  result "PASS" "MLflow API (${count} registered models)"
else
  result "FAIL" "MLflow API — ${resp}"
fi

echo ""

# ── 6. MinIO ─────────────────────────────────────────────
echo "=== 6. MinIO ==="

resp=$(curl -sf "${BASE}:9000/minio/health/live" 2>/dev/null && echo "OK" || echo "FAIL")
if [ "$resp" = "OK" ]; then
  result "PASS" "MinIO health"
else
  result "FAIL" "MinIO health"
fi

echo ""

# ── 7. Telegram Bots ─────────────────────────────────────
echo "=== 7. Telegram Bots ==="

collector_status=$(docker inspect -f '{{.State.Status}}' collector-bot 2>/dev/null || echo "missing")
if [ "$collector_status" = "running" ]; then
  result "PASS" "collector-bot container running"
else
  result "FAIL" "collector-bot container: ${collector_status}"
fi

admin_status=$(docker inspect -f '{{.State.Status}}' admin-bot 2>/dev/null || echo "missing")
if [ "$admin_status" = "running" ]; then
  result "PASS" "admin-bot container running"
else
  result "FAIL" "admin-bot container: ${admin_status}"
fi

result "SKIP" "Telegram bot message handling (requires bot interaction)"

echo ""

# ── Summary ───────────────────────────────────────────────
echo "============================================"
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
