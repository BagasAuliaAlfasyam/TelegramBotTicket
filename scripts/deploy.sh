#!/usr/bin/env bash
# ============================================================
# deploy.sh — Manual deploy script for the GCP VM
# ============================================================
# Run this ON the VM to deploy or update services.
#
# Usage:
#   ./scripts/deploy.sh                 # Deploy latest from Artifact Registry
#   ./scripts/deploy.sh --local         # Build locally (dev mode)
#   ./scripts/deploy.sh --tag abc1234   # Deploy specific image tag
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# ── Configuration ─────────────────────────────────────────
GCP_PROJECT="${GCP_PROJECT_ID:-mytech-480618}"
REGION="asia-southeast2"
REGISTRY="${REGION}-docker.pkg.dev/${GCP_PROJECT}/microservices"
TAG="${IMAGE_TAG:-latest}"
MODE="registry"     # registry | local

# ── Parse arguments ───────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --local)   MODE="local"; shift ;;
    --tag)     TAG="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--local] [--tag TAG]"
      echo "  --local    Build images locally instead of pulling from registry"
      echo "  --tag TAG  Image tag to deploy (default: latest)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  TelegramBotMyTech — Deploy"
echo "============================================"
echo "  Mode     : ${MODE}"
echo "  Tag      : ${TAG}"
echo "  Project  : ${PROJECT_DIR}"
echo "============================================"

# ── Pull latest code ──────────────────────────────────────
if git rev-parse --is-inside-work-tree &>/dev/null; then
  echo ""
  echo ">>> Pulling latest code..."
  git pull origin main --ff-only || echo "Warning: git pull failed, continuing with current code"
fi

# ── Deploy ────────────────────────────────────────────────
if [[ "$MODE" == "local" ]]; then
  echo ""
  echo ">>> Building & deploying locally..."
  docker compose \
    -f docker-compose.yml \
    up -d --build

else
  echo ""
  echo ">>> Configuring Docker auth for Artifact Registry..."
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet 2>/dev/null || true

  export IMAGE_REGISTRY="${REGISTRY}/"
  export IMAGE_TAG="${TAG}"

  echo ">>> Pulling images (tag: ${TAG})..."
  docker compose \
    -f docker-compose.yml \
    -f docker-compose.override.yml \
    pull prediction-api training-api data-api collector-bot admin-bot

  echo ""
  echo ">>> Building local services (mlflow)..."
  docker compose \
    -f docker-compose.yml \
    -f docker-compose.override.yml \
    build mlflow

  echo ""
  echo ">>> Deploying..."
  docker compose \
    -f docker-compose.yml \
    -f docker-compose.override.yml \
    up -d --no-build
fi

# ── Health checks ─────────────────────────────────────────
echo ""
echo ">>> Waiting for services to start (20s)..."
sleep 20

echo ""
echo "=== Container Status ==="
docker ps --format 'table {{.Names}}\t{{.Status}}'

echo ""
echo "=== Health Checks ==="
for svc in prediction-api:8001 data-api:8002 training-api:8005; do
  name="${svc%%:*}"
  port="${svc##*:}"
  status=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:${port}/health" 2>/dev/null || echo "000")
  if [[ "$status" == "200" ]]; then
    echo "  ✓ ${name} — healthy (HTTP ${status})"
  else
    echo "  ✗ ${name} — UNHEALTHY (HTTP ${status})"
  fi
done

echo ""
echo ">>> Deploy complete!"
