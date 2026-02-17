# CI/CD Setup Guide (Cloud Build)

## Architecture Overview

Unified **Google Cloud Build** pipeline triggered by GitHub push events:

```text
GitHub Push (main/develop)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│           Cloud Build (cloudbuild.yaml)                 │
│                                                         │
│  1️⃣  Lint with ruff (Python code quality)              │
│  2️⃣  Validate Dockerfiles & docker-compose syntax      │
│  3️⃣  Build 5 microservices in parallel                 │
│        - prediction-api, training-api, data-api        │
│        - collector-bot, admin-bot                      │
│  4️⃣  Push images to Artifact Registry                  │
│  5️⃣  Deploy to VM (if main branch)                     │
│        - SSH to VM, git pull, docker compose up        │
│  6️⃣  Health check validation                           │
│        - Verify all services responding                │
└─────────────────────────────────────────────────────────┘
    │
    ├─→ Artifact Registry (asia-southeast2-docker.pkg.dev)
    │
    └─→ GCP VM (mytechops) [on main branch only]
```

## Pipeline Flow

| Branch | Trigger | Action |
|--------|---------|--------|
| `main` | Push | Full CI + Deploy to VM |
| `develop` | Push | CI only (lint + build, no deploy) |
| Any | Manual | Trigger Cloud Build from Console |

## Setup Instructions

### 1. Enable Required GCP APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  secretmanager.googleapis.com \
  --project=mytech-480618
```

### 2. Create/Configure Service Account (one-time)

If not already done:

```bash
SA_EMAIL="cloud-build@mytech-480618.iam.gserviceaccount.com"

# If service account doesn't exist, create it
gcloud iam service-accounts create cloud-build \
  --display-name="Cloud Build Service Account" \
  --project=mytech-480618 2>/dev/null || true

# Grant required roles
gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/compute.networkAdmin"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

# Specific SSH permission
gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/compute.osLogin.basicInstanceAccessRole"
```

### 3. Store Secrets in Cloud Secret Manager

```bash
# Store VM user
echo -n "bagas" | gcloud secrets create gcp-vm-user \
  --data-file=- --project=mytech-480618 2>/dev/null || \
  gcloud secrets versions add gcp-vm-user --data-file=- --project=mytech-480618 <<< "bagas"

# Store VM name
echo -n "mytechops" | gcloud secrets create gcp-vm-name \
  --data-file=- --project=mytech-480618 2>/dev/null || \
  gcloud secrets versions add gcp-vm-name --data-file=- --project=mytech-480618 <<< "mytechops"

# Store VM zone
echo -n "asia-southeast2-a" | gcloud secrets create gcp-vm-zone \
  --data-file=- --project=mytech-480618 2>/dev/null || \
  gcloud secrets versions add gcp-vm-zone --data-file=- --project=mytech-480618 <<< "asia-southeast2-a"
```

### 4. Connect GitHub Repository to Cloud Build

#### Option A: Using gcloud CLI

```bash
gcloud builds connect \
  --repository=TelegramBotMyTech \
  --repository-name=TelegramBotMyTech \
  --github-owner=BagasAuliaAlfasyam \
  --region=asia-southeast2 \
  --project=mytech-480618
```

#### Option B: Using Google Cloud Console (Recommended for first-time setup)

1. Open [Google Cloud Console](https://console.cloud.google.com)
2. Go to **Cloud Build** → **Connected repositories**
3. Click **Connect repository**
4. Select **GitHub**
5. Authorize Cloud Build app to access your GitHub account
6. Select repository: `BagasAuliaAlfasyam/TelegramBotMyTech`
7. Click **Connect**

### 5. Create Cloud Build Triggers

#### Trigger 1: CI only (develop and PR)

In Cloud Console → **Cloud Build** → **Triggers** → **Create trigger**:

```yaml
Name: "CI - Lint & Build (develop)"
Repository: TelegramBotMyTech
Branch: ^develop$
Build config: Cloud Build (cloudbuild.yaml)
Substitutions:
  _DEPLOY_ENABLED: "false"
Description: "CI only: lint + build test (no deploy)"
```

#### Trigger 2: Full CI/CD (main branch)

Create another trigger:

```yaml
Name: "CI/CD - Full Pipeline (main)"
Repository: TelegramBotMyTech
Branch: ^main$
Build config: Cloud Build (cloudbuild.yaml)
Substitutions:
  _DEPLOY_ENABLED: "true"
Description: "Full CI + Deploy to VM"
```

### 6. Configure VM for Docker Pulls

SSH into VM and run:

```bash
# Install gcloud CLI CLI (if not present)
# https://cloud.google.com/sdk/docs/install

# Configure Docker auth
gcloud auth configure-docker asia-southeast2-docker.pkg.dev --quiet

# Ensure repo is cloned locally
cd ~
if [ ! -d "TelegramBotMyTech" ]; then
  git clone https://github.com/BagasAuliaAlfasyam/TelegramBotMyTech.git
fi
cd TelegramBotMyTech

# Verify .env.local and other secrets are present
ls -la .env.local
```

---

## Pipeline Details

### CI Stage (Always Runs)

- **Ruff linting** — Check Python code quality
- **Dockerfile validation** — Ensure all 5 Dockerfiles exist
- **docker-compose validation** — Syntax and config check
- **Docker builds** — Build all 5 service images in parallel
- **Push to Artifact Registry** — Tag with commit SHA and "latest"

### Deploy Stage (Main Branch Only)

- **Conditional execution** — Only if `_DEPLOY_ENABLED: "true"`
- **Git pull** — Fetch latest code from GitHub on VM
- **Docker pull** — Download pre-built images from Artifact Registry
- **docker compose up** — Deploy services to VM
- **Health checks** — Verify all services are responding on `/health` endpoints

---

## Manual Testing

### Test CI Only (No Deploy):

```bash
cd </path/to/TelegramBotMyTech>
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=test-manual,_DEPLOY_ENABLED=false \
  --project=mytech-480618
```

### Test Full Pipeline (With Deploy):

```bash
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=test-manual,_DEPLOY_ENABLED=true \
  --project=mytech-480618
```

### Monitor Build in Console:

```bash
# Watch build logs in real-time
gcloud builds log <BUILD_ID> --stream --project=mytech-480618
```

---

## Rollback

### Option 1: Deploy a previous version using image tag

SSH into VM:

```bash
cd ~/TelegramBotMyTech

# Set older image tag
export IMAGE_TAG=<previous-commit-sha>
export IMAGE_REGISTRY=asia-southeast2-docker.pkg.dev/mytech-480618/microservices/

# Redeploy
docker compose \
  -f docker-compose.microservices.yml \
  -f docker-compose.prod.yml \
  --env-file .env.local \
  pull && docker compose up -d --no-build
```

### Option 2: Manually re-run Cloud Build trigger

1. Go to Cloud Console → **Cloud Build** → **History**
2. Find the previous successful build
3. Click **Rebuild**

---

## File Structure

```text
.github/
  workflows/          # ← GitHub Actions (DELETED - using Cloud Build now)

cloudbuild.yaml             # ← Main CI/CD pipeline config
docker-compose.microservices.yml   # Base compose (dev with build:)
docker-compose.prod.yml            # Prod override (uses prebuilt images)
services/
  prediction/         # Service dockerfile
  training/           # Service dockerfile
  data/               # Service dockerfile
  collector/          # Service dockerfile
  admin/              # Service dockerfile
pyproject.toml        # Ruff linter config
```

---

## Troubleshooting

### Build fails during lint

```bash
# Check ruff config
cat pyproject.toml | grep -A 20 "\[tool.ruff\]"

# Fix locally before pushing
ruff check services/
```

### Build fails during docker build

```bash
# Test locally
docker build -f services/prediction/Dockerfile -t test .
```

### Deploy fails (VM SSH error)

```bash
# Verify VM exists and is reachable
gcloud compute instances describe mytechops --zone=asia-southeast2-a --project=mytech-480618

# Test SSH manually
gcloud compute ssh bagas@mytechops --zone=asia-southeast2-a --project=mytech-480618
```

### Services unhealthy after deploy

```bash
# SSH to VM and check
gcloud compute ssh bagas@mytechops --zone=asia-southeast2-a --project=mytech-480618

# Check logs
docker compose logs prediction-api | head -20
docker compose logs data-api | head -20

# Restart a service
docker compose restart prediction-api
```

---

## Migration from GitHub Actions

✅ **Completed migration:** GitHub Actions → pure Cloud Build
- Removed `.github/workflows/ci.yml`
- Removed `.github/workflows/deploy.yml`
- All logic consolidated into single `cloudbuild.yaml`
- GitHub → Cloud Build integration via connected repository
- No more GitHub Actions quota/billing concerns
