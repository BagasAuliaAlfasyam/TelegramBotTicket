# CI/CD Setup Guide

## Architecture

```
GitHub (push to main)
    │
    ▼
GitHub Actions
    │
    ├─── CI Job: Lint + Docker build test
    │
    ├─── Cloud Build: Build 5 service images in parallel
    │         │
    │         ▼
    │    Artifact Registry (asia-southeast2-docker.pkg.dev)
    │         │
    │         ▼
    └─── Deploy Job: SSH to VM → pull images → docker compose up
              │
              ▼
         GCP VM (mytechops)
```

## Pipeline Flow

| Event | Action |
|-------|--------|
| Push to `main` | CI → Cloud Build → Deploy to VM |
| Push to `develop` / PR | CI only (lint + build test) |
| Manual trigger | Deploy with optional skip-build |

## Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions** in your repo:

| Secret | Description | Example |
|--------|-------------|---------|
| `GCP_SA_KEY` | Service account JSON key (full JSON, not base64) | `{"type":"service_account",...}` |
| `GCP_PROJECT_ID` | GCP project ID | `mytech-480618` |
| `GCP_VM_NAME` | VM instance name | `mytechops` |
| `GCP_VM_ZONE` | VM zone | `asia-southeast2-a` |
| `GCP_VM_USER` | SSH user on VM | `bagas` |

## GCP Setup (One-time)

### 1. Enable required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  --project=mytech-480618
```

### 2. Create service account for CI/CD

```bash
# Create service account
gcloud iam service-accounts create github-deploy \
  --display-name="GitHub Actions Deploy" \
  --project=mytech-480618

SA_EMAIL="github-deploy@mytech-480618.iam.gserviceaccount.com"

# Grant required roles
gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/compute.instanceAdmin.v1"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iap.tunnelResourceAccessor"

gcloud projects add-iam-policy-binding mytech-480618 \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

# Create key
gcloud iam service-accounts keys create github-deploy-key.json \
  --iam-account="${SA_EMAIL}"

echo "Copy the contents of github-deploy-key.json into GitHub secret GCP_SA_KEY"
```

### 3. Configure VM for Artifact Registry

SSH into the VM and run:

```bash
# Install gcloud CLI (if not already present)
# https://cloud.google.com/sdk/docs/install

# Configure Docker to pull from Artifact Registry
gcloud auth configure-docker asia-southeast2-docker.pkg.dev --quiet
```

### 4. Ensure the VM has a git clone of the repo

```bash
cd ~
git clone https://github.com/BagasAuliaAlfasyam/TelegramBotMyTech.git
cd TelegramBotMyTech

# Copy .env.local (secrets stay on VM, never in git)
# Make sure .env.local and white-set-*.json are present
```

## Manual Deploy (without CI/CD)

From the VM:

```bash
cd ~/TelegramBotMyTech

# Option 1: Deploy from Artifact Registry (production)
./scripts/deploy.sh

# Option 2: Deploy with specific tag
./scripts/deploy.sh --tag abc1234

# Option 3: Build locally (development)
./scripts/deploy.sh --local
```

## Rollback

```bash
# Deploy a previous version by tag (use git commit SHA)
./scripts/deploy.sh --tag <previous-commit-sha>

# Or via GitHub: re-run a previous Deploy workflow
```

## File Structure

```
.github/
  workflows/
    ci.yml              # Lint + docker build test
    deploy.yml          # Cloud Build + deploy to VM
cloudbuild.yaml         # Cloud Build config (parallel image builds)
docker-compose.microservices.yml   # Base compose (dev, with build:)
docker-compose.prod.yml            # Prod override (uses image: from registry)
scripts/
  deploy.sh             # Manual deploy script for VM
pyproject.toml          # Ruff linter config
```

## Secrets Management

- **`.env.local`** — lives on the VM only, never committed
- **`white-set-*.json`** — Google service account, lives on VM, in `.gitignore`
- **GitHub Secrets** — GCP auth for CI/CD pipeline
- **No secrets in Docker images** — all injected via env vars at runtime
