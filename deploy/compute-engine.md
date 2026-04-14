# Compute Engine Deployment Guide

This guide deploys the current CrickAI backend to a single Ubuntu VM on Google
Compute Engine. It is intentionally optimized for your current app design:

- FastAPI runs as a long-lived service.
- Redis runs on the same VM to keep cost down.
- `systemd` keeps the API alive across restarts.
- `nginx` handles uploads and reverse proxies to Uvicorn.

## 1. Recommended VM

- OS: Ubuntu 24.04 LTS
- Machine type: `e2-standard-2`
- Disk: `30 GB` standard persistent disk
- Firewall: allow `HTTP`

If you only want a light demo, `e2-medium` can work, but video and ML steps
will be slower.

## 2. Copy the repo to the VM

```bash
git clone <your-repo-url> /opt/crickai
cd /opt/crickai
```

## 3. Install system packages

```bash
sudo apt update
sudo apt install -y git curl ffmpeg nginx redis-server build-essential
```

Enable Redis now:

```bash
sudo systemctl enable redis-server
sudo systemctl start redis-server
sudo systemctl status redis-server
```

## 4. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If you prefer a system-wide install, create a symlink after install:

```bash
sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
```

## 5. Create the production env file

Start from the example in [deploy/.env.gce.example](/home/tharu/projects/Final_CrickAI_Backend/deploy/.env.gce.example).

```bash
cp deploy/.env.gce.example .env
nano .env
```

Minimum values to change:

- `JWT_SECRET`
- `REVENUECAT_WEBHOOK_SECRET`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET`

Notes:

- This app reads `.env` directly via `pydantic-settings`.
- Local Redis on the VM should stay `redis://127.0.0.1:6379/0`.
- SQLite is the cheapest first deployment choice for this VM.

## 6. Install Python dependencies

This repo currently relies on optional ML packages for the implemented video
pipeline, so install the same extras used by the container image:

```bash
uv sync \
  --extra bat-contact \
  --extra ml-base-gpu \
  --extra bowler-performance \
  --extra action-legality \
  --extra shot-classifier \
  --extra shot-similarity
```

If the VM does not have a GPU, switch `ml-base-gpu` to `ml-base-cpu`:

```bash
uv sync \
  --extra bat-contact \
  --extra ml-base-cpu \
  --extra bowler-performance \
  --extra action-legality \
  --extra shot-classifier \
  --extra shot-similarity
```

## 7. Smoke test the app

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

From another shell on the VM:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Stop the manual server after the health check.

## 8. Install the `systemd` service

Copy the service file from
[deploy/crickai-api.service](/home/tharu/projects/Final_CrickAI_Backend/deploy/crickai-api.service):

```bash
sudo cp deploy/crickai-api.service /etc/systemd/system/crickai-api.service
sudo systemctl daemon-reload
sudo systemctl enable crickai-api
sudo systemctl start crickai-api
sudo systemctl status crickai-api
```

Useful logs:

```bash
sudo journalctl -u crickai-api -f
```

## 9. Configure nginx

Copy the provided site config from
[deploy/nginx-crickai.conf](/home/tharu/projects/Final_CrickAI_Backend/deploy/nginx-crickai.conf):

```bash
sudo cp deploy/nginx-crickai.conf /etc/nginx/sites-available/crickai
sudo ln -s /etc/nginx/sites-available/crickai /etc/nginx/sites-enabled/crickai
sudo nginx -t
sudo systemctl restart nginx
```

If the default site is still enabled, remove it:

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl reload nginx
```

## 10. Verify the public endpoint

Replace `<VM_EXTERNAL_IP>` with the VM IP:

```bash
curl http://<VM_EXTERNAL_IP>/health
```

## 11. Firewall checks

Make sure the VM or VPC firewall allows:

- TCP `80`
- TCP `22`

Do not expose:

- Redis `6379`
- Uvicorn `8000`

## 12. Operational notes

- Uploaded videos and temp files can fill disk over time, so monitor free
  space.
- The current app uses FastAPI `BackgroundTasks`, so very heavy parallel job
  volume can still overwhelm a single VM.
- The first scalability upgrade should be moving video processing into a
  dedicated worker service.

## 13. Update workflow

When you push new code:

```bash
cd /opt/crickai
git pull
source "$HOME/.local/bin/env"
uv sync \
  --extra bat-contact \
  --extra ml-base-cpu \
  --extra bowler-performance \
  --extra action-legality \
  --extra shot-classifier \
  --extra shot-similarity
sudo systemctl restart crickai-api
sudo journalctl -u crickai-api -n 100 --no-pager
```

## 14. Security follow-up

If any real secrets were committed to `.env` or shared during setup, rotate
them before going live:

- AWS keys
- JWT secret
- RevenueCat webhook secret