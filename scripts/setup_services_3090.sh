#!/bin/bash

# PhotoSynth Service Setup (Run on 3090)
# This script installs systemd services to auto-start the pipeline and UI on boot.

USER="aditya"
PROJECT_DIR="/home/$USER/personal/PhotoSynth"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
CELERY="$VENV_DIR/bin/celery"
UVICORN="$VENV_DIR/bin/uvicorn"

# Ensure log directory exists
mkdir -p $PROJECT_DIR/logs

# --- 1. Celery Worker Service ---
cat << EOF | sudo tee /etc/systemd/system/photosynth-worker.service
[Unit]
Description=PhotoSynth Celery Worker (Node A)
After=network.target redis-server.service

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$CELERY -A photosynth.tasks worker --loglevel=info -Q detection_queue,vlm_queue -n worker_3090 --concurrency=1
Restart=always
StandardOutput=append:$PROJECT_DIR/logs/worker.log
StandardError=append:$PROJECT_DIR/logs/worker.log

[Install]
WantedBy=multi-user.target
EOF

# --- 2. UI Backend Service ---
cat << EOF | sudo tee /etc/systemd/system/photosynth-ui.service
[Unit]
Description=PhotoSynth Face Tagging UI
After=network.target

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$UVICORN photosynth.ui.backend:app --host 0.0.0.0 --port 8000
Restart=always
StandardOutput=append:$PROJECT_DIR/logs/ui.log
StandardError=append:$PROJECT_DIR/logs/ui.log

[Install]
WantedBy=multi-user.target
EOF

# --- 3. NAS Watcher Service ---
cat << EOF | sudo tee /etc/systemd/system/photosynth-watcher.service
[Unit]
Description=PhotoSynth NAS Watcher
After=network.target photosynth-worker.service

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON -m photosynth.nas_watcher
Restart=always
StandardOutput=append:$PROJECT_DIR/logs/watcher.log
StandardError=append:$PROJECT_DIR/logs/watcher.log

[Install]
WantedBy=multi-user.target
EOF

# --- Enable and Start ---
echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling services..."
sudo systemctl enable photosynth-worker
sudo systemctl enable photosynth-ui
sudo systemctl enable photosynth-watcher

echo "Starting services..."
sudo systemctl start photosynth-worker
sudo systemctl start photosynth-ui
sudo systemctl start photosynth-watcher

echo "✅ PhotoSynth Services Installed & Started!"
echo "   - Worker Status:  sudo systemctl status photosynth-worker"
echo "   - UI Status:      sudo systemctl status photosynth-ui"
echo "   - Watcher Status: sudo systemctl status photosynth-watcher"
