#!/bin/bash
# AnySort Linux launcher
# Equivalent to AnySort.cmd on Windows
# Usage: ./AnySort.sh  (or double-click in file manager after chmod +x)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow X11 display forwarding into Docker
xhost +local:docker > /dev/null 2>&1

# Start container (no-op if already running)
docker compose -f "$SCRIPT_DIR/docker/docker-compose.yml" --env-file "$SCRIPT_DIR/docker/.env" up -d

# Launch AnySort pipeline
docker compose -f "$SCRIPT_DIR/docker/docker-compose.yml" --env-file "$SCRIPT_DIR/docker/.env" exec graspgen bash -c \
  "source /opt/GraspGen/.venv/bin/activate && cd /ros2_ws/app && python grasp_execute_pipeline.py"
