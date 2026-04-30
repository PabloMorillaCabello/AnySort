@echo off
docker compose -f "%~dp0docker\docker-compose.yml" --env-file "%~dp0docker\.env" up -d >nul 2>&1
docker compose -f "%~dp0docker\docker-compose.yml" --env-file "%~dp0docker\.env" exec graspgen bash -c "source /opt/GraspGen/.venv/bin/activate && cd /ros2_ws/app && python anysort.py"
