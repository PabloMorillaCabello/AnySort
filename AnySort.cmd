@echo off
docker compose -f "%~dp0docker\docker-compose.yml" up -d >nul 2>&1
docker compose -f "%~dp0docker\docker-compose.yml" exec graspgen bash -c "source /opt/GraspGen/.venv/bin/activate && cd /ros2_ws/app && python grasp_execute_pipeline.py"
