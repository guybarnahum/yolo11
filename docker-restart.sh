#!/bin/bash
# Usage ./docker-restart.sh [ fastapi | fiftyone ]

# Enable / disable debugging
set -x  

# Stop and remove Docker containers managed by docker-compose
echo "Stopping containers..."
# Docker version: 4.32.0 need to remove the dash from docker-compose to docker compose
docker-compose down --volumes || docker compose down --volumes

# Prune unused images
echo "Pruning unused Docker images..."
docker image prune -f
docker volume prune -f
docker network prune -f

# Start Docker containers
echo "Starting containers..."
# Get the last modified timestamp of the current script in Unix time
last_run_time=$(stat -f "%m" "$0")
#echo "Last modified timestamp: $modified_timestamp"

last_modified_requirements=$(stat -f "%m" "requirements.txt")
#echo "Last modified requirements: $last_modified_requirements"

# Compare and assign a variable
build=""
if [ "$last_modified_requirements" -gt "$last_run_time" ]; then
    build="--build"
    echo "requirements.txt was modified >> rebuild docker image"
fi

# Docker version: 4.32.0 need to remove the dash from docker-compose to docker compose
docker-compose up $build $1 || docker compose up $build $1
