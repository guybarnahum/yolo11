#!/bin/bash
# Usage ./docker-restart.sh [docker compose arg]

# Enable / disable debugging
set -x  

# Stop and remove Docker containers managed by docker-compose
echo "Stopping containers..."
# Docker version: 4.32.0 need to remove the dash from docker-compose to docker compose
docker compose down 2>/dev/null || docker-compose down 2>/dev/null

# Prune unused images
echo "Pruning unused Docker images..."
docker image prune -f
docker volume prune -f
docker network prune -f

# Start Docker containers
echo "Starting containers..."
# Docker version: 4.32.0 need to remove the dash from docker-compose to docker compose
docker compose up --build $1 2>/dev/null || docker-compose up --build $1 2>/dev/null
