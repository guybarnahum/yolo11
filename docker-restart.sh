#!/bin/bash

# Enable / disable debugging
set -x  

# Stop and remove Docker containers managed by docker-compose
echo "Stopping containers..."
docker-compose down

# Prune unused images
echo "Pruning unused Docker images..."
docker image prune -f

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

docker-compose up $build

