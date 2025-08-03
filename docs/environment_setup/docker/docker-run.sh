#!/bin/bash
# Pylon Docker Run Script with Configurable Mounts
# Usage: ./docker-run.sh [options] -- [command]

set -e

# Default values
IMAGE_NAME="pylon:latest"
CONTAINER_NAME=""
GPU_DEVICES="all"
DATA_MOUNT=""
LOGS_MOUNT=""
ADDITIONAL_MOUNTS=""
PORTS=""
COMMAND=""
INTERACTIVE=false
DETACH=false

# Help function
show_help() {
    cat << EOF
Pylon Docker Run Script

Usage: $0 [OPTIONS] -- [COMMAND]

OPTIONS:
    -i, --image NAME          Docker image name (default: pylon:latest)
    -n, --name NAME           Container name (auto-generated if not provided)
    -g, --gpu DEVICES         GPU devices (default: all, use 'none' to disable)
    -d, --data PATH           Data directory mount (e.g., /pub5/data:/pub5/data:ro)
    -l, --logs PATH           Logs directory mount (e.g., ./logs:/workspace/logs)
    -m, --mount MOUNT         Additional mount (can be used multiple times)
    -p, --port PORT           Port mapping (can be used multiple times)
    --interactive             Run in interactive mode
    --detach                  Run in detached mode
    -h, --help                Show this help

EXAMPLES:
    # Basic run with custom data and logs paths
    $0 -d "/pub5/data:/pub5/data:ro" -l "./logs:/workspace/logs" -- python main.py --config configs/exp1.py

    # Interactive development
    $0 --interactive -d "/data:/workspace/data" -l "./logs:/workspace/logs"

    # Multi-server with shared storage
    $0 -d "/shared/datasets:/workspace/datasets:ro" -l "/shared/logs:/workspace/logs" -p "8050:8050" -- python main.py --config configs/exp1.py

    # Disable GPU access
    $0 -g none -d "/data:/workspace/data" -- python main.py --config configs/cpu_exp.py

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICES="$2"
            shift 2
            ;;
        -d|--data)
            DATA_MOUNT="$2"
            shift 2
            ;;
        -l|--logs)
            LOGS_MOUNT="$2"
            shift 2
            ;;
        -m|--mount)
            ADDITIONAL_MOUNTS="$ADDITIONAL_MOUNTS -v $2"
            shift 2
            ;;
        -p|--port)
            PORTS="$PORTS -p $2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            COMMAND="$*"
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if [ -z "$DATA_MOUNT" ] && [ -z "$LOGS_MOUNT" ]; then
    echo "Warning: No data or logs mounts specified. Container will use default paths."
fi

# Build docker run command
DOCKER_CMD="docker run"

# Container name
if [ -n "$CONTAINER_NAME" ]; then
    DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
else
    # Auto-generate container name
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DOCKER_CMD="$DOCKER_CMD --name pylon_${TIMESTAMP}"
fi

# GPU access
if [ "$GPU_DEVICES" != "none" ]; then
    DOCKER_CMD="$DOCKER_CMD --gpus $GPU_DEVICES"
fi

# Interactive/detach modes
if [ "$INTERACTIVE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -it"
elif [ "$DETACH" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
else
    DOCKER_CMD="$DOCKER_CMD --rm"
fi

# Volume mounts
DOCKER_CMD="$DOCKER_CMD -v $(pwd):/workspace/pylon"

if [ -n "$DATA_MOUNT" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $DATA_MOUNT"
fi

if [ -n "$LOGS_MOUNT" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $LOGS_MOUNT"
fi

if [ -n "$ADDITIONAL_MOUNTS" ]; then
    DOCKER_CMD="$DOCKER_CMD $ADDITIONAL_MOUNTS"
fi

# Port mappings
if [ -n "$PORTS" ]; then
    DOCKER_CMD="$DOCKER_CMD $PORTS"
fi

# Working directory
DOCKER_CMD="$DOCKER_CMD -w /workspace/pylon"

# Image
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"

# Command
if [ -n "$COMMAND" ]; then
    DOCKER_CMD="$DOCKER_CMD conda run --no-capture-output -n pylon $COMMAND"
elif [ "$INTERACTIVE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD conda run --no-capture-output -n pylon /bin/bash"
else
    echo "No command specified. Use --interactive or provide a command after --"
    exit 1
fi

# Execute
echo "Running: $DOCKER_CMD"
eval $DOCKER_CMD