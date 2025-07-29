#!/bin/bash
# Multi-Server Pylon Deployment Script
set -e

# Configuration
IMAGE_NAME="pylon:latest"
IMAGE_FILE="pylon.tar.gz"
SERVERS=("server1" "server2" "server3")  # Add your server names
SHARED_LOGS_PATH="/shared/storage/pylon_logs"
CONFIG_DIR="configs/experiments"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Function to build and save Docker image
build_image() {
    log "Building Pylon Docker image..."
    docker build -t $IMAGE_NAME .
    
    log "Saving Docker image to $IMAGE_FILE..."
    docker save $IMAGE_NAME | gzip > $IMAGE_FILE
    
    log "Image built and saved successfully. Size: $(du -h $IMAGE_FILE | cut -f1)"
}

# Function to deploy to a single server
deploy_to_server() {
    local server=$1
    local config_file=$2
    
    log "Deploying to $server with config $config_file..."
    
    # Copy image to server
    log "Copying image to $server..."
    scp $IMAGE_FILE $server:
    
    # Copy production compose file
    scp docker/production-compose.yml $server:
    
    # Load image and run experiment on server
    ssh $server << EOF
        set -e
        echo "Loading Docker image..."
        docker load < $IMAGE_FILE
        
        echo "Cleaning up old containers..."
        docker-compose -f production-compose.yml down || true
        
        echo "Starting experiment with config: $config_file"
        CONFIG_FILEPATH="$config_file" docker-compose -f production-compose.yml up -d pylon-experiment
        
        echo "Experiment started on $server"
        docker-compose -f production-compose.yml ps
EOF
    
    log "Deployment to $server completed successfully"
}

# Function to check server status
check_server_status() {
    local server=$1
    
    log "Checking status on $server..."
    ssh $server << 'EOF'
        echo "=== Docker Containers ==="
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        echo -e "\n=== GPU Usage ==="
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        
        echo -e "\n=== Disk Usage ==="
        df -h /shared/storage 2>/dev/null || echo "Shared storage not mounted"
        
        echo -e "\n=== Recent Logs ==="
        docker-compose -f production-compose.yml logs --tail=10 pylon-experiment 2>/dev/null || echo "No recent logs"
EOF
}

# Function to collect results from all servers
collect_results() {
    local results_dir="results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p $results_dir
    
    log "Collecting results to $results_dir/..."
    
    for server in "${SERVERS[@]}"; do
        log "Collecting from $server..."
        mkdir -p "$results_dir/$server"
        
        # Copy logs and results
        scp -r "$server:$SHARED_LOGS_PATH/*" "$results_dir/$server/" || warn "Failed to collect from $server"
    done
    
    log "Results collected in $results_dir/"
}

# Function to clean up deployment
cleanup() {
    log "Cleaning up deployment..."
    
    for server in "${SERVERS[@]}"; do
        log "Cleaning up $server..."
        ssh $server << 'EOF'
            docker-compose -f production-compose.yml down || true
            docker image prune -f || true
            rm -f pylon.tar.gz production-compose.yml || true
EOF
    done
    
    rm -f $IMAGE_FILE
    log "Cleanup completed"
}

# Main script logic
case "${1:-help}" in
    "build")
        build_image
        ;;
    "deploy")
        if [[ -z "$2" ]]; then
            error "Usage: $0 deploy <config_file> [server1,server2,...]"
        fi
        
        config_file=$2
        if [[ ! -f "$config_file" ]]; then
            error "Config file not found: $config_file"
        fi
        
        # Use specific servers if provided
        if [[ -n "$3" ]]; then
            IFS=',' read -ra SERVERS <<< "$3"
        fi
        
        log "Deploying to servers: ${SERVERS[*]}"
        log "Using config: $config_file"
        
        # Build image first
        build_image
        
        # Deploy to all servers in parallel
        for server in "${SERVERS[@]}"; do
            deploy_to_server $server $config_file &
        done
        wait
        
        log "Deployment completed to all servers"
        ;;
    "status")
        for server in "${SERVERS[@]}"; do
            echo "=== $server ==="
            check_server_status $server
            echo
        done
        ;;
    "collect")
        collect_results
        ;;
    "cleanup")
        cleanup
        ;;
    "batch")
        if [[ -z "$2" ]]; then
            error "Usage: $0 batch <config_directory>"
        fi
        
        config_dir=$2
        if [[ ! -d "$config_dir" ]]; then
            error "Config directory not found: $config_dir"
        fi
        
        # Build image once
        build_image
        
        # Deploy each config to different servers (round-robin)
        configs=($(find $config_dir -name "*.py" | sort))
        server_idx=0
        
        for config in "${configs[@]}"; do
            server=${SERVERS[$server_idx]}
            log "Deploying $config to $server"
            
            deploy_to_server $server $config &
            
            server_idx=$(( (server_idx + 1) % ${#SERVERS[@]} ))
            
            # Limit parallel deployments
            if (( $(jobs -r | wc -l) >= 3 )); then
                wait -n  # Wait for at least one job to complete
            fi
        done
        wait
        
        log "Batch deployment completed"
        ;;
    *)
        echo "Pylon Multi-Server Deployment Script"
        echo
        echo "Usage: $0 <command> [options]"
        echo
        echo "Commands:"
        echo "  build                     - Build Docker image"
        echo "  deploy <config> [servers] - Deploy experiment to servers"
        echo "  status                    - Check status of all servers"
        echo "  collect                   - Collect results from all servers"
        echo "  cleanup                   - Clean up all deployments"
        echo "  batch <config_dir>        - Deploy all configs in directory (round-robin)"
        echo
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 deploy configs/experiments/exp1.py"
        echo "  $0 deploy configs/experiments/exp1.py server1,server3"
        echo "  $0 batch configs/experiments/"
        echo "  $0 status"
        echo "  $0 collect"
        ;;
esac