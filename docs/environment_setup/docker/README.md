# Docker Setup for Pylon

## Quick Start

### 1. Build the Docker Image
```bash
# Build the image (takes 15-30 minutes first time)
docker build -t pylon:latest .

# Alternative: Use docker-compose 
docker-compose build
```

### 2. Run Experiments

**Single experiment:**
```bash
# Run training experiment
docker run --gpus all -v $(pwd):/workspace/pylon -v $(pwd)/logs:/workspace/logs \
  pylon:latest conda run -n pylon python main.py --config-filepath configs/examples/linear/config.py

# Interactive development
docker run --gpus all -it -v $(pwd):/workspace/pylon pylon:latest
```

**Using docker-compose (recommended):**
```bash
# Start interactive container
docker-compose run --rm pylon

# Inside container, run experiments:
python main.py --config-filepath configs/examples/linear/config.py --debug
python -m data.viewer.cli --host 0.0.0.0 --port 8050
```

## Docker I/O and Data Management

### Key Concepts for Large-Scale Research
- **Volumes are bind mounts**: Direct filesystem access, NO data copying
- **Symlinks preserved**: Container sees same `/pub5/data/` paths as host
- **Zero data duplication**: 500GB dataset ≠ 500GB container overhead
- **Bidirectional sync**: Changes to logs immediately visible on host and other containers

### Your Pylon I/O Pattern
```bash
# Host filesystem structure
/pub5/data/ -> Large shared datasets (read-only)
/pub4/daniel/ -> Personal datasets (read-only)  
./logs/ -> Experiment outputs (read-write, shared)
./data/datasets/soft_links/ -> Symlinks to above (preserved in container)
```

### Image vs Container
- **Image**: Immutable template with your software stack (like a VM snapshot)
- **Container**: Running instance of an image (temporary, disposable)
- **Key insight**: One image → many containers (parallel experiments)

### Reusability Patterns

**✅ Recommended: Build once, run everywhere**
```bash
# Build on one server
docker build -t pylon:latest .
docker save pylon:latest | gzip > pylon.tar.gz

# Transfer and run on other servers
scp pylon.tar.gz server2:
ssh server2 "docker load < pylon.tar.gz"
ssh server2 "docker run --gpus all -v /data:/workspace/data pylon:latest python main.py ..."
```

**✅ Parallel experiments on same server:**
```bash
# Experiment 1
docker run --name exp1 --gpus '"device=0"' -v $(pwd):/workspace/pylon pylon:latest python main.py --config configs/exp1.py

# Experiment 2 (parallel)
docker run --name exp2 --gpus '"device=1"' -v $(pwd):/workspace/pylon pylon:latest python main.py --config configs/exp2.py
```

### Volume Patterns

**Code + Results persistence:**
```bash
docker run --gpus all \
  -v $(pwd):/workspace/pylon \          # Source code
  -v $(pwd)/logs:/workspace/logs \      # Experiment results  
  -v /datasets:/workspace/datasets \    # External datasets
  pylon:latest python main.py ...
```

## Production Workflows

### 1. Development Workflow
```bash
# Interactive development
docker-compose run --rm pylon

# Inside container:
# - Edit code (changes persist via volume mount)
# - Run experiments 
# - Results saved to ./logs/ (persists)
```

### 2. Batch Experiment Workflow
```bash
# Run multiple experiments in background
for config in configs/experiments/*.py; do
  docker run -d --name "exp_$(basename $config)" \
    --gpus all -v $(pwd):/workspace/pylon \
    pylon:latest python main.py --config-filepath "$config"
done

# Monitor progress
docker logs -f exp_config1.py
```

### 3. Multi-Server Production Deployment
```bash
# Server 1: Build and export
docker build -t pylon:latest .
docker save pylon:latest | gzip > pylon.tar.gz

# Servers 2-N: Import and run with shared storage
scp pylon.tar.gz server2:
ssh server2 "
  docker load < pylon.tar.gz
  docker run --gpus all \
    -v /pub5/data:/pub5/data:ro \
    -v /pub4/daniel:/pub4/daniel:ro \
    -v /shared/storage/pylon_logs:/workspace/logs \
    pylon:latest python main.py --config-filepath configs/exp1.py
"

# Alternative: Use production compose file
scp docker/production-compose.yml server2:
ssh server2 "CONFIG_FILEPATH=configs/exp1.py docker-compose -f production-compose.yml up pylon-experiment"
```

## Best Practices

### Resource Management
```bash
# Limit GPU memory
docker run --gpus '"device=0"' --shm-size=8g pylon:latest

# Limit CPU/memory
docker run --cpus="4.0" --memory="16g" pylon:latest
```

### Container Lifecycle
```bash
# Remove containers after completion
docker run --rm --gpus all pylon:latest

# Keep containers for debugging
docker run --name debug-exp pylon:latest
docker exec -it debug-exp bash  # Debug interactively
```

### Image Management
```bash
# List images and sizes
docker images

# Remove old images
docker image prune

# Clean up everything
docker system prune -a
```

## Troubleshooting

### Common Issues

**GPU not accessible:**
```bash
# Check NVIDIA Docker runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**Volume permissions:**
```bash
# Fix permission issues
docker run --user $(id -u):$(id -g) pylon:latest
```

**Out of disk space:**
```bash
# Clean up Docker cache
docker builder prune
docker system prune -a
```

### Debugging Containers
```bash
# Inspect running container
docker exec -it container_name bash

# Check container logs
docker logs container_name

# Copy files from container
docker cp container_name:/workspace/logs/results.json .
```

## Advanced Usage

### Custom Build Args
```dockerfile
# In Dockerfile
ARG CUDA_VERSION=11.8
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Build with custom CUDA version
docker build --build-arg CUDA_VERSION=12.0 -t pylon:cuda12 .
```

### Multi-stage Development
```bash
# Development image (with additional tools)
docker build --target builder -t pylon:dev .

# Production image (smaller)
docker build --target runtime -t pylon:prod .
```

This Docker setup ensures identical environments across all your servers while maintaining the flexibility for parallel experiments and easy deployment.