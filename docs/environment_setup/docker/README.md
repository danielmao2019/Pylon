# Docker Environment Setup

This directory contains all files needed for Docker-based Pylon deployment.

## Quick Start

### 1. Basic Setup
```bash
# Build Docker image (from Pylon root directory)
docker build -f docs/environment_setup/docker/Dockerfile -t pylon:latest .

# Run with configurable mounts
./docs/environment_setup/docker/docker-run.sh \
  -d "/your/data/path:/pub5/data:ro" \
  -l "./logs:/workspace/logs" \
  -- python main.py --config configs/examples/linear/config.py --debug
```

### 2. Development Setup
```bash
# Use docker-compose for development (configure mounts first)
cd docs/environment_setup/docker/
# Edit docker-compose.yml to uncomment and set your data/logs paths
docker-compose run --rm pylon
```

### 3. Production Multi-Server Deployment
```bash
# Use automated deployment script
./docs/environment_setup/docker/multi-server-deploy.sh deploy configs/your_experiment.py

# Monitor across servers
./docs/environment_setup/docker/multi-server-deploy.sh status
```

## Files Overview

### Core Docker Files
- **[`Dockerfile`](Dockerfile)** - Multi-stage Docker image with all dependencies
- **[`docker-compose.yml`](docker-compose.yml)** - Local development setup
- **[`.dockerignore`](.dockerignore)** - Build context optimization

### Deployment Tools
- **[`docker-run.sh`](docker-run.sh)** - Configurable container launcher with mount options
- **[`production-compose.yml`](production-compose.yml)** - Production multi-server configuration
- **[`multi-server-deploy.sh`](multi-server-deploy.sh)** - Automated multi-server deployment

## Key Features

### 🔧 Configurable Mounts
No hardcoded paths - specify your data and logs directories:

```bash
# Example: Configure for your server setup
./docker-run.sh \
  -d "/pub5/data:/pub5/data:ro" \          # Your dataset storage
  -d "/pub4/daniel:/pub4/daniel:ro" \      # Personal datasets  
  -l "/shared/logs:/workspace/logs" \      # Shared experiment logs
  -p "8050:8050" \                         # Web viewer port
  -- python main.py --config configs/exp1.py
```

### 🚀 Multi-Server Deployment
Single command deploys across multiple servers:

```bash
# Edit server list in multi-server-deploy.sh
# Then deploy automatically
./multi-server-deploy.sh batch configs/experiments/
```

### 📊 Performance Optimized
- **Multi-stage build**: Smaller production images
- **Bind mounts**: Zero data duplication for large datasets
- **Native I/O**: Same performance as host filesystem
- **GPU access**: Full CUDA support with `--gpus all`

## Configuration Guide

### Docker Compose Setup
1. **Edit `docker-compose.yml`**:
   ```yaml
   volumes:
     - ../..:/workspace/pylon
     # Uncomment and configure for your setup:
     - /your/data/path:/pub5/data:ro
     - /your/logs/path:/workspace/logs
   ```

2. **Run development container**:
   ```bash
   docker-compose run --rm pylon
   ```

### Production Deployment
1. **Configure servers** in `multi-server-deploy.sh`:
   ```bash
   SERVERS=("your-server1" "your-server2" "your-server3")
   ```

2. **Deploy experiments**:
   ```bash
   ./multi-server-deploy.sh deploy configs/exp1.py
   ./multi-server-deploy.sh deploy configs/exp2.py server1,server3  # Specific servers
   ```

### Manual Docker Run
Use `docker-run.sh` for maximum flexibility:

```bash
# Interactive debugging
./docker-run.sh --interactive -d "/data:/workspace/data" -l "./logs:/workspace/logs"

# Background experiment
./docker-run.sh --detach -d "/pub5/data:/pub5/data:ro" -l "/shared/logs:/workspace/logs" -- python main.py --config configs/exp1.py

# Disable GPU for CPU-only experiments
./docker-run.sh -g none -d "/data:/workspace/data" -- python main.py --config configs/cpu_exp.py
```

## Advanced Usage

### Custom Image Builds
```bash
# Build with custom CUDA version
docker build --build-arg CUDA_VERSION=12.0 -f Dockerfile -t pylon:cuda12 .

# Development vs production images
docker build --target builder -t pylon:dev .    # With build tools
docker build --target runtime -t pylon:prod .   # Optimized runtime
```

### Resource Management
```bash
# Limit resources
./docker-run.sh \
  --docker-args "--cpus=4.0 --memory=16g --shm-size=8g" \
  -d "/data:/workspace/data" \
  -- python main.py --config configs/exp1.py
```

### Troubleshooting
- **GPU not accessible**: `docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi`
- **Permission issues**: `./docker-run.sh --docker-args "--user $(id -u):$(id -g)"`
- **Volume mount problems**: Check paths exist and are accessible
- **Out of space**: `docker system prune -a` to clean up old images

For detailed Docker concepts and I/O performance information, see the main [environment setup guide](../README.md).