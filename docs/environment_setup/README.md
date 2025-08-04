# Pylon Environment Setup

This directory contains all environment setup documentation and configuration files for Pylon.

## Setup Options

### 1. Conda Environment (Traditional)
- **Directory**: [`conda/`](conda/)
- **Use case**: Local development, single machine setup
- **Pros**: Direct control, familiar workflow
- **Cons**: Manual setup on each server, environment drift

### 2. Docker Environment (Recommended)
- **Directory**: [`docker/`](docker/)
- **Use case**: Multi-server deployment, consistent environments
- **Pros**: Perfect reproducibility, easy scaling, isolated environments
- **Cons**: Additional Docker knowledge required

## Quick Start

### Docker (Recommended for Multi-Server)
```bash
# Build and run with configurable mounts
docker build -f docs/environment_setup/docker/Dockerfile -t pylon:latest .
./docs/environment_setup/docker/docker-run.sh \
  -d "/your/data/path:/pub5/data:ro" \
  -l "./logs:/workspace/logs" \
  -- python main.py --config configs/examples/linear/config.py --debug

# Or use deployment script for multi-server
./docs/environment_setup/docker/multi-server-deploy.sh deploy configs/examples/linear/config.py
```

### Conda (Traditional)
```bash
# Automated installation
conda create --name pylon python=3.10 -y && conda activate pylon
bash docs/environment_setup/install_packages.sh

# Or use environment.yml
conda env create -f docs/environment_setup/conda/environment.yml
conda activate pylon
```

## Directory Overview

| Directory/File | Purpose |
|------|---------|
| [`install_packages.sh`](install_packages.sh) | **Shared installation script** for conda and Docker |
| [`conda/`](conda/) | Traditional conda environment setup |
| [`docker/`](docker/) | Docker-based deployment and multi-server tools |

### Conda Directory
| File | Purpose |
|------|---------|
| [`conda/README.md`](conda/README.md) | Conda setup guide and overview |
| [`conda/conda.md`](conda/conda.md) | Detailed manual setup instructions |
| [`conda/environment.yml`](conda/environment.yml) | Conda environment specification |

### Docker Directory
| File | Purpose |
|------|---------|
| [`docker/README.md`](docker/README.md) | Docker setup guide and usage |
| [`docker/Dockerfile`](docker/Dockerfile) | Multi-stage Docker image definition |
| [`docker/docker-compose.yml`](docker/docker-compose.yml) | Local development setup |
| [`docker/docker-run.sh`](docker/docker-run.sh) | Configurable container launcher |
| [`docker/production-compose.yml`](docker/production-compose.yml) | Production multi-server setup |
| [`docker/multi-server-deploy.sh`](docker/multi-server-deploy.sh) | Automated deployment script |
| [`docker/.dockerignore`](docker/.dockerignore) | Build context exclusions |

## Choosing Your Setup

**Choose Docker if:**
- Running experiments on multiple servers
- Want guaranteed environment consistency
- Need to avoid "works on my machine" problems
- Working in a team with shared infrastructure

**Choose Conda if:**
- Developing primarily on a single machine
- Need direct access to Python debugging
- Want full control over package versions
- Prefer traditional Python development workflow

## Migration Path

If you're currently using conda and want to move to Docker:

1. **Test compatibility**: Run existing experiments in Docker container
2. **Validate data paths**: Ensure volume mounts work with your datasets
3. **Update deployment**: Use provided scripts for multi-server deployment
4. **Train team**: Share Docker documentation with collaborators

## Troubleshooting

Common issues and solutions are documented in:
- [`docker/README.md`](docker/README.md) - Docker-specific troubleshooting
- [`conda.md`](conda.md) - Conda environment issues

For additional help, see the main [troubleshooting section](../README.md) in the docs.