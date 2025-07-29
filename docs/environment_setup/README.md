# Pylon Environment Setup

This directory contains all environment setup documentation and configuration files for Pylon.

## Setup Options

### 1. Conda Environment (Traditional)
- **File**: [`conda.md`](conda.md)
- **Use case**: Local development, single machine setup
- **Pros**: Direct control, familiar workflow
- **Cons**: Manual setup on each server, environment drift

### 2. Docker Environment (Recommended)
- **Files**: [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml), [`docker/`](docker/)
- **Use case**: Multi-server deployment, consistent environments
- **Pros**: Perfect reproducibility, easy scaling, isolated environments
- **Cons**: Additional Docker knowledge required

## Quick Start

### Docker (Recommended for Multi-Server)
```bash
# Build and run
docker build -f docs/environment_setup/Dockerfile -t pylon:latest .
docker-compose -f docs/environment_setup/docker-compose.yml run --rm pylon

# Or use deployment script
./docs/environment_setup/docker/multi-server-deploy.sh build
./docs/environment_setup/docker/multi-server-deploy.sh deploy configs/examples/linear/config.py
```

### Conda (Traditional)
```bash
# Follow the conda setup guide
cat docs/environment_setup/conda.md

# Or use environment.yml
conda env create -f docs/environment_setup/environment.yml
conda activate pylon
```

## File Overview

| File | Purpose |
|------|---------|
| [`conda.md`](conda.md) | Traditional conda environment setup guide |
| [`environment.yml`](environment.yml) | Conda environment specification file |
| [`Dockerfile`](Dockerfile) | Multi-stage Docker image definition |
| [`docker-compose.yml`](docker-compose.yml) | Local Docker development setup |
| [`.dockerignore`](.dockerignore) | Docker build context exclusions |
| [`docker/README.md`](docker/README.md) | Comprehensive Docker usage guide |
| [`docker/production-compose.yml`](docker/production-compose.yml) | Production multi-server setup |
| [`docker/multi-server-deploy.sh`](docker/multi-server-deploy.sh) | Automated deployment script |

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