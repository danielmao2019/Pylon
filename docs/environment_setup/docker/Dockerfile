# Multi-stage Dockerfile for Pylon Deep Learning Framework
# Stage 1: Build environment with all compilation tools
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    # Essential build tools
    gcc-9 g++-9 make cmake \
    # Python and package management
    python3.10 python3.10-dev python3-pip \
    # Git for repository cloning
    git \
    # Download utilities
    wget curl \
    # Additional libraries for computer vision
    libgl1-mesa-glx libglib2.0-0 \
    # Clean up to reduce layer size
    && rm -rf /var/lib/apt/lists/*

# Set up GCC alternatives to use version 9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Update conda and create Pylon environment  
RUN conda update -n base -c defaults conda -y && \
    conda create --name pylon python=3.10 -y

# Make conda activate work in Docker
SHELL ["conda", "run", "-n", "pylon", "/bin/bash", "-c"]

# Copy and run package installation script
COPY docs/environment_setup/install_packages.sh /tmp/install_packages.sh
RUN chmod +x /tmp/install_packages.sh && \
    CONDA_DEFAULT_ENV=pylon /tmp/install_packages.sh && \
    rm /tmp/install_packages.sh

# Set working directory for building extensions
WORKDIR /workspace

# Copy only the necessary files for building C++ extensions first (for caching)
COPY data/collators/geotransformer/ ./data/collators/geotransformer/
COPY data/collators/buffer/cpp_wrappers/ ./data/collators/buffer/cpp_wrappers/
COPY data/collators/overlappredator/cpp_wrappers/ ./data/collators/overlappredator/cpp_wrappers/

# Build C++ extensions
RUN cd data/collators/geotransformer && python setup.py install
RUN cd data/collators/buffer/cpp_wrappers && bash compile_wrappers.sh
RUN cd data/collators/overlappredator/cpp_wrappers && bash compile_wrappers.sh

# Clone and build external repositories
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git && \
    cd Pointnet2_PyTorch && pip install pointnet2_ops_lib/.

RUN git clone https://github.com/KinglittleQ/torch-batch-svd.git && \
    cd torch-batch-svd && python setup.py install

# Stage 2: Runtime environment (smaller, production-ready)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    libgl1-mesa-glx libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder stage
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy built extensions from builder stage
COPY --from=builder /workspace/data/collators/ /workspace/data/collators/
COPY --from=builder /workspace/Pointnet2_PyTorch /workspace/Pointnet2_PyTorch
COPY --from=builder /workspace/torch-batch-svd /workspace/torch-batch-svd

# Set up conda environment activation
SHELL ["conda", "run", "-n", "pylon", "/bin/bash", "-c"]

# Set working directory
WORKDIR /workspace

# Create directory for mounting Pylon code
RUN mkdir -p /workspace/pylon

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Default command activates conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "pylon", "/bin/bash"]