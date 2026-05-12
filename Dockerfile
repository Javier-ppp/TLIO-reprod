# Use NVIDIA CUDA base image for GPU support
# Compatible with both x86_64 (NVIDIA GPUs) and arm64 (CPU mode)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables for NVIDIA and Conda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/opt/conda/bin:${PATH}"
ARG PATH="/opt/conda/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive


# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    unzip \
    bzip2 \
    ca-certificates \
    build-essential \
    pandoc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda (handles multiple architectures automatically)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Miniconda for current architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_ARCH="x86_64"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $ARCH"; \
        exit 1; \
    fi && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda install -n base -c conda-forge mamba -y && \
    conda clean -afy

# Set the working directory in the container
WORKDIR /app

# Copy environment files
COPY environment.yaml .
COPY requirements.txt .

# Create the Mamba environment
# If no NVIDIA hardware is present, PyTorch will automatically default to CPU.

RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -f environment.yaml && \
    conda clean -afy && \
    if [ "$(uname -m)" = "x86_64" ]; then \
        mamba install -n tlio -y pytorch-cuda=12.1 -c pytorch -c nvidia; \
    fi && \
    conda clean -afy && \
    rm -rf /opt/conda/envs/tlio/lib/python3.9/site-packages/open3d/visualization/tensorboard_plugin && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "tlio", "/bin/bash", "-c"]

# Copy the rest of the application code
COPY . .

# Set runtime environment variables
ENV OS_ENV=linux
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Default command: show help for the main script
CMD ["conda", "run", "--no-capture-output", "-n", "tlio", "python", "src/main_net.py", "--help"]
