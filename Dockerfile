# Use NVIDIA CUDA base image for GPU support
# Compatible with both x86_64 (NVIDIA GPUs) and arm64 (CPU mode)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables for NVIDIA and Conda
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PATH="/opt/conda/bin:${PATH}"
ARG PATH="/opt/conda/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (handles multiple architectures automatically)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Copy environment files
COPY environment.yaml .
COPY requirements.txt .

# Create the Conda environment
# If no NVIDIA hardware is present, PyTorch will automatically default to CPU.
RUN conda env create -f environment.yaml && \
    # Ensure pytorch-cuda is installed on x86_64 systems for GPU support
    if [ "$(uname -m)" = "x86_64" ]; then \
        conda install -n tlio -y pytorch-cuda=12.1 -c pytorch -c nvidia; \
    fi && \
    conda clean -afy && \
    rm -rf /opt/conda/envs/tlio/lib/python3.9/site-packages/open3d/visualization/tensorboard_plugin

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
