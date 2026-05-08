# Use an official Python runtime as a parent image
FROM anaconda/miniconda:latest

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment file
COPY environment.yaml .
COPY requirements.txt .

# Accept Anaconda Terms of Service for non-interactive environment creation
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the Conda environment
RUN conda env create -f environment.yaml && \
    rm -rf /opt/conda/envs/tlio/lib/python3.9/site-packages/open3d/visualization/tensorboard_plugin

# Verify dependencies
RUN conda run -n tlio python -c "import pkg_resources; print('pkg_resources is available')"

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "tlio", "/bin/bash", "-c"]

# Copy the rest of the application code
COPY . .

# Set the environment variable for Open3D (headless mode by default)
ENV OS_ENV=linux
ENV PYTHONUNBUFFERED=1

# Default command: show help for the main script
CMD ["conda", "run", "--no-capture-output", "-n", "tlio", "python", "src/main_net.py", "--help"]
