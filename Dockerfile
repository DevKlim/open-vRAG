# vChat/Dockerfile
# Use a base image with CUDA 12.1 and Python 3.11 pre-installed
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/models
# Add pip's local bin to the PATH so 'uv' is available after installation
ENV PATH="/root/.local/bin:${PATH}"

# Verify Python version is 3.11
RUN python --version && \
    python -c "import sys; assert sys.version_info.major == 3 and sys.version_info.minor == 11, 'This Dockerfile requires a base image with Python 3.11.'"

# Install system dependencies, including ffmpeg for video processing and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Docker Layer Caching Optimization ---
# 1. Copy only the requirements file. This layer only changes if requirements.txt is modified.
COPY requirements.txt .

# 2. Install the Python dependencies. This slow step will now be cached and only re-run
#    if the requirements.txt file (from the layer above) has changed.
RUN python3 -m pip install uv && \
    uv pip install \
    --system \
    -r requirements.txt \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 3. Copy the rest of the application code. This layer changes frequently, but because it's
#    at the end, it won't invalidate the cached layers above.
COPY . .

# Expose the port the web server will run on
EXPOSE 8000

# The default command to run the web server. For development, we will override this
# in docker-compose.yml to enable auto-reloading.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]