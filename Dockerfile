# vChat/Dockerfile
# Use a base image with CUDA and PyTorch pre-installed for compatibility
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/models
# Add pip's local bin to the PATH so 'uv' is available after installation
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies, including ffmpeg for video processing and git to clone the repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app


# Copy the requirements file
COPY requirements.txt .

# Install all Python dependencies using uv for speed
RUN python3 -m pip install uv && \
    uv pip install \
    --system \
    -r requirements.txt \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.18/flash_attn-2.6.3+cu130torch2.9-cp311-cp311-linux_x86_64.whl
COPY . .

# Expose the port the web server will run on
EXPOSE 8000

# Command to run the web server
# Use 'python -m uvicorn' to avoid PATH issues inside the container
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]