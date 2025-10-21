# open-vRAG/video_rag/Dockerfile
# Start from NVIDIA's CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set working directory
WORKDIR /app

# Install system dependencies
# This layer is cached as it rarely changes.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libnss3 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libexpat1 \
    libgbm1 \
    libasound2 \
    xdg-utils \
    fonts-liberation \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome for Selenium
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    dpkg -i google-chrome-stable_current_amd64.deb; apt-get -fy install && \
    rm google-chrome-stable_current_amd64.deb

# --- Docker Cache Optimization ---

# Install PyTorch with CUDA 11.8 support FIRST
# This is a large, slow-to-install dependency that rarely changes.
# By installing it before copying requirements.txt, this layer will be cached
# and will NOT be re-run even if requirements.txt or app code changes.
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# Install the rest of the Python dependencies
# This layer is only re-built if the requirements.txt file changes.
RUN pip3 install --no-cache-dir -r requirements.txt

# Now, copy the rest of the application code
# Since this is one of the last steps, changes to your source code (e.g., app.py)
# will only invalidate this layer, leading to very fast rebuilds.
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]