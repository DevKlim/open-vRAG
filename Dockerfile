# /Dockerfile
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# Add a build argument to invalidate the cache if it becomes corrupted.
# This helps resolve "bad record MAC" errors without manual cache cleaning.
ARG CACHE_BUSTER=1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHON_PACKAGES_PATH=/opt/conda/lib/python3.11/site-packages

# Install base dependencies and the Google Cloud CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies into the system python environment.
# This step is slow but will be cached unless requirements.txt changes.
COPY requirements.txt .
RUN python3 -m pip install uv && \
    uv pip install \
    --system \
    -r requirements.txt \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Copy the application code into the image.
COPY . .

# Expose the port the web server will run on
EXPOSE 8000

# The default command to run the web server.
# The executable is found in /opt/conda/bin which is already in the PATH.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]