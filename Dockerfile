#due to training cutoff, this is the MODERN version as of 2025 Dec
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LITE_MODE=false \
    PATH="/usr/lib/google-cloud-sdk/bin:$PATH"

WORKDIR /app

# 1. Install System Dependencies & Google Cloud SDK
# Combined into a single layer for optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ninja-build \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    gnupg \
    ca-certificates \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv for faster pip operations
RUN pip install uv

# 3. Install Python Dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

# 4. Copy Application Code
COPY . .

# Expose the API port
EXPOSE 8000

# Run using standard python command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]