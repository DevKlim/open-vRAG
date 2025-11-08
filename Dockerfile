# Dockerfile
# A multi-stage build to drastically reduce the final image size.
# Stage 1: Builder - uses the larger 'devel' image to build dependencies.
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Define a variable for the site-packages directory for easy copying later
ENV PYTHON_PACKAGES_PATH=/opt/conda/lib/python3.11/site-packages

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
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

# ---
# Stage 2: Final Image - uses the smaller 'runtime' image.
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/models
ENV PYTHON_PACKAGES_PATH=/opt/conda/lib/python3.11/site-packages

# Install only the required runtime system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the builder stage.
COPY --from=builder ${PYTHON_PACKAGES_PATH} ${PYTHON_PACKAGES_PATH}
# Copy any executables installed by pip (like uvicorn, uv) from the builder.
# When using --system in the base image, they are installed in /opt/conda/bin.
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Copy the application code into the final image.
COPY . .

# Expose the port the web server will run on
EXPOSE 8000

# The default command to run the web server.
# The executable is found in /opt/conda/bin which is already in the PATH.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]