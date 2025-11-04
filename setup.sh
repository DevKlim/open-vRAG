#!/bin/sh
# setup.sh: Prepares a Python virtual environment and installs dependencies.
# This script is designed to run in a user's home directory without sudo privileges.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
PYTHON_EXE="python3.11"
VENV_DIR=".venv"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# --- Helper Functions ---
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo_info() {
  echo "INFO: $1"
}

echo_warn() {
  echo "WARN: $1"
}

echo_error() {
  echo "ERROR: $1" >&2
  exit 1
}

# --- Prerequisite Checks ---
echo_info "Step 1: Checking prerequisites..."

# Check for Python 3.11
if ! command_exists $PYTHON_EXE; then
  echo_error "$PYTHON_EXE is not found. Please ensure Python 3.11 is installed and in your PATH."
  echo_info "On many systems (like DSMLP), you may need to load a module first, e.g., 'module load python/3.11'"
fi

# Check for ffmpeg and git
if ! command_exists ffmpeg; then
  echo_warn "ffmpeg command not found. Video downloading/processing will likely fail."
  echo_warn "Please install ffmpeg or load the appropriate module."
fi
if ! command_exists git; then
  echo_warn "git command not found. This is needed for version control."
fi
echo_info "Prerequisite check passed."

# --- Virtual Environment Setup ---
echo_info "\nStep 2: Setting up Python virtual environment in './$VENV_DIR'..."
if [ -d "$VENV_DIR" ]; then
  echo_info "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
  $PYTHON_EXE -m venv $VENV_DIR
  echo_info "Virtual environment created."
fi

# Activate the virtual environment for this script's session
# shellcheck source=/dev/null
. "./$VENV_DIR/bin/activate"
echo_info "Virtual environment activated."

# --- Dependency Installation ---
echo_info "\nStep 3: Installing dependencies..."

# Upgrade pip and install uv
pip install --upgrade pip
pip install uv
echo_info "Installed 'uv' package manager."

# Install packages from requirements.txt
echo_info "Installing packages from requirements.txt using uv..."
uv pip install -r requirements.txt
echo_info "Base requirements installed."

# Install flash-attention
echo_info "Installing flash-attention from wheel..."
echo_info "URL: $FLASH_ATTN_URL"
echo_info "(This requires a compatible environment: PyTorch>=2.3, CUDA>=12.2, Python 3.11)"
uv pip install "$FLASH_ATTN_URL"
echo_info "flash-attention installed."

echo_info "Dependency installation complete."

# --- Directory Setup ---
echo_info "\nStep 4: Creating required directories..."
mkdir -p videos
mkdir -p huggingface_cache
mkdir -p data
mkdir -p lora_adapters
echo_info "Created directories: videos/, huggingface_cache/, data/, lora_adapters/"

# --- Final Message ---
echo_info "\n========================================="
echo_info "Setup complete!"
echo_info "To run the application, execute: ./run.sh"
echo_info "========================================="