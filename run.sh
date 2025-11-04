#!/bin/sh
# run.sh: Activates the virtual environment and starts the vChat web server.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
VENV_DIR=".venv"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"

# Allow user to override port via command line argument, e.g., ./run.sh 8080
PORT=${1:-$DEFAULT_PORT}

# --- Pre-run Checks ---
if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: Virtual environment directory './$VENV_DIR' not found." >&2
  echo "Please run './setup.sh' first to create the environment and install dependencies." >&2
  exit 1
fi

# --- Environment Setup ---
# Activate the virtual environment
# shellcheck source=/dev/null
. "./$VENV_DIR/bin/activate"

# Set HF_HOME to use a local cache directory.
# This is crucial to prevent writing to the system's default ~/.cache directory,
# avoiding potential permission or quota issues.
export HF_HOME
HF_HOME="$(pwd)/huggingface_cache"
export TRANSFORMERS_CACHE
TRANSFORMERS_CACHE="$HF_HOME/models"

echo "INFO: Virtual environment activated."
echo "INFO: Hugging Face cache set to: $HF_HOME"

# --- Start Application ---
echo "\nINFO: Starting vChat application server..."
echo "URL: http://localhost:$PORT (or http://<server_ip>:$PORT)"
echo "Press Ctrl+C to stop the server."

# Run the uvicorn server.
# Using python -m uvicorn to ensure it's the one from the venv.
# Add --reload for development if you want the server to restart on code changes.
python -m uvicorn app:app --host "$DEFAULT_HOST" --port "$PORT"

./run.sh 8081