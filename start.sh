#!/bin/bash

# 1. Start Python FastAPI in the background
# We set PYTHONPATH to include the /app/src directory so imports work correctly
# We run from /app so that data/ and videos/ directories are created in the root volume mount
echo "Starting Python Inference Engine..."
export PYTHONPATH=$PYTHONPATH:/app/src
python -m uvicorn src.app:app --host 127.0.0.1 --port 8001 &

# Wait for Python to initialize
sleep 5

# 2. Start Golang Web Server
echo "Starting Go Web Server..."
# FIXED: Run the binary from the system path
/usr/local/bin/vchat-server