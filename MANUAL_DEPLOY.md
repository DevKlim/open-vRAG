<!-- vChat/MANUAL_DEPLOY.md -->
# Manual Deployment Guide for DSMLP (Non-Docker, Non-Kubernetes)

This guide provides instructions for setting up and running the vChat application directly on a DSMLP login or compute node without using Docker or Kubernetes. This method is ideal if you are encountering permission errors or prefer a more direct setup.

The process involves creating a self-contained Python environment in your project directory, ensuring you don't need `sudo` or administrator privileges.

## Prerequisites

1.  **SSH Access**: You must have SSH access to a DSMLP machine (e.g., `dsmlp-login.ucsd.edu`).
2.  **Required Software**: The following commands must be available on the remote machine:
    *   `git`
    *   `python3.11`
    *   `ffmpeg`
3.  **GPU**: The local models require an NVIDIA GPU with a compatible CUDA toolkit (version 12.2 or newer is recommended).

### DSMLP Module System

DSMLP environments often use a module system to manage software versions. Before you begin, you may need to load the correct modules. Connect to the DSMLP machine and run commands similar to these (the exact names might vary, check your system's documentation):

```bash
# Example commands - adjust for your environment
module load anaconda3 # or another Python distribution
conda activate your_env_with_python3.11

# Or, if using a different module system:
module load python/3.11
module load cuda/12.2 # or a newer version
module load ffmpeg
```
**Verify your Python version is 3.11**:
```bash
python3.11 --version
# Should output: Python 3.11.x
```
If you don't have `python3.11` in your path, you might need to use `python` or `python3` depending on your environment setup. If so, you will need to edit the `PYTHON_EXE` variable at the top of the `setup.sh` script.

---

## Step 1: Clone the Repository

Connect to the DSMLP machine and clone the project into your home directory.

```bash
ssh your_username@dsmlp-login.ucsd.edu
git clone <your-repo-url>
cd vChat/
```

---

## Step 2: Run the Setup Script

The `setup.sh` script automates the entire setup process. It will create a local Python virtual environment, install all required dependencies, and create necessary data directories.

Make the script executable and run it:
```bash
chmod +x setup.sh run.sh
./setup.sh
```
This script may take several minutes to complete as it downloads and installs packages. It only needs to be run once. If `requirements.txt` changes, you can run it again to update your environment.

---

## Step 3: Run the Application

The `run.sh` script starts the web server.

```bash