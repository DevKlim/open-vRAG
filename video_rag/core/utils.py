import os
import re
from datetime import datetime

def get_job_directory(base_dir="data/processed"):
    """Creates a unique directory for a processing job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(base_dir, f"job_{timestamp}")
    
    # Create subdirectories
    os.makedirs(os.path.join(job_dir, "frames"), exist_ok=True)
    
    return job_dir

def seconds_to_filename_str(seconds):
    """Converts seconds into a file-safe string for filenames."""
    # Using integer seconds for simplicity and to avoid floating point issues in filenames
    sec = int(seconds)
    return f"{sec:06d}s"

def clean_filename(filename):
    """Removes invalid characters from a string to make it a safe filename."""
    # Remove path-like structures and keep the base name
    filename = os.path.basename(filename)
    # Remove file extension
    filename = os.path.splitext(filename)[0]
    # Replace invalid characters with an underscore
    return re.sub(r'[\\/*?:"<>|]', "_", filename)
