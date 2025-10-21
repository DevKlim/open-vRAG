import logging
import os

def setup_logger(job_dir, log_filename='analysis.log'):
    """
    Sets up a logger that writes to a file in the job directory and to the console.
    """
    log_filepath = os.path.join(job_dir, log_filename)
    
    # Get a logger instance. Using a unique name is good practice.
    logger = logging.getLogger(f"job_{os.path.basename(job_dir)}")
    logger.setLevel(logging.INFO) # Set the minimum level of messages to handle
    
    # Prevent handlers from being added multiple times if the function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    
    # Create a stream handler to print logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger