import os
import time
import random
from functools import wraps
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from PIL import Image
import streamlit as st

def retry_with_backoff(retries=5, initial_delay=10, backoff_factor=2, jitter=5):
    """
    A decorator for retrying a function with exponential backoff and jitter
    in case of a ResourceExhausted (429) error from the Gemini API.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            logger = kwargs.get('logger') # Assumes 'logger' is a keyword argument
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except ResourceExhausted as e:
                    if i == retries - 1:
                        if logger:
                            logger.error(f"Gemini API call failed after {retries} retries.")
                        raise
                    
                    # Extract retry delay from error if available, otherwise use backoff
                    try:
                        retry_delay = e.metadata[0].retry_delay.seconds + random.uniform(0, jitter)
                    except (AttributeError, IndexError):
                        retry_delay = delay + random.uniform(0, jitter)

                    message = (
                        f"Gemini API rate limit hit. Retrying in {retry_delay:.2f} seconds... "
                        f"(Attempt {i + 1}/{retries})"
                    )
                    if logger:
                        logger.warning(message)
                    st.toast(message, icon="⏳")
                    
                    time.sleep(retry_delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

@st.cache_resource
def setup_gemini(_logger, model_name='gemini-1.5-pro-latest'):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _logger.error("GOOGLE_API_KEY environment variable not set.")
        st.error("GOOGLE_API_KEY environment variable not set. Please create a .env file with your key.")
        return None
    try:
        _logger.info(f"Configuring Google Gemini API with model: {model_name}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        _logger.info("Successfully configured Gemini model.")
        return model
    except Exception as e:
        _logger.exception("Failed to configure Gemini.")
        st.error(f"Failed to configure Gemini: {e}")
        return None

@retry_with_backoff()
def _call_gemini_api(model, prompt, img, logger):
    """Internal function wrapped with retry logic."""
    return model.generate_content([prompt, img])

def get_frame_description(model, image_path, logger):
    """Uses Gemini to generate a description of an image frame from a video editing perspective."""
    if not model:
        logger.error("Gemini model not initialized, cannot get frame description.")
        return "Gemini model not initialized."
    
    logger.info(f"Requesting Gemini description for {os.path.basename(image_path)}")
    try:
        img = Image.open(image_path)
        prompt = """
        Analyze this video frame from a video editor's perspective. Be concise but detailed. Describe the following:
        1.  **Composition**: Analyze the shot composition (e.g., rule of thirds, leading lines, framing, depth of field, symmetry).
        2.  **Lighting**: Describe the lighting setup (e.g., key light, fill light, backlight, high-key/low-key, natural/artificial) and the mood it creates.
        3.  **Subject and Action**: Identify the main subject and describe any ongoing action or emotion conveyed.
        4.  **Editing Potential**: Evaluate this frame as an edit point. Is this a good moment to cut to another shot, apply a transition, or hold for dramatic effect? Explain why.
        """
        response = _call_gemini_api(model=model, prompt=prompt, img=img, logger=logger)
        logger.info(f"Successfully received Gemini description for {os.path.basename(image_path)}")
        return response.text
    except Exception as e:
        logger.exception(f"Gemini API call failed for {os.path.basename(image_path)}")
        st.warning(f"Could not get Gemini description for {os.path.basename(image_path)}: {e}")
        return "Error generating description."
