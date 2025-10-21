import os
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configure the Gemini API key
@st.cache_resource
def setup_gemini(logger):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        st.error("GOOGLE_API_KEY environment variable not set. Please create a .env file with your key.")
        return None
    try:
        logger.info("Configuring Google Gemini API...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        logger.info("Successfully configured Gemini model 'gemini-1.5-pro-latest'.")
        return model
    except Exception as e:
        logger.exception("Failed to configure Gemini.")
        st.error(f"Failed to configure Gemini: {e}")
        return None

def get_frame_description(model, image_path, logger):
    """
    Uses Gemini to generate a description of an image frame from a video editing perspective.
    """
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
        response = model.generate_content([prompt, img])
        logger.info(f"Successfully received Gemini description for {os.path.basename(image_path)}")
        return response.text
    except Exception as e:
        logger.exception(f"Gemini API call failed for {os.path.basename(image_path)}")
        st.warning(f"Could not get Gemini description for {os.path.basename(image_path)}: {e}")
        return "Error generating description."
