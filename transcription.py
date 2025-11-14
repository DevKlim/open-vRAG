import whisper
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
transcription_model = None

def load_model():
    """Loads the Whisper transcription model."""
    global transcription_model
    if transcription_model is None:
        try:
            # Using 'base.en' for a good balance of speed and accuracy with English videos.
            # For multilingual support, 'base' could be used.
            logger.info("Loading 'base.en' Whisper model for transcription...")
            transcription_model = whisper.load_model("base.en")
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            # The app can continue without transcription, but we should log the error.
            transcription_model = None

def generate_transcript(audio_path_str: str) -> str:
    """
    Transcribes the given audio file and saves the transcript as a .vtt file.
    Returns the path to the generated .vtt file.
    """
    if transcription_model is None:
        logger.warning("Transcription model is not available. Cannot generate transcript.")
        return None

    try:
        audio_path = Path(audio_path_str)
        logger.info(f"Starting transcription for: {audio_path.name}")

        # The result object contains the transcript in various formats.
        result = transcription_model.transcribe(audio_path_str, verbose=False)

        # Define the output VTT path
        vtt_path = audio_path.with_suffix('.vtt')

        # Use whisper's built-in VTT writer
        from whisper.utils import get_writer
        writer = get_writer("vtt", str(vtt_path.parent))
        writer(result, str(audio_path.name))
        
        logger.info(f"Transcription complete. VTT file saved to: {vtt_path}")
        return str(vtt_path)

    except Exception as e:
        logger.error(f"An error occurred during transcription for {audio_path_str}: {e}", exc_info=True)
        return None
