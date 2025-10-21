import os
import torch
import whisper
from pydub import AudioSegment
from pydub.silence import detect_silence
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st

# Check for GPU and set device globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load whisper model globally to avoid reloading
@st.cache_resource
def load_whisper_model(logger):
    """Loads the Whisper model onto the appropriate device (GPU or CPU)."""
    logger.info("Loading Whisper model...")
    try:
        model = whisper.load_model("base", device=DEVICE)
        logger.info(f"Whisper model 'base' loaded successfully onto {DEVICE.upper()}.")
        return model
    except Exception as e:
        logger.exception("Failed to load Whisper model.")
        st.error(f"Failed to load Whisper model: {e}")
        st.info("This might be due to a network issue or an incorrect PyTorch/CUDA setup.")
        return None

def get_youtube_transcript(video_id, logger):
    """Fetches the transcript for a YouTube video if available."""
    logger.info(f"Searching for existing YouTube transcript for video ID: {video_id}")
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        formatted_transcript = []
        for item in transcript_list:
            formatted_transcript.append({
                "timestamp": item['start'],
                "text": item['text']
            })
        logger.info(f"Found and formatted transcript with {len(formatted_transcript)} segments.")
        return formatted_transcript
    except Exception as e:
        logger.warning(f"Could not retrieve YouTube transcript for video ID {video_id}. It may not exist. Error: {e}")
        return None

def transcribe_audio_with_whisper(audio_path, logger):
    """Transcribes audio using OpenAI's Whisper model on GPU if available."""
    try:
        model = load_whisper_model(logger)
        if not model:
            logger.error("Whisper model is not loaded, cannot transcribe.")
            return None
            
        logger.info(f"Starting transcription for {audio_path} on {DEVICE.upper()}.")
        st.write(f"Transcribing audio with Whisper on {DEVICE.upper()}... (This may take a while for long videos)")
        result = model.transcribe(audio_path, word_timestamps=True)
        
        transcript = []
        for segment in result['segments']:
            transcript.append({
                "timestamp": segment['start'],
                "text": segment['text'].strip()
            })
        
        logger.info(f"Transcription complete. Generated {len(transcript)} segments.")
        return transcript
    except Exception as e:
        logger.exception(f"Error during transcription with Whisper for {audio_path}.")
        st.error(f"Error during transcription with Whisper: {e}")
        return None

def detect_audio_cuts(audio_path, logger, min_silence_len=300, silence_thresh=-40):
    """
    Detects potential audio cuts by looking for short, sharp silences.
    """
    logger.info(f"Inspecting {audio_path} for audio cuts...")
    st.write("Inspecting audio for cuts...")
    try:
        audio = AudioSegment.from_file(audio_path)
        silences = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        cut_timestamps = [start / 1000.0 for start, end in silences]
        logger.info(f"Detected {len(cut_timestamps)} potential audio events/cuts.")
        st.write(f"Detected {len(cut_timestamps)} potential audio events/cuts.")
        return cut_timestamps
    except Exception as e:
        logger.exception(f"Could not process audio for cut detection on {audio_path}.")
        st.error(f"Could not process audio for cut detection: {e}")
        return []
