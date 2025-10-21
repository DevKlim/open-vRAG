import os
import torch
import whisper
from pydub import AudioSegment
from pydub.silence import detect_silence
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from multiprocessing import Process, Queue

# Check for GPU and set device globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _transcribe_worker(audio_path, device, queue):
    """
    This function runs in a separate process to isolate Whisper model loading
    and transcription, which can sometimes conflict with Streamlit's main thread
    or resource management.
    """
    try:
        # Load model inside the worker process
        model = whisper.load_model("base", device=device)
        result = model.transcribe(audio_path, word_timestamps=True)
        
        transcript = []
        for segment in result['segments']:
            transcript.append({
                "timestamp": segment['start'],
                "text": segment['text'].strip()
            })
        queue.put(transcript)
    except Exception as e:
        # Pass the exception back to the main process
        queue.put(e)

def transcribe_audio_with_whisper(audio_path, logger):
    """
    Transcribes audio using OpenAI's Whisper model by spawning an isolated
    process to run the transcription, ensuring better stability with Streamlit.
    """
    logger.info(f"Starting transcription for {audio_path} in a separate process on {DEVICE.upper()}.")
    st.write(f"Transcribing audio with Whisper on {DEVICE.upper()}... (This may take a while for long videos)")

    q = Queue()
    p = Process(target=_transcribe_worker, args=(audio_path, DEVICE, q))
    
    try:
        p.start()
        p.join()  # Wait for the process to complete
        
        result = q.get()
        if isinstance(result, Exception):
            # If an exception was put in the queue, re-raise it in the main process
            raise result
        
        logger.info(f"Transcription complete. Generated {len(result)} segments.")
        return result
    except Exception as e:
        logger.exception(f"Error during transcription with Whisper for {audio_path}.")
        st.error(f"Error during transcription with Whisper: {e}")
        return None
    finally:
        # Ensure the process is terminated if it's still alive (e.g., due to a timeout or unhandled error)
        if p.is_alive():
            p.terminate()

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
