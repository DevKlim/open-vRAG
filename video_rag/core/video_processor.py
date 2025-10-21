import os
import cv2
import yt_dlp
from .utils import seconds_to_filename_str, clean_filename
import streamlit as st
import ffmpeg

def download_youtube_video(url, output_path, logger):
    """Downloads the best MP4 stream from YouTube using yt-dlp for reliability."""
    safe_title = "youtube_video"
    video_id = None
    try:
        # First, get video info without downloading
        logger.info(f"Fetching metadata for YouTube URL: {url}")
        ydl_info_opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            safe_title = clean_filename(info.get('title', 'youtube_video'))
            video_id = info.get('id', None)
        
        video_filename = f"{safe_title}.mp4"
        video_filepath = os.path.join(output_path, video_filename)

        ydl_download_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_filepath,
            'quiet': True,
            'progress_hooks': [lambda d: logger.info(f"yt-dlp status: {d['status']}, {_bytes_to_str(d.get('downloaded_bytes', 0))} / {_bytes_to_str(d.get('total_bytes', 0))}") if d['status'] == 'downloading' else None]
        }
        
        logger.info(f"Starting download for '{safe_title}'...")
        st.write(f"Downloading '{safe_title}'...")
        with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
            ydl.download([url])
        
        logger.info(f"Successfully downloaded video to {video_filepath}")
        st.write("Download complete.")
        return video_filepath, video_id

    except Exception as e:
        error_message = f"Error downloading YouTube video: {e}"
        logger.exception(error_message) # Log the full traceback to the file
        st.error(error_message)
        st.info("This could be due to a private video, an age-restricted video, or a regional block. Check the log file for details.")
        return None, None

def _bytes_to_str(b):
    """Helper to format bytes into KB/MB/GB"""
    if b is None: return "N/A"
    if b < 1024: return f"{b} B"
    elif b < 1024**2: return f"{b/1024:.2f} KB"
    elif b < 1024**3: return f"{b/1024**2:.2f} MB"
    else: return f"{b/1024**3:.2f} GB"


def save_uploaded_file(uploaded_file, save_path, logger):
    """Saves an uploaded file to a specified path."""
    try:
        safe_filename = clean_filename(uploaded_file.name)
        filepath = os.path.join(save_path, f"{safe_filename}{os.path.splitext(uploaded_file.name)[-1]}")
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {filepath}")
        return filepath
    except Exception as e:
        logger.exception(f"Failed to save uploaded file {uploaded_file.name}")
        st.error(f"Error saving file: {e}")
        return None

def extract_audio(video_path, output_path, logger):
    """Extracts audio from a video file and saves it as WAV."""
    logger.info(f"Attempting to extract audio from {video_path}")
    try:
        audio_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.wav"
        audio_filepath = os.path.join(output_path, audio_filename)
        
        (
            ffmpeg
            .input(video_path)
            .output(audio_filepath, acodec='pcm_s16le', ac=1, ar='16000')
            .run(overwrite_output=True, quiet=True)
        )
        logger.info(f"Successfully extracted audio to {audio_filepath}")
        return audio_filepath
    except Exception as e:
        logger.exception(f"Error extracting audio with ffmpeg from {video_path}")
        st.error(f"Error extracting audio with ffmpeg: {e}")
        st.info("Ensure ffmpeg is correctly installed and accessible in your system's PATH or Docker container.")
        return None

def extract_frames(video_path, timestamps, output_dir, logger):
    """Extracts frames from a video at a list of specific timestamps."""
    logger.info(f"Starting frame extraction from {video_path} for {len(timestamps)} timestamps.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"cv2.VideoCapture failed to open {video_path}")
        st.error("Error: Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.error(f"Video FPS is zero for {video_path}, cannot extract frames.")
        st.error("Video FPS is zero, cannot extract frames.")
        return []
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    logger.info(f"Video properties: FPS={fps}, Frame Count={frame_count}, Duration={duration}s")
    
    extracted_frames = []
    sorted_timestamps = sorted(list(set(timestamps)))

    progress_bar = st.progress(0, text="Extracting frames...")
    
    for i, ts in enumerate(sorted_timestamps):
        if ts > duration or ts < 0:
            logger.warning(f"Skipping timestamp {ts}s as it's outside the video duration ({duration}s).")
            continue
            
        frame_id = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        ret, frame = cap.read()
        if ret:
            frame_filename = f"frame_at_{seconds_to_filename_str(ts)}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append({"timestamp": ts, "path": frame_path})
        else:
            logger.warning(f"Failed to retrieve frame at timestamp {ts}s (frame ID {frame_id}).")
        
        progress_bar.progress((i + 1) / len(sorted_timestamps), text=f"Extracting frame at {ts:.2f}s...")

    cap.release()
    progress_bar.empty()
    logger.info(f"Successfully extracted {len(extracted_frames)} frames.")
    return extracted_frames
