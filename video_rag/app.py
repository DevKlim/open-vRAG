import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from datetime import timedelta
import torch

from core.utils import get_job_directory, clean_filename
from core.video_processor import download_youtube_video, save_uploaded_file, extract_audio, extract_frames
from core.audio_processor import get_youtube_transcript, transcribe_audio_with_whisper, detect_audio_cuts
from core.youtube_scraper import get_most_replayed_timestamps
from core.gemini_analyzer import setup_gemini, get_frame_description
from core.logger import setup_logger

st.set_page_config(layout="wide")
st.title("Open-vRAG: Video Content Analyzer")
st.markdown("Analyze videos from a video editing perspective. Extract frames, audio, transcripts, and generate AI-powered analysis.")

# --- UI Sidebar for Inputs ---
with st.sidebar:
    st.header("System Status")
    if torch.cuda.is_available():
        st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("No GPU detected. Transcription will run on CPU and may be slow.")
    st.divider()
    
    st.header("1. Select Video Source")
    source_type = st.radio("Choose source:", ("YouTube URL", "Upload Local File"))

    video_path = None
    video_id = None
    video_duration = 0
    youtube_url = ""

    if source_type == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube URL:")
    else:
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

    st.header("2. Analysis Settings")
    frame_interval = st.slider(
        "Frame Extraction Frequency (seconds)", 
        min_value=1, max_value=60, value=10,
        help="Extract one frame every X seconds. More frames will be added automatically at key events."
    )

    process_button = st.button("Start Analysis")

# --- Main Application Logic ---
if process_button:
    job_dir = get_job_directory()
    st.success(f"Created a new job directory: `{job_dir}`")
    
    # --- Setup Logger ---
    logger = setup_logger(job_dir)
    logger.info("==================================================")
    logger.info("           STARTING NEW ANALYSIS JOB            ")
    logger.info("==================================================")

    # --- Step 1: Prepare Video File ---
    with st.spinner("Preparing video..."):
        if source_type == "YouTube URL" and youtube_url:
            logger.info(f"Source type: YouTube URL ({youtube_url})")
            video_path, video_id = download_youtube_video(youtube_url, job_dir, logger)
        elif source_type == "Upload Local File" and uploaded_file:
            logger.info(f"Source type: Local File ({uploaded_file.name})")
            video_path = save_uploaded_file(uploaded_file, job_dir, logger)
        else:
            st.warning("Please provide a video source.")
            logger.warning("Analysis started without a video source.")
            st.stop()
            
    if not video_path or not os.path.exists(video_path):
        st.error("Failed to load video file. Check the logs for details.")
        logger.error("Video processing stopped because video file could not be loaded.")
        st.stop()

    logger.info(f"Video file ready at: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    cap.release()
    st.info(f"Video duration: {timedelta(seconds=int(video_duration))}")
    logger.info(f"Video duration calculated: {video_duration} seconds.")

    # --- Step 2: Initialize Gemini ---
    gemini_model = setup_gemini(logger)
    if not gemini_model:
        logger.error("Gemini model setup failed. Stopping analysis.")
        st.stop()

    # --- Step 3: Process Audio and Transcripts ---
    all_data = []
    with st.spinner("Extracting audio and generating transcript..."):
        audio_path = extract_audio(video_path, job_dir, logger)
        transcript = None
        if video_id:
            transcript = get_youtube_transcript(video_id, logger)
            if transcript:
                st.info("Found and downloaded existing YouTube transcript.")
        
        if not transcript and audio_path:
            transcript = transcribe_audio_with_whisper(audio_path, logger)

        if transcript:
            for item in transcript:
                all_data.append({
                    "timestamp_seconds": item['timestamp'], "event_type": "transcript",
                    "data": item['text'], "source_file_path": audio_path or "YouTube API"
                })

    # --- Step 4: Advanced Event Detection ---
    with st.spinner("Detecting key events for frame extraction..."):
        timestamps_to_extract = set(np.arange(0, video_duration, frame_interval))
        logger.info(f"Generated {len(timestamps_to_extract)} timestamps from regular interval ({frame_interval}s).")
        
        if audio_path:
            audio_cut_timestamps = detect_audio_cuts(audio_path, logger)
            for ts in audio_cut_timestamps:
                timestamps_to_extract.add(round(ts))
                all_data.append({"timestamp_seconds": ts, "event_type": "audio_cut_event", "data": "Potential audio cut or sharp silence detected.", "source_file_path": audio_path})
            logger.info(f"Added {len(audio_cut_timestamps)} timestamps from audio cut detection.")

        if youtube_url:
            replay_timestamps = get_most_replayed_timestamps(youtube_url, video_duration, logger)
            for ts in replay_timestamps:
                timestamps_to_extract.add(round(ts))
                all_data.append({"timestamp_seconds": ts, "event_type": "replay_peak_event", "data": "High user engagement detected around this point.", "source_file_path": "YouTube Scraper"})
            logger.info(f"Added {len(replay_timestamps)} timestamps from 'most replayed' scraper.")

    st.success(f"Combined all sources. Total unique frames to extract: **{len(timestamps_to_extract)}**")
    
    # --- Step 5: Extract Frames and Analyze with Gemini ---
    frames_output_dir = os.path.join(job_dir, "frames")
    extracted_frames = extract_frames(video_path, list(timestamps_to_extract), frames_output_dir, logger)

    frame_analysis_progress = st.progress(0, text="Analyzing frames with Gemini...")
    for i, frame_info in enumerate(extracted_frames):
        description = get_frame_description(gemini_model, frame_info['path'], logger)
        all_data.append({"timestamp_seconds": frame_info['timestamp'], "event_type": "frame_analysis", "data": description, "source_file_path": frame_info['path']})
        frame_analysis_progress.progress((i + 1) / len(extracted_frames), text=f"Analyzing frame at {timedelta(seconds=int(frame_info['timestamp']))}...")
    frame_analysis_progress.empty()

    # --- Step 6: Finalize and Display Results ---
    st.header("Analysis Results")
    logger.info("Finalizing results and generating CSV.")

    if not all_data:
        st.warning("No data was generated from the analysis.")
        logger.warning("Pipeline finished but no data was generated.")
        st.stop()
        
    df = pd.DataFrame(all_data)
    df = df.sort_values(by="timestamp_seconds").reset_index(drop=True)
    
    csv_filename = f"{clean_filename(os.path.basename(video_path))}_analysis_log.csv"
    csv_path = os.path.join(job_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    st.success(f"All data saved to `{csv_path}`")
    logger.info(f"CSV report saved to {csv_path}")

    with open(csv_path, "rb") as f:
        st.download_button("Download Analysis CSV", f, file_name=csv_filename, mime="text/csv")

    st.dataframe(df)

    st.header("Keyframe Viewer")
    frame_analysis_df = df[df['event_type'] == 'frame_analysis'].copy()
    if not frame_analysis_df.empty:
        selected_timestamp = st.select_slider(
            "Select a timestamp to view its frame and analysis:",
            options=sorted(frame_analysis_df['timestamp_seconds'].tolist()),
            format_func=lambda x: str(timedelta(seconds=int(x)))
        )
        selected_row = frame_analysis_df[frame_analysis_df['timestamp_seconds'] == selected_timestamp].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.image(selected_row['source_file_path'], caption=f"Frame at {selected_timestamp:.2f}s", use_column_width=True)
        with col2:
            st.subheader("Gemini Vision Analysis")
            st.markdown(selected_row['data'])
    else:
        st.warning("No frames were analyzed to display in the viewer.")
    logger.info("Analysis job finished successfully.")
