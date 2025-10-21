import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from datetime import timedelta
import torch

from core.utils import get_job_directory, clean_filename
from core.video_processor import download_youtube_video, save_uploaded_file, extract_audio, extract_frames
from core.audio_processor import transcribe_audio_with_whisper, get_youtube_transcript, detect_audio_cuts
from core.youtube_scraper import get_most_replayed_timestamps
from core.gemini_analyzer import setup_gemini, get_frame_description
from core.logger import setup_logger
from core.rag_builder import build_and_save_vector_store, create_rag_chain

st.set_page_config(layout="wide", page_title="Open-vRAG: Video Analyzer")
st.title("Open-vRAG: Video Content Analyzer & RAG Pipeline")

# --- Initialize Session State ---
if 'job_dir' not in st.session_state:
    st.session_state.job_dir = None
    st.session_state.logger = None
    st.session_state.job_status = "stopped" # stopped, running, cancelled
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")
    if torch.cuda.is_available(): st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else: st.warning("No GPU detected. Processing will be slower.")
    st.divider()

    if st.session_state.job_dir:
        st.success(f"Active Job: `{os.path.basename(st.session_state.job_dir)}`")
        if st.session_state.job_status == 'running':
            if st.button("Cancel Job", type="primary"):
                st.session_state.job_status = "cancelled"
                st.warning("Cancellation requested. The job will stop shortly.")
        
        if st.button("Start New Job"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    else:
        st.info("No active job. Start by providing a video in Tab 1.")

# --- Main App with Tabs ---
tab1, tab2, tab3 = st.tabs(["1. Data Extraction & Analysis", "2. Build RAG Vector Store", "3. Chat with Video"])

with tab1:
    st.header("Step 1: Process a Video")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Source")
        source_type = st.radio("Choose source:", ("YouTube URL", "Upload Local File"), key="source_type")
        youtube_url = ""; uploaded_file = None; cookie_file = None
        if source_type == "YouTube URL":
            youtube_url = st.text_input("Enter YouTube URL:")
            quality_options = {"Best (Compatible H.264)": 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best', "Highest Quality (May be AV1)": 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'}
            selected_quality = st.selectbox("Select Video Quality", options=list(quality_options.keys()), help="Choose 'Compatible' if frame extraction fails. AV1 codecs are not always supported.")
            quality_format = quality_options[selected_quality]
            st.subheader("YouTube Settings (Optional)")
            cookie_file = st.file_uploader("Upload cookies.txt", type=['txt'], help="For rate-limited or age-restricted videos, provide a cookies.txt file from your browser.")
        else:
            uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

    with col2:
        st.subheader("Analysis Settings")
        gemini_model_name = st.text_input("Gemini Model Name", value="gemini-1.5-pro-latest", help="Specify the Gemini model to use for frame analysis.")
        frame_interval = st.slider("Base Frame Extraction Frequency (seconds)", 1, 60, 10, help="Extract one frame every X seconds.")
        enable_replay_scraper = st.checkbox("Extract 'Most Replayed' timestamps (YouTube only)", value=True)
        enable_audio_cuts = st.checkbox("Extract timestamps at audio cuts/silences", value=True)

    if st.button("Start Analysis", type="primary", disabled=(st.session_state.job_status == 'running')):
        # --- Job Initialization ---
        if (source_type == "YouTube URL" and not youtube_url) or \
           (source_type == "Upload Local File" and not uploaded_file):
            st.warning("Please provide a video source to start analysis.")
        else:
            st.session_state.job_dir = get_job_directory()
            st.session_state.logger = setup_logger(st.session_state.job_dir)
            st.session_state.job_status = "running"
            st.rerun() # Rerun to show the cancel button and start the analysis logic

    if st.session_state.job_status == "running":
        logger = st.session_state.logger
        st.success(f"Running job: `{os.path.basename(st.session_state.job_dir)}`")
        logger.info(f"--- STARTING JOB: {os.path.basename(st.session_state.job_dir)} ---")

        with st.status("Analyzing video...", expanded=True) as status:
            cookie_path = None
            if source_type == "YouTube URL" and cookie_file:
                cookie_path = os.path.join(st.session_state.job_dir, cookie_file.name)
                with open(cookie_path, "wb") as f: f.write(cookie_file.getvalue())
                logger.info(f"Saved cookie file to {cookie_path}")
            
            status.update(label="Step 1/5: Loading video...")
            video_path = None
            video_id = None
            if source_type == "YouTube URL" and youtube_url:
                video_path, video_id = download_youtube_video(youtube_url, st.session_state.job_dir, logger, quality_format, cookie_path)
            elif source_type == "Upload Local File" and uploaded_file:
                video_path, video_id = save_uploaded_file(uploaded_file, st.session_state.job_dir, logger), None
            
            if not video_path or not os.path.exists(video_path):
                st.error("Failed to load video file. Check logs for details."); st.session_state.job_status = "stopped"; st.stop()

            cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); video_duration = frame_count/fps; cap.release()
            st.info(f"Video duration: {timedelta(seconds=int(video_duration))}")
            gemini_model = setup_gemini(logger, model_name=gemini_model_name)
            if not gemini_model: st.error("Failed to setup Gemini model. Check logs for details."); st.session_state.job_status = "stopped"; st.stop()
            
            status.update(label="Step 2/5: Processing audio and transcript...")
            all_data = []
            audio_path = extract_audio(video_path, st.session_state.job_dir, logger)
            transcript = get_youtube_transcript(video_id, logger) if video_id else None
            if not transcript and audio_path: transcript = transcribe_audio_with_whisper(audio_path, logger)
            if transcript: all_data.extend([{"timestamp_seconds": item['timestamp'], "event_type": "transcript", "data": item['text'], "source_file_path": audio_path or "YouTube API"} for item in transcript])

            status.update(label="Step 3/5: Detecting key events...")
            timestamps_to_extract = set(np.arange(0, video_duration, frame_interval))
            if enable_audio_cuts and audio_path: timestamps_to_extract.update([round(ts) for ts in detect_audio_cuts(audio_path, logger)])
            if enable_replay_scraper and youtube_url: timestamps_to_extract.update([round(ts) for ts in get_most_replayed_timestamps(youtube_url, video_duration, logger)])
            st.success(f"Identified {len(timestamps_to_extract)} unique timestamps for frame extraction.")

            status.update(label=f"Step 4/5: Extracting and analyzing {len(timestamps_to_extract)} frames...")
            frames_dir = os.path.join(st.session_state.job_dir, "frames")
            extracted_frames = extract_frames(video_path, list(timestamps_to_extract), frames_dir, logger)
            
            for i, frame_info in enumerate(extracted_frames):
                if st.session_state.job_status == "cancelled":
                    st.warning("Job cancelled by user."); break
                status.update(label=f"Step 4/5: Analyzing frame {i+1}/{len(extracted_frames)} with Gemini...")
                description = get_frame_description(gemini_model, frame_info['path'], logger)
                all_data.append({"timestamp_seconds": frame_info['timestamp'], "event_type": "frame_analysis", "data": description, "source_file_path": frame_info['path']})

            status.update(label="Step 5/5: Finalizing results...")
            if all_data:
                df = pd.DataFrame(all_data).sort_values(by="timestamp_seconds").reset_index(drop=True)
                csv_filename = f"{clean_filename(os.path.basename(video_path))}_analysis_log.csv"
                csv_path = os.path.join(st.session_state.job_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                st.success(f"Analysis saved to `{csv_path}`. You can now proceed to Tab 2.")
                st.dataframe(df)
            else: st.warning("Analysis finished but no data was generated.")
        
        st.session_state.job_status = "stopped"
        st.rerun()

with tab2:
    st.header("Step 2: Create a Queryable RAG Index")
    if not st.session_state.job_dir:
        st.info("Please complete Step 1 first to process a video.")
    else:
        st.success(f"Ready to process job: `{os.path.basename(st.session_state.job_dir)}`")
        if st.button("Build Vector Store", type="primary"):
            with st.spinner("Building vector store... This may take a few moments."):
                vector_store_path = build_and_save_vector_store(st.session_state.job_dir, st.session_state.logger)
                if vector_store_path:
                    st.success(f"Vector store built successfully! You can now chat with the video in Tab 3.")
                    st.session_state.vector_store_path = vector_store_path
                else:
                    st.error("Failed to build vector store. Check logs for details.")

with tab3:
    st.header("Step 3: Chat with Your Video's Content")
    if not st.session_state.job_dir or not os.path.exists(os.path.join(st.session_state.job_dir, "vectorstore")):
        st.info("Please complete Steps 1 and 2 first.")
    else:
        if 'rag_chain' not in st.session_state or st.session_state.rag_chain is None:
            with st.spinner("Initializing chat..."):
                gemini_model = setup_gemini(st.session_state.logger)
                if gemini_model:
                    st.session_state.rag_chain = create_rag_chain(gemini_model, st.session_state.job_dir, st.session_state.logger)
        
        if st.session_state.rag_chain:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about the video content..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain({"query": prompt})
                    answer = response['result']
                    with st.expander("View Sources"):
                        for doc in response['source_documents']:
                           st.info(f"Timestamp: {doc.metadata.get('timestamp_seconds', 'N/A')}\n\nContent: {doc.page_content}")

                with st.chat_message("assistant"): st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("Could not initialize the chat interface.")
