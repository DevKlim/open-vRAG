# open-vRAG/video_rag/README.md
# Video RAG Content Analyzer

## Overview

This module provides a web-based tool for conducting a deep analysis of video content from a video editor's perspective. It ingests a video from a local file or a YouTube URL, processes its audio and visual components, and generates a detailed, timestamped log of events and descriptions. The analysis is powered by state-of-the-art models, including Gemini for visual description and Whisper for transcription.

The primary goal is to create a rich, structured dataset from a video that can be used for Retrieval-Augmented Generation (RAG), content strategy, or detailed editing analysis.

## Features

-   **Multi-Source Input**: Analyze videos by uploading a local file or pasting a YouTube URL.
-   **GPU Accelerated**: Automatically utilizes an NVIDIA GPU if available to significantly speed up audio transcription.
-   **Configurable Frame Extraction**: Users can set the frequency for regular frame extraction (e.g., one frame every 10 seconds).
-   **Intelligent Frame Triggering**: In addition to regular intervals, frames are automatically extracted at key moments, such as:
    -   Potential audio cuts or edits detected in the soundtrack.
    -   "Most replayed" sections of a YouTube video, indicating high audience engagement.
-   **Audio and Transcript Processing**:
    -   Automatically fetches existing transcripts for YouTube videos.
    -   If no transcript is available, it transcribes the audio using a speech-to-text model (Whisper).
    -   All transcript segments are timestamped.
-   **Gemini-Powered Frame Analysis**: Each extracted frame is analyzed by the Gemini model to generate a description focusing on video editing principles like composition, lighting, subject, and action.
-   **Comprehensive Data Export**: All collected data points (transcripts, frame analyses, audio events) are compiled into a single, sorted CSV file with precise timestamps, ready for further analysis or use in a RAG system.

## Setup and Installation

### Prerequisites

-   Docker and Docker Compose
-   A Google API key with access to the Gemini API.
-   **For GPU Acceleration**:
    -   An NVIDIA GPU.
    -   The latest NVIDIA drivers for your operating system.
    -   The NVIDIA Container Toolkit installed on your system to allow Docker to access the GPU.

### Local Setup

1.  **Clone the Repository**:
    If you have not already, clone the `open-vRAG` repository to your local machine.

2.  **Environment Variables**:
    Create a `.env` file in the `open-vRAG/video_rag` directory. This file will store your Google API key. Add the following line to it, replacing `your_google_api_key_here` with your actual key:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

3.  **Build and Run with Docker Compose**:
    Navigate to the `open-vRAG/video_rag` directory in your terminal and run the following commands:

    ```bash
    # Build the Docker image (this will take longer the first time)
    docker-compose build

    # Start the application in detached mode
    docker-compose up -d
    ```
    The application will automatically detect and use your NVIDIA GPU if the prerequisites are met.

4.  **Access the Web UI**:
    Open your web browser and navigate to `http://localhost:8501`. The application should now be running. The sidebar will indicate whether a GPU was successfully detected.

## How to Use the Application

1.  **Check System Status**:
    The sidebar will show a "System Status" section, confirming if a GPU has been detected for accelerated processing.

2.  **Select Video Source**:
    Choose whether you want to analyze a video from a "YouTube URL" or by "Upload Local File".

3.  **Provide the Video**:
    -   If you chose YouTube, paste the full URL into the text input box.
    -   If you chose to upload, use the file uploader to select a video from your computer.

4.  **Configure Analysis Settings**:
    Use the slider to set the "Frame Extraction Frequency". This determines how often a frame is captured for analysis (e.g., a value of 5 means one frame will be extracted every 5 seconds).

5.  **Start Analysis**:
    Click the "Start Analysis" button. The application will begin processing the video. You can monitor its progress through the status messages and progress bars.

6.  **Review Results**:
    Once the analysis is complete, the results will be displayed in a table.
    -   A download button will allow you to save the complete analysis as a CSV file.
    -   The "Keyframe Viewer" lets you select any analyzed frame and view the image alongside its Gemini analysis.

7.  **Stopping the Application**:
    To stop the application, return to your terminal and run:
    ```bash
    docker-compose down
    ```