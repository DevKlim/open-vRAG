# Video RAG Content Analyzer

## Overview

This module provides a web-based tool for conducting a deep analysis of video content from a video editor's perspective. It ingests a video from a local file or a YouTube URL, processes its audio and visual components, and generates a detailed, timestamped log of events and descriptions. The analysis is powered by state-of-the-art models, including Gemini for visual description and Whisper for transcription.

The primary goal is to create a rich, structured dataset from a video that can be used for Retrieval-Augmented Generation (RAG), allowing users to "chat" with the video's content to get detailed, context-aware answers.

## What the tooling does!!!

-   **Multi-Source Input**: Analyze videos by uploading a local file or pasting a YouTube URL.
-   **GPU Accelerated**: Automatically utilizes an NVIDIA GPU if available to significantly speed up audio transcription.
-   **Configurable Frame Extraction**: Users can set the frequency for regular frame extraction (for instance, one frame every 10 seconds).
-   **Intelligent Frame Triggering**: In addition to regular intervals, frames are automatically extracted at key moments, such as:
    -   Potential audio cuts or edits detected in the soundtrack.
    -   "Most replayed" sections of a YouTube video, indicating high audience engagement.
-   **Comprehensive Audio/Transcript Processing**:
    -   Automatically fetches existing transcripts for YouTube videos.
    -   If no transcript is available, it transcribes the audio using OpenAI's Whisper model.
    -   All transcript segments are timestamped.
-   **Gemini-Powered Frame Analysis**: Each extracted frame is analyzed by the Gemini 1.5 Pro model to generate a description focusing on video editing principles like composition, lighting, subject, and action.
-   **End-to-End RAG Pipeline**:
    -   All collected data points (transcripts, frame analyses, audio events) are compiled into a single, sorted CSV file.
    -   A vector store is built from this data, creating a queryable index.
    -   A "Chat with Video" interface allows users to ask questions and receive answers based on the video's content.

## Setup and Installation

### Prerequisites

-   Docker and Docker Compose (or you can run it locally with installed packages)
-   A Google API key with access to the Gemini API. (place into a .env file with GOOGLE_API_KEY=...)

-   **For GPU Acceleration**:
    -   An NVIDIA GPU.
    -   The latest NVIDIA drivers for your operating system.
    -   The NVIDIA Container Toolkit installed on your system to allow Docker to access the GPU.

### Local Setup

1.  **Clone the Repository**:
    If you have not already, clone the `open-vRAG` repository to your local machine.

2.  **Environment Variables**:
    Create a `.env` file in the `open-vRAG/video_rag` directory. This file will store your Google API key. Add the following line to it, replacing `your_google_api_key_here` with your actual key:
