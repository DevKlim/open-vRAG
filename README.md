# VideoChat-R1.5 Web Interface

This project provides a complete, runnable environment to interact with the `OpenGVLab/VideoChat-R1_5` model through a web interface. You can provide a video URL, and the backend will download it, sanitize it, and analyze it based on your question and configuration.

**This application requires an NVIDIA GPU to run.**

## Quick Start with Docker (Recommended)

1.  **Prerequisites:**
    *   [Docker](https://docs.docker.com/get-docker/)
    *   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support in Docker.
    *   An NVIDIA GPU with at least 16GB of VRAM is recommended.

2.  **Clone the project:**
    ```bash
    git clone <your-repo-url>
    cd vChat
    ```

3.  **Build and run the Docker container:**
    This command will build the Docker image and start the service in the background. The first build may take a while as it downloads the base image, Python packages, and the model from Hugging Face.
    ```bash
    docker-compose up --build -d
    ```

4.  **Access the Web UI:**
    Open your web browser and navigate to `http://localhost:8005`.

5.  **View Logs:**
    To see the application logs, including model loading progress and any errors:
    ```bash
    docker-compose logs -f
    ```

## Features

-   **Web UI for Inference:** Easily analyze videos by providing a URL and a question.
-   **Factuality Analysis:** Perform automated checks for visual artifacts, content credibility, and audio anomalies.
-   **Batch Processing:** Analyze multiple videos at once by uploading a CSV file.
-   **Fine-Tuning Integration:** Train the model on your own data using LoRA and switch between the base and fine-tuned model directly in the UI.
-   **Advanced Inference Configuration:** Control generation parameters like temperature, top-p, and the number of perception iterations.
-   **Robust Video Handling:** Videos are automatically downloaded (from URLs) or processed (from local paths) and re-encoded with `ffmpeg` to ensure compatibility.
-   **Local Model Caching:** Automatically uses a local copy of the model if you place it in the `VideoChat-R1` directory, saving download time on subsequent runs.

## Batch Processing

You can process a list of videos in a batch using the "Batch Processing from CSV" section in the web UI.

**1. Create your CSV file:**
The CSV file must contain a header row with a column named `url`. Each subsequent row should contain a single URL or a local file path.

**Example `batch_videos.csv`:**
```csv
url
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://www.youtube.com/watch?v=xyz...
videos/my_local_video.mp4
```

**2. Place Local Videos (if any):**
If your CSV includes local file paths, make sure those videos are placed inside the `vChat/videos` directory on your host machine. The Docker setup automatically maps this directory into the container. The path in the CSV should be relative to the container's app directory, e.g., `videos/my_video.mp4`.

**3. Run the Batch Job:**
-   Go to the web UI at `http://localhost:8005`.
-   Configure your desired analysis settings (question, factuality checks, advanced options). These settings will be applied to **all** videos in the batch.
-   In the "Batch Processing" section, upload your CSV file.
-   Click "Analyze Batch". The results will stream into the "Batch Processing Log".

## Fine-Tuning the Model with LoRA (Advanced)

This project includes `finetune.py` to adapt the model to your specific data.

**1. Prepare Your Dataset:**
Create a directory named `data` inside the `vChat` directory. Inside it, create a JSONL file (e.g., `my_dataset.jsonl`). Each line must be a JSON object with two keys:

*   `video_path`: The path to the video file **as it will be seen inside the container**. For example, if you place videos in the `vChat/videos` directory, the path should be `/app/videos/my_video_1.mp4`.
*   `text`: A single string representing a full conversational turn, including the roles.

**Example `my_dataset.jsonl`:**
```json
{"video_path": "/app/videos/sample1.mp4", "text": "USER: What is the person doing?\nASSISTANT: The person is writing on a whiteboard."}
{"video_path": "/app/videos/sample2.mp4", "text": "USER: Describe the main action.\nASSISTANT: A dog is catching a frisbee in a park."}
```

**2. Run the Fine-Tuning Script:**
Execute the script from inside the running Docker container. This will start training, which may take a significant amount of time.
```bash
docker exec -it videochat_webui python finetune.py
```
This script will quantize the model to 4-bit for memory efficiency and train LoRA adapters, saving them to `./lora_adapters/final_checkpoint`.

**3. Use the Fine-Tuned Model:**
Once training is complete, **restart the Docker container**:
```bash
docker-compose restart
```
On startup, the application will automatically detect the saved adapters. Now, when you visit `http://localhost:8005`, a "Select Model" dropdown will appear, allowing you to choose your "Custom Fine-tuned" model for inference.
