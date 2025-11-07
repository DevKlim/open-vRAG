# VideoChat-R1.5 Web Interface

This project provides a complete, runnable environment to interact with our fine-tuned model. Our model is built based off of `OpenGVLab/VideoChat-R1_5` model through a web interface. Ultimately, link model with `gemini-flash-latest` for optimal capabilities with video digestion. You can provide a video URL, and the backend will download it, sanitize it, and analyze it based on your question and configuration.

**This application requires an NVIDIA GPU to run the default local model.**

## Quick Start with Docker (Recommended)

1.  **Prerequisites:**
    *   [Docker](https://docs.docker.com/get-docker/)
    *   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support in Docker.
    *   An NVIDIA GPU with at least 16GB of VRAM is recommended for the local model.

2.  **Clone the project:**
    ```bash
    git clone <your-repo-url>
    cd vChat
    ```

3.  **(Optional but Recommended) Create Cache Directories:**
    To prevent re-downloading the large models every time you start the container, create the cache directory now:
    ```bash
    mkdir huggingface_cache
    ```

4.  **Build and run the Docker container:**
    This command will build the Docker image and start the service in the background. The first build may take a while as it downloads the base image and Python packages.
    ```bash
    docker-compose up --build -d
    ```

5.  **Access the Web UI:**
    Open your web browser and navigate to `http://localhost:8005`.

6.  **View Logs:**
    To see the application logs, including model loading progress and any errors:
    ```bash
    docker-compose logs -f
    ```

## Performance and Startup Time

The main cause of slow startup is the need to download and load the multi-gigabyte machine learning models. This project is configured to cache these models on your host machine to dramatically speed up subsequent startups.

-   **Hugging Face Cache:** The `docker-compose.yml` file maps the local `./huggingface_cache` directory into the container. When the application downloads the `VideoChat-R1_5` model for the first time, it will be saved here. On all future runs, it will load the model from this cache instead of re-downloading it.
-   **Pre-seeding the Model (Fastest Startup):** For the fastest possible first-time startup, you can download the model manually.
    1.  Run `git lfs install && git clone https://huggingface.co/OpenGVLab/VideoChat-R1_5` on your host machine.
    2.  Move the downloaded `VideoChat-R1_5` folder into this project's `vChat` directory.
    3.  Rename it to `VideoChat-R1`.
    The `docker-compose.yml` is already configured to detect this folder and use it, completely skipping any downloads.

## Features

-   **Multiple Model Support:** Analyze videos using the default `VideoChat-R1_5` model, a custom fine-tuned version, or Google's Gemini Pro Vision.
-   **Web UI for Inference:** Easily analyze videos by providing a URL and a question.
-   **Factuality Analysis:** Perform automated checks for visual artifacts, content credibility, and audio anomalies.
-   **Batch Processing:** Analyze multiple videos at once by uploading a CSV file.
-   **Fine-Tuning Integration:** Train the model on your own data using LoRA and switch between the base and fine-tuned model directly in the UI.
-   **Robust Video Handling:** Videos are automatically downloaded, sanitized, and re-encoded to ensure compatibility.
-   **Local Model Caching:** Drastically reduces startup times by caching models on your host machine.

## Using Gemini Models

You can choose to use a Google Gemini model instead of the local one. This offloads the computation to Google's servers and does not require a local GPU.

1.  **Get a Gemini API Key:** Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  **Select in UI:** In the web interface, change the "Select Model" dropdown to "Gemini Pro Vision".
3.  **Provide Credentials:**
    *   **In the UI:** The Gemini API Key and Model Name fields will appear. Enter your key there.
    *   **Via URL:** You can also provide credentials in the URL for easy bookmarking or sharing. The application will read them automatically on page load:
        `http://localhost:8005/?gemini_api_key=YOUR_API_KEY_HERE&gemini_model_name=models/gemini-1.5-pro-latest`

## Batch Processing

You can process a list of videos in a batch using the "Batch Processing from CSV" section in the web UI.

**1. Create your CSV file:**
The CSV file must contain a header row with a column named `url`.

**Example `batch_videos.csv`:**
```csv
url
https://www.youtube.com/watch?v=dQw4w9WgXcQ
videos/my_local_video.mp4
```

**2. Run the Batch Job:**
-   Go to the web UI, select your desired model (including Gemini) and configure all settings. These settings will be applied to **all** videos in the batch.
-   In the "Batch Processing" section, upload your CSV file and click "Analyze Batch".

## Fine-Tuning the Model with LoRA (Advanced)

This project includes `finetune.py` to adapt the base `VideoChat-R1_5` model to your specific data.

**1. Prepare Your Dataset:**
Create a directory named `data` inside the `vChat` directory. Inside it, create a JSONL file (e.g., `my_dataset.jsonl`). Each line must be a JSON object with two keys: `video_path` and `text`.

**Example `my_dataset.jsonl`:**
```json
{"video_path": "/app/videos/sample1.mp4", "text": "USER: What is the person doing?\nASSISTANT: The person is writing on a whiteboard."}
{"video_path": "/app/videos/sample2.mp4", "text": "USER: Describe the main action.\nASSISTANT: A dog is catching a frisbee in a park."}
```

**2. Run the Fine-Tuning Script:**
Execute the script from inside the running Docker container.
```bash
docker exec -it videochat_webui python finetune.py
```
This script will train LoRA adapters and save them to `./lora_adapters/final_checkpoint`.

**3. Use the Fine-Tuned Model:**
Once training is complete, **restart the Docker container**:
```bash
docker-compose restart
```
On startup, the application will automatically detect the saved adapters. Now, when you visit `http://localhost:8000`, a "Select Model" dropdown will appear, allowing you to choose your "Custom Fine-tuned" model for inference.

Citation:
```
@article{li2025videochatr1,
  title={VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning},
  author={Li, Xinhao and Yan, Ziang and Meng, Desen and Dong, Lu and Zeng, Xiangyu and He, Yinan and Wang, Yali and Qiao, Yu and Wang, Yi and Wang, Limin},
  journal={arXiv preprint arXiv:2504.06958},
  year={2025}
}

@article{yan2025videochatr15,
  title={VideoChat-R1.5: Visual Test-Time Scaling to Reinforce Multimodal Reasoning by Iterative Perception},
  author={Yan, Ziang and Li, Xinhao and He, Yinan and Zhengrong Yue and Zeng, Xiangyu and Wang, Yali and Qiao, Yu and Wang, Limin and Wang, Yi},
  journal={arXiv preprint arXiv:2509.21100},
  year={2025}
}
```
