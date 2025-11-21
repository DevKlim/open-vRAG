# VideoChat-R1.5 Web Interface

This project provides a complete, runnable environment to interact with our fine-tuned model based on `OpenGVLab/VideoChat-R1_5`.

## Quick Start with Docker

1.  **Build and run:**
    ```bash
    docker-compose up --build -d
    ```

2.  **Access the UI:**
    http://localhost:8005

## Project Structure

-   `src/`: Python backend and model logic.
-   `frontend/`: React/Vite web interface.
-   `data/`: Storage for datasets, labels, and videos.
-   `model/`: Place model weights here (mapped to container).

## Fine-Tuning the Model

To run fine-tuning, execute the script inside the container:

```bash
docker exec -it videochat_webui python src/finetune.py
```

Ensure your dataset is located at `data/insertlocaldataset.jsonl`.
