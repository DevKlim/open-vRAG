Analysis of Logic in `src` Folder
=================================

Overview
--------
The system is a **Video Factuality Analysis Engine** built with FastAPI. It uses a Retrieval-Augmented Generation (RAG) approach (specifically "Fractal Chain of Thought") to analyze videos for misinformation, deepfakes, and logical inconsistencies.

Key Components
--------------

1. app.py (The Controller)
   - **Role:** Main entry point. Handles HTTP requests, static files, and queue management.
   - **Key Features:**
     - **Video Ingestion:** Downloads videos using `yt_dlp` and processes them with `ffmpeg`.
     - **Queue System:** Manages a CSV-based queue (`data/batch_queue.csv`) for batch processing.
     - **Endpoints:** `/queue/run` triggers the analysis pipeline; `/extension/ingest` allows adding videos from a browser extension.
     - **Storage:** Saves results as JSON, TOON (raw text), and updates CSV datasets.

2. inference_logic.py (The Brain)
   - **Role:** Orchestrates the AI analysis.
   - **Key Features:**
     - **Multi-Model Support:** Supports Google Gemini (Legacy & Modern SDKs), Vertex AI, and local models (Qwen3VL).
     - **Pipelines:**
       - **Standard CoT:** Single-turn Chain of Thought.
       - **Fractal CoT (FCoT):** A multi-step recursive process (Macro Hypothesis -> Meso Expansion -> Synthesis) for deeper analysis.
     - **Auto-Repair:** Includes a mechanism (`attempt_toon_repair`) to fix malformed AI outputs using a second AI call.

3. labeling_logic.py (The Instructions)
   - **Role:** Defines the "Prompt Engineering" layer.
   - **Key Features:**
     - **Prompts:** Contains the specific instructions for the AI (e.g., `FCOT_MACRO_PROMPT`, `LABELING_PROMPT_TEMPLATE`).
     - **Schemas:** Defines the **TOON** (Token-Oriented Object Notation) format used for structured output, which is designed to be more token-efficient than JSON.

4. toon_parser.py (The Translator)
   - **Role:** Parses the AI's raw text output into structured Python dictionaries.
   - **Key Features:**
     - **Robust Parsing:** Uses Regex and CSV parsing to handle the custom TOON format.
     - **Fuzzy Fallback:** Can extract scores even if the strict format is broken.

5. factuality_logic.py (Legacy/Alternative)
   - **Role:** Appears to be a separate or legacy pipeline for specific checks (Visual, Audio, Content) using local models.
   - **Status:** It is imported by `app.py` but its main pipeline function `run_factuality_pipeline` is **not currently used** in the main application flow. Only `parse_vtt` is utilized.

Data Flow
---------
1. **User/Extension** adds a link to the Queue.
2. **`app.py`** downloads the video and extracts audio/transcript.
3. **`app.py`** calls `inference_logic.run_gemini_labeling_pipeline` (or Vertex).
4. **`inference_logic`** sends the video + prompts (`labeling_logic`) to the LLM.
5. **LLM** "thinks" (Chain of Thought) and outputs a TOON-formatted response.
6. **`inference_logic`** uses `toon_parser` to convert TOON to JSON.
7. **`app.py`** saves the JSON, updates the CSV dataset, and generates metadata.
