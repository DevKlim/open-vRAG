import os
import sys
import asyncio
import subprocess
from pathlib import Path
import logging
import csv
import io
import datetime
import json
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yt_dlp
import inference_logic
import factuality_logic
import transcription
from factuality_logic import parse_vtt

import custom_clickbait_model

# --- CroissantML Imports with error handling ---
try:
    from croissant import nodes as cnodes
    from croissant import Metadata
    from croissant.data_types import DataType
    CROISSANT_AVAILABLE = True
except ImportError:
    cnodes = None
    Metadata = None
    DataType = None
    CROISSANT_AVAILABLE = False
# 

#  config application-wide logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# 

#  fastAPI app setup 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("videos", exist_ok=True)
os.makedirs("data", exist_ok=True) # Ensure the data directory for the CSV exists
os.makedirs("data/labels", exist_ok=True) # Ensure the directory for JSON labels exists
os.makedirs("metadata", exist_ok=True) # Ensure the metadata directory exists


@app.on_event("startup")
async def startup_event():
    """Load all models on application startup."""
    logging.info("Application starting up...")
    try:
        inference_logic.load_models()
        transcription.load_model()
    except Exception as e:
        logging.fatal(f"Could not load models. Error: {e}", exc_info=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page, indicating if a custom model is available."""
    custom_model_available = inference_logic.peft_model is not None
    return templates.TemplateResponse("index.html", {
        "request": request,
        "custom_model_available": custom_model_available
    })

@app.get("/model-architecture", response_class=PlainTextResponse)
async def get_model_architecture():
    """Returns the base model architecture as a plain text string."""
    if inference_logic.base_model:
        return str(inference_logic.base_model)
    return "Base model not loaded yet. Please wait a moment and try again."

progress_message = ""
def progress_hook(d):
    global progress_message
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A').strip()
        speed = d.get('_speed_str', 'N/A').strip()
        eta = d.get('_eta_str', 'N/A').strip()
        progress_message = f"Downloading: {percent} at {speed}, ETA: {eta}\r"
    elif d['status'] == 'finished':
        progress_message = f"\nDownload finished. Preparing video assets...\n"

async def run_subprocess_async(command: list[str]):
    """Asynchronously runs a subprocess and captures its output."""
    process = await asyncio.create_subprocess_exec(
        *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        error_details = stderr.decode()
        logging.error(f"FFmpeg command failed: {command}")
        logging.error(f"FFmpeg stderr: {error_details}")
        raise RuntimeError(f"Process failed:\n{error_details}")
    return stdout.decode()

async def prepare_video_assets_async(url: str) -> dict:
    """
    Downloads video (if URL) or finds local video, then prepares assets including transcription.
    Returns a dictionary of file paths and extracted metadata using STANDARDIZED keys.
    """
    global progress_message
    loop = asyncio.get_event_loop()
    original_path = None
    transcript_path = None
    metadata = {}
    audio_path_str = None

    is_local_file = not (url.startswith("http://") or url.startswith("https://"))

    if is_local_file:
        progress_message = f"Processing local file: {url}\n"
        original_path = Path(url)
        if not original_path.exists():
            raise FileNotFoundError(f"Local video file not found at path: {url}")
        metadata = {
            "id": original_path.stem,
            "link": url, # Standardized key
            "caption": original_path.stem, # Standardized key
            "likes": 0,
            "shares": 0,
            "postdatetime": "N/A" # Standardized key
        }
    else: # It's a URL, download it
        progress_message = "starting video download...\r"
        ydl_opts = {
            'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'outtmpl': 'videos/%(id)s.%(ext)s',
            'progress_hooks': [progress_hook],
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=True))
            original_filepath_str = ydl.prepare_filename(info)
            original_path = Path(original_filepath_str)
            video_id = info.get("id")
            video_dir = Path("videos")
            transcript_path = next(video_dir.glob(f"{video_id}*.en.vtt"), None)
            if not transcript_path:
                 transcript_path = next(video_dir.glob(f"{video_id}*.vtt"), None)
            
            caption_text = info.get("description", info.get("title", "N/A"))
            clean_caption = caption_text.encode('ascii', 'ignore').decode('ascii').strip()

            custom_clickbait_score = await loop.run_in_executor(
                    None, 
                    custom_clickbait_model.predict_clickbait_binary, 
                    clean_caption
                )
            
            progress_message = f"Custom clickbait prediction (1/0) complete: {custom_clickbait_score}\n"
            logging.info(progress_message)


            metadata = {
                "id": info.get("id", "N/A"),
                "link": info.get("webpage_url", url), # Standardized key
                "caption": clean_caption, # Standardized key
                "likes": info.get("like_count", 0),
                "shares": info.get("repost_count", 0),
                "postdatetime": info.get("upload_date", "N/A"), # Standardized key (YYYYMMDD)
            }

            metadata["custom_clickbait_binary"] = custom_clickbait_score


    progress_message = f"Cleaning video file: {original_path}\n"
    logging.info(f"Original video path: {original_path}")
    
    sanitized_path = original_path.with_name(f"{original_path.stem}_fixed.mp4")
    ffmpeg_video_command = [
        "ffmpeg", "-i", str(original_path), "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-c:a", "aac", "-y", str(sanitized_path)
    ]
    await run_subprocess_async(ffmpeg_video_command)
    progress_message = "Video processed. Extracting audio...\n"

    audio_path = sanitized_path.with_suffix('.wav')
    try:
        ffmpeg_audio_command = [
            "ffmpeg", "-i", str(sanitized_path), "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "-y", str(audio_path)
        ]
        await run_subprocess_async(ffmpeg_audio_command)
        progress_message = "Audio extracted successfully.\n"
        audio_path_str = str(audio_path)
    except RuntimeError as e:
        progress_message = "Could not extract audio. The video might be silent.\n"
        logging.warning(f"Could not extract audio from {sanitized_path}, continuing without it. Error: {e}")
        audio_path_str = None

    if not transcript_path or not Path(transcript_path).exists():
        progress_message = "No pre-existing transcript found. Generating one locally...\n"
        logging.info(progress_message)
        if audio_path_str and Path(audio_path_str).exists():
            generated_vtt_path = await loop.run_in_executor(None, transcription.generate_transcript, audio_path_str)
            transcript_path = generated_vtt_path if generated_vtt_path else transcript_path
        else:
            progress_message = "No audio file found, skipping local transcription.\n"
    else:
        progress_message = "Using pre-existing transcript file.\n"
    
    logging.info(f"Assets prepared: Video='{sanitized_path}', Audio='{audio_path_str}', Transcript='{transcript_path}'")
    
    return {
        "video": str(sanitized_path),
        "transcript": str(transcript_path) if transcript_path and Path(transcript_path).exists() else None,
        "metadata": metadata,
    }

async def process_request_stream(video_url: str, question: str, generation_config: dict, prompts: dict, model_selection: str, checks: dict, gemini_config: dict, vertex_config: dict):
    """
    Generator function that yields progress updates for a single video.
    Routes to the appropriate model pipeline (Gemini or local).
    """
    global progress_message
    paths = None
    try:
        # Step 1: Uniformly prepare video assets regardless of the model
        preparation_task = asyncio.create_task(prepare_video_assets_async(video_url))
        while not preparation_task.done():
            yield f"data: {progress_message}\n\n"
            await asyncio.sleep(0.2)
        yield f"data: {progress_message}\n\n"
        paths = await preparation_task
        video_path = paths.get("video")
        if not video_path:
            raise ValueError("Video file could not be prepared.")

        # Step 2: Route to the correct model logic
        if model_selection == 'gemini':
            yield "data: Using Gemini Model for inference.\n\n"
            async for message in inference_logic.run_gemini_pipeline(video_path, question, checks, gemini_config):
                yield f"data: {message}\n\n"
        elif model_selection == 'vertex':
            yield "data: Using Vertex AI Model for inference.\n\n"
            async for message in inference_logic.run_vertex_pipeline(video_path, question, checks, vertex_config):
                yield f"data: {message}\n\n"
        else:
            inference_logic.switch_active_model(model_selection)
            yield f"data: Using {model_selection.capitalize()} Model for inference.\n\n"
            is_factuality_run = any(checks.values())
            
            if is_factuality_run:
                yield "data: Starting Factuality & Credibility pipeline...\n\n"
                async for message in factuality_logic.run_factuality_pipeline(paths, checks, generation_config):
                    yield f"data: {message}\n\n"
            else:
                yield "data: Starting General Q&A pipeline...\n\n"
                async for message in inference_logic.run_inference_pipeline(video_path, question, generation_config, prompts):
                    yield f"data: {message}\n\n"

    except Exception as e:
        error_message = f"\n\n ERROR: \nAn error occurred: {str(e)}"
        logging.error(f"error in processing stream for URL '{video_url}'", exc_info=True)
        yield f"data: {error_message}\n\n"
    finally:
        pass


@app.post("/process")
async def process_video_endpoint(
    # core params
    video_url: str = Form(...),
    question: str = Form(...),
    model_selection: str = Form("default"),

    # gemini params
    gemini_api_key: str = Form(""),
    gemini_model_name: str = Form(""),

    # vertex params
    vertex_project_id: str = Form(""),
    vertex_location: str = Form(""),
    vertex_model_name: str = Form(""),
    vertex_api_key: str = Form(""),

    # local model params
    num_perceptions: int = Form(...),
    sampling_fps: float = Form(...),
    max_new_tokens: int = Form(...),
    temperature: float = Form(...),
    top_p: float = Form(...),
    repetition_penalty: float = Form(...),
    prompt_glue: str = Form(...),
    prompt_final: str = Form(...),
    
    # factuality checks
    check_visuals: bool = Form(False),
    check_content: bool = Form(False),
    check_audio: bool = Form(False),
):
    """Endpoint to handle a single video analysis request."""
    generation_config = {
        "num_perceptions": num_perceptions,
        "sampling_fps": sampling_fps,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    prompts = {"glue": prompt_glue, "final": prompt_final}
    checks = {"visuals": check_visuals, "content": check_content, "audio": check_audio}
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}

    async def stream_wrapper():
        async for message in process_request_stream(video_url, question, generation_config, prompts, model_selection, checks, gemini_config, vertex_config):
            yield message
        yield "event: close\ndata: Task finished.\n\n"

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

async def generate_and_save_croissant_metadata(row_data: dict) -> str:
    """
    Generates and saves a Croissant-ml metadata JSON file for a single row of data.
    Returns the path to the created file as a string.
    """
    if not CROISSANT_AVAILABLE:
        logging.warning("'croissant' library not installed. Skipping metadata generation.")
        return "N/A (croissant library not installed)"

    try:
        # Standardized field names
        fields = [
            # Tier 1 & 2
            cnodes.Field(name="id", description="Unique identifier for the video.", data_types=DataType.TEXT),
            cnodes.Field(name="link", description="URL to the original social media post.", data_types=DataType.URL),
            cnodes.Field(name="caption", description="Original user-provided caption.", data_types=DataType.TEXT),
            cnodes.Field(name="likes", description="Number of likes for the post.", data_types=DataType.INTEGER),
            cnodes.Field(name="shares", description="Number of shares for the post.", data_types=DataType.INTEGER),
            cnodes.Field(name="postdatetime", description="Original post time of the video.", data_types=DataType.TEXT),
            cnodes.Field(name="collecttime", description="Timestamp when the data was collected.", data_types=DataType.TEXT),
            cnodes.Field(name="videotranscriptionpath", description="Path to the VTT transcript file.", data_types=DataType.TEXT),

            # Tier 3: General Labels
            cnodes.Field(name="videocontext", description="AI-generated summary of the video's content.", data_types=DataType.TEXT),
            cnodes.Field(name="politicalbias", description="AI score (1-10) for political bias.", data_types=DataType.INTEGER),
            cnodes.Field(name="criticism", description="AI score (1-10) for criticism level.", data_types=DataType.INTEGER),
            cnodes.Field(name="videoaudiopairing", description="AI score (1-10) for video-audio alignment.", data_types=DataType.INTEGER),
            cnodes.Field(name="videocaptionpairing", description="AI score (1-10) for video-caption alignment.", data_types=DataType.INTEGER),
            cnodes.Field(name="audiocaptionpairing", description="AI score (1-10) for audio-caption alignment.", data_types=DataType.INTEGER),

            # Tier 3: Disinformation Analysis Labels
            cnodes.Field(name="disinfo_level", description="Classification of misinformation severity.", data_types=DataType.TEXT),
            cnodes.Field(name="disinfo_intent", description="The inferred intent behind any misinformation.", data_types=DataType.TEXT),
            cnodes.Field(name="disinfo_threat_vector", description="The specific technique used for deception.", data_types=DataType.TEXT),
            cnodes.Field(name="disinfo_emotional_charge", description="AI score (1-10) for how emotionally charged the content is.", data_types=DataType.INTEGER),
            cnodes.Field(name="disinfo_targets_cognitive_bias", description="Boolean indicating if content targets cognitive biases.", data_types=DataType.BOOL),
            cnodes.Field(name="disinfo_promotes_tribalism", description="Boolean indicating if content promotes an 'us vs. them' mentality.", data_types=DataType.BOOL),
        ]

        temp_csv = io.StringIO()
        fieldnames = [f.name for f in fields]
        writer = csv.DictWriter(temp_csv, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(row_data)

        record_set = cnodes.RecordSet(
            name="video_metadata_record",
            description="A single record of labels and metadata for one social media video.",
            fields=fields,
            data=temp_csv.getvalue()
        )

        video_id = row_data.get('id', 'unknown_video')
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        link_url = row_data.get('link', f"http://vchat-dataset.org/data/{video_id}/{timestamp}")

        distribution = [cnodes.FileObject(name=f"source_video_url_{video_id}", encoding_format="text/html", content_url=link_url)]

        croissant_metadata = Metadata(
            name=f"vchat-label-{video_id}",
            description=f"Croissant metadata for video ID {video_id} generated on {timestamp}.",
            url=link_url, distribution=distribution, record_sets=[record_set],
        )

        metadata_dir = Path("metadata")
        metadata_filename = f"{video_id}_{timestamp}.json"
        metadata_path = metadata_dir / metadata_filename

        loop = asyncio.get_event_loop()
        json_content = croissant_metadata.to_json()
        await loop.run_in_executor(None, lambda: metadata_path.write_text(json.dumps(json_content, indent=2)))

        logging.info(f"Croissant metadata saved to {metadata_path}")
        return str(metadata_path)

    except Exception as e:
        logging.error(f"Failed to generate or save Croissant metadata: {e}", exc_info=True)
        return f"Error generating metadata: {e}"

async def get_labels_for_link(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    """
    Prepares a video, gets labels, and returns a dictionary of results with standardized keys.
    """
    global progress_message
    paths = None
    try:
        yield "Step 1: Preparing video assets..."
        preparation_task = asyncio.create_task(prepare_video_assets_async(video_url))
        while not preparation_task.done():
            yield progress_message
            await asyncio.sleep(0.2)
        yield progress_message
        paths = await preparation_task

        video_path = paths.get("video")
        transcript_path = paths.get("transcript")
        metadata = paths.get("metadata", {})

        if not video_path:
            raise ValueError("Video file could not be prepared.")
        
        yield "Step 2: Reading audio transcript..."
        transcript_text = "No transcript available for this video."
        if transcript_path:
            transcript_text = parse_vtt(transcript_path)
            yield "  - Transcript found and parsed."
        
        caption = metadata.get("caption", "No caption available.")
        yield f"Step 3: Sending video and context to {model_selection.capitalize()} for labeling..."
        
        final_labels = None
        if model_selection == 'gemini':
            async for message in inference_logic.run_gemini_labeling_pipeline(video_path, caption, transcript_text, gemini_config, include_comments):
                if isinstance(message, dict): final_labels = message
                elif isinstance(message, str): yield message.replace(os.linesep, ' ')
        elif model_selection == 'vertex':
            async for message in inference_logic.run_vertex_labeling_pipeline(video_path, caption, transcript_text, vertex_config, include_comments):
                if isinstance(message, dict): final_labels = message
                elif isinstance(message, str): yield message.replace(os.linesep, ' ')

        if final_labels is None:
            raise RuntimeError(f"Failed to get parsed labels from the {model_selection.capitalize()} pipeline.")

        def get_score(value):
            """Safely extracts the score, whether it's a direct value or in a nested dict."""
            if isinstance(value, dict):
                return value.get('score', '')
            return value

        disinfo_analysis = final_labels.get("disinformation_analysis", {})
        sentiment_tactics = disinfo_analysis.get("sentiment_and_bias_tactics", {})

        output_row_data = {
            "id": metadata.get("id", ""),
            "link": metadata.get("link", video_url),
            "caption": metadata.get("caption", ""),
            "likes": metadata.get("likes", 0),
            "shares": metadata.get("shares", 0),
            "postdatetime": metadata.get("postdatetime", ""),
            "collecttime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "videotranscriptionpath": transcript_path or "",
            "videocontext": final_labels.get("video_context_summary", ""),
            "politicalbias": get_score(final_labels.get("political_bias", "")),
            "criticism": get_score(final_labels.get("criticism_level", "")),
            "videoaudiopairing": get_score(final_labels.get("video_audio_pairing", "")),
            "videocaptionpairing": get_score(final_labels.get("video_caption_pairing", "")),
            "audiocaptionpairing": get_score(final_labels.get("audio_caption_pairing", "")),

            "custom_clickbait_binary": metadata.get("custom_clickbait_binary", -1),
            
            "disinfo_level": disinfo_analysis.get("disinformation_level", ""),
            "disinfo_intent": disinfo_analysis.get("disinformation_intent", ""),
            "disinfo_threat_vector": disinfo_analysis.get("threat_vector", ""),
            "disinfo_emotional_charge": sentiment_tactics.get("emotional_charge", ""),
            "disinfo_targets_cognitive_bias": sentiment_tactics.get("targets_cognitive_bias", ""),
            "disinfo_promotes_tribalism": sentiment_tactics.get("promotes_tribalism", "")
        }
        
        yield {"csv_row": output_row_data, "full_json": final_labels}

    except Exception as e:
        error_message = f"ERROR in get_labels_for_link for {video_url}: {str(e)}"
        logging.error(error_message, exc_info=True)
        yield {"error": error_message}

async def process_labeling_stream(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    """
    Generator function for the automated labeling of a SINGLE video, appending to dataset.csv.
    """
    final_data = None
    async for result in get_labels_for_link(video_url, gemini_config, vertex_config, model_selection, include_comments):
        if isinstance(result, str):
            yield f"data: {result}\n\n"
        elif isinstance(result, dict):
            if "error" in result:
                yield f"data: ERROR: {result['error']}\n\n"
                return
            final_data = result

    if not final_data or "csv_row" not in final_data:
        yield f"data: ERROR: Could not generate label data.\n\n"
        return
    
    final_row_data = final_data["csv_row"]
    full_labels_json = final_data["full_json"]
    
    # Save the full JSON analysis to its own file
    labels_dir = Path("data/labels")
    video_id = final_row_data.get('id', 'unknown_video')
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    json_filename = f"{video_id}_{timestamp}_labels.json"
    json_path = labels_dir / json_filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_labels_json, f, indent=2, ensure_ascii=False)
    yield f"data: Step 4: Full analysis with reasoning saved to {json_path}\n\n"

    yield "data: Step 5: Generating metadata...\n\n"
    metadatapath_value = await generate_and_save_croissant_metadata(final_row_data)
    final_row_data["metadatapath"] = metadatapath_value
    if "Error" in metadatapath_value:
        yield f"data:   - WARNING: {metadatapath_value}\n\n"
    else:
        yield f"data:   - Croissant metadata successfully saved to {metadatapath_value}\n\n"

    dataset_path = Path("data/dataset.csv")
    yield f"data: Step 6: Appending data to persistent dataset at {dataset_path}...\n\n"
    
    all_headers = list(dict.fromkeys(list(final_row_data.keys()) + ["metadatapath"]))
    
    file_exists = dataset_path.is_file() and dataset_path.stat().st_size > 0
    with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(final_row_data)
    
    yield f"data: Successfully appended a new record to {dataset_path}.\n"
    yield "data: \nLabeling process complete.\n"


@app.post("/label_video")
async def label_video_endpoint(
    video_url: str = Form(...),
    model_selection: str = Form(...),
    gemini_api_key: str = Form(""),
    gemini_model_name: str = Form(""),
    vertex_project_id: str = Form(""),
    vertex_location: str = Form(""),
    vertex_model_name: str = Form(""),
    vertex_api_key: str = Form(""),
    include_comments: bool = Form(False),
):
    """Endpoint to handle the automated video labeling process for a single video."""
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}
    
    if model_selection == 'gemini' and not gemini_api_key:
        async def error_stream():
            yield "data: ERROR: Gemini API Key is required for this feature.\n\n"
            yield "event: close\ndata: Task finished.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    if model_selection == 'vertex' and not vertex_project_id:
        async def error_stream():
            yield "data: ERROR: Vertex AI Project ID is required for this feature.\n\n"
            yield "event: close\ndata: Task finished.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")


    async def stream_wrapper():
        async for message in process_labeling_stream(video_url, gemini_config, vertex_config, model_selection, include_comments):
            yield message
        yield "event: close\ndata: Task finished.\n\n"

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


async def process_batch_labeling_stream(csv_file: UploadFile, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    """
    Processes a CSV, appends new, labeled data to a persistent `data/dataset.csv`,
    and generates metadata. Skips links that are already processed.
    """
    yield f"data: Batch labeling process started with {model_selection.capitalize()}. Results will be appended to data/dataset.csv.\n\n"
    
    dataset_path = Path("data/dataset.csv")
    processed_links = set()
    try:
        if dataset_path.exists() and dataset_path.stat().st_size > 0:
            with open(dataset_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('link') and row.get('videocontext'):
                        processed_links.add(row['link'])
            yield f"data: Found {len(processed_links)} previously processed links in {dataset_path}.\n\n"
        else:
            yield f"data: No existing dataset found at {dataset_path}. A new one will be created.\n\n"
    except Exception as e:
        yield f"data: WARNING: Could not read existing dataset.csv. Error: {e}\n\n"

    try:
        contents = await csv_file.read()
        decoded_content = contents.decode('utf-8', errors='ignore')
        reader = csv.DictReader(io.StringIO(decoded_content))
        
        original_headers = reader.fieldnames or []
        if 'link' not in original_headers:
            yield "data: FATAL ERROR: Uploaded CSV must contain a header with a 'link' column.\n\n"
            return
        
        input_rows = [row for row in reader if row.get('link')]
        
        label_columns = [
            "id", "link", "caption", "likes", "shares", "postdatetime", "collecttime", 
            "videotranscriptionpath", "videocontext", "politicalbias", "criticism", 
            "videoaudiopairing", "videocaptionpairing", "audiocaptionpairing", 
            "disinfo_level", "disinfo_intent", "disinfo_threat_vector", 
            "disinfo_emotional_charge", "disinfo_targets_cognitive_bias", 
            "disinfo_promotes_tribalism", "metadatapath"
        ]
        output_headers = list(dict.fromkeys(original_headers + label_columns))

        yield f"data: Found {len(input_rows)} rows with links in the uploaded CSV to process.\n\n"
        new_records_appended = 0
        for i, row in enumerate(input_rows):
            link = row.get('link')
            yield f"data: ==================================================================\n"
            yield f"data: Processing Video {i+1}/{len(input_rows)}: {link}\n"
            yield f"data: ==================================================================\n\n"
            
            if link in processed_links:
                yield f"data:   -> SKIPPING: This link has already been processed in dataset.csv.\n\n"
                continue

            label_data_packet = None
            async for result in get_labels_for_link(link, gemini_config, vertex_config, model_selection, include_comments):
                if isinstance(result, str):
                    yield f"data:   {result}\n"
                elif isinstance(result, dict):
                    label_data_packet = result
            
            if label_data_packet and "error" not in label_data_packet:
                csv_row = label_data_packet["csv_row"]
                full_labels_json = label_data_packet["full_json"]
                full_row_data = row.copy()
                full_row_data.update(csv_row)
                yield f"data: \n  -> Successfully generated labels for {link}.\n"
                
                labels_dir = Path("data/labels")
                video_id = csv_row.get('id', f'unknown_batch_{i+1}')
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                json_filename = f"{video_id}_{timestamp}_labels.json"
                json_path = labels_dir / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(full_labels_json, f, indent=2, ensure_ascii=False)
                yield f"data:   -> Full analysis with reasoning saved to {json_path}\n"
                
                yield f"data:   -> Generating Croissant metadata...\n"
                metadata_path = await generate_and_save_croissant_metadata(full_row_data)
                full_row_data['metadatapath'] = metadata_path
                if "Error" in metadata_path:
                     yield f"data:   -> WARNING: {metadata_path}\n\n"
                else:
                     yield f"data:   -> Metadata saved to: {metadata_path}\n\n"
                
                file_exists = dataset_path.is_file() and dataset_path.stat().st_size > 0
                with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=output_headers, extrasaction='ignore')
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(full_row_data)
                
                new_records_appended += 1
                yield f"data:   -> SUCCESS: Appended new record to {dataset_path}.\n\n"
            else:
                error_msg = label_data_packet.get('error', 'An unknown error occurred.') if label_data_packet else 'An unknown error occurred.'
                yield f"data: \n  -> FAILED to process {link}. Reason: {error_msg}\n\n"

        yield "data: \n\n================================\n"
        yield f"data: Batch processing complete. Appended {new_records_appended} new records to {dataset_path}.\n"
        yield "data: ================================\n\n"

    except Exception as e:
        error_message = f"\n\nFATAL BATCH ERROR: {str(e)}"
        logging.error("A fatal error occurred during batch labeling.", exc_info=True)
        yield f"data: {error_message}\n\n"
    finally:
        yield "event: close\ndata: Task finished.\n\n"


@app.post("/batch_label")
async def batch_label_endpoint(
    csv_file: UploadFile = File(...),
    model_selection: str = Form(...),
    gemini_api_key: str = Form(""),
    gemini_model_name: str = Form(""),
    vertex_project_id: str = Form(""),
    vertex_location: str = Form(""),
    vertex_model_name: str = Form(""),
    vertex_api_key: str = Form(""),
    include_comments: bool = Form(False),
):
    """Endpoint to handle batch video labeling, appending to a persistent dataset."""
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}

    if model_selection == 'gemini' and not gemini_api_key:
        async def error_stream():
            yield "data: ERROR: Gemini API Key is required for batch auto-labeling.\n\n"
            yield "event: close\ndata: Task finished.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    if model_selection == 'vertex' and not vertex_project_id:
        async def error_stream():
            yield "data: ERROR: Vertex AI Project ID is required for batch auto-labeling.\n\n"
            yield "event: close\ndata: Task finished.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    return StreamingResponse(
        process_batch_labeling_stream(csv_file, gemini_config, vertex_config, model_selection, include_comments),
        media_type="text/event-stream"
    )
