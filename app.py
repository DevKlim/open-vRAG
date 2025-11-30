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
import hashlib
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, Response, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yt_dlp
import inference_logic
import factuality_logic
import transcription
from factuality_logic import parse_vtt
from toon_parser import parse_veracity_toon

import custom_clickbait_model

# --- CroissantML Imports with error handling ---
try:
    import mlcroissant as cnodes
    # Depending on the version, Metadata might be top level or under nodes. 
    # Checking typical usage:
    from mlcroissant import Metadata, RecordSet, Field, DataType
    CROISSANT_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older/variant naming if strictly needed
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

# Check for LITE_MODE.
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

#  fastAPI app setup 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("videos", exist_ok=True)
os.makedirs("data", exist_ok=True) # Ensure the data directory for the CSV exists
os.makedirs("data/labels", exist_ok=True) # Ensure the directory for labels exists
os.makedirs("metadata", exist_ok=True) # Ensure the metadata directory exists


@app.on_event("startup")
async def startup_event():
    """Load all models on application startup."""
    logging.info("Application starting up...")
    if not LITE_MODE:
        try:
            inference_logic.load_models()
            transcription.load_model()
        except Exception as e:
            logging.fatal(f"Could not load models. Error: {e}", exc_info=True)
    else:
        logging.info("Running in LITE mode. Local models and transcription are disabled.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page, indicating if a custom model is available."""
    custom_model_available = False
    if not LITE_MODE:
        custom_model_available = inference_logic.peft_model is not None
        
    return templates.TemplateResponse("index.html", {
        "request": request,
        "custom_model_available": custom_model_available,
        "lite_mode": LITE_MODE
    })

@app.get("/model-architecture", response_class=PlainTextResponse)
async def get_model_architecture():
    """Returns the base model architecture as a plain text string."""
    if LITE_MODE:
        return "Running in LITE mode. No local model is loaded. Using Google Cloud APIs."
        
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
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        logging.warning(f"Subprocess cancelled: {command}")
        process.kill()
        raise

    if process.returncode != 0:
        error_details = stderr.decode()
        logging.error(f"FFmpeg command failed: {command}")
        logging.error(f"FFmpeg stderr: {error_details}")
        raise RuntimeError(f"Process failed:\n{error_details}")
    return stdout.decode()

def check_if_processed(link: str) -> bool:
    """Checks if a link has already been processed in dataset.csv."""
    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists():
        return False
    try:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('link') == link:
                    return True
    except Exception:
        return False
    return False

async def prepare_video_assets_async(url: str) -> dict:
    """
    Downloads video (if URL) or finds local video, then prepares assets including transcription.
    Skips download and ffmpeg processing if files already exist.
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
            "id": hashlib.md5(str(url).encode('utf-8')).hexdigest()[:16],
            "link": url,
            "caption": original_path.stem,
            "likes": 0,
            "shares": 0,
            "postdatetime": "N/A"
        }
    else: # It's a URL
        # Use yt-dlp to get metadata and ID first without downloading if possible
        ydl_opts_meta = {'quiet': True, 'noplaylist': True}
        video_id = "unknown"
        try:
             with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
                info_meta = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=False))
                video_id = info_meta.get("id")
        except Exception:
            # Fallback ID generation if fetch fails but we proceed to download loop
            video_id = hashlib.md5(url.encode('utf-8')).hexdigest()[:16]

        # Check if we already have the sanitized file for this ID
        sanitized_check = Path(f"videos/{video_id}_fixed.mp4")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'outtmpl': 'videos/%(id)s.%(ext)s',
            'progress_hooks': [progress_hook],
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True,
            'noplaylist': True,
            # Don't download if we already have the sanitized version
            'no_overwrites': True 
        }

        if sanitized_check.exists():
            progress_message = "Video already exists locally. Skipping download.\n"
            # Mock info for metadata
            info = info_meta if 'info_meta' in locals() else {'id': video_id, 'title': video_id}
            original_path = Path(f"videos/{video_id}.mp4") # Placeholder, won't be used if sanitized exists
        else:
            progress_message = "Starting video download...\r"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=True))
                original_filepath_str = ydl.prepare_filename(info)
                original_path = Path(original_filepath_str)
        
        video_id = info.get("id", video_id)
        video_dir = Path("videos")
        transcript_path = next(video_dir.glob(f"{video_id}*.en.vtt"), None)
        if not transcript_path:
                transcript_path = next(video_dir.glob(f"{video_id}*.vtt"), None)
        
        caption_text = info.get("description", info.get("title", "N/A"))
        clean_caption = caption_text.encode('ascii', 'ignore').decode('ascii').strip()

        metadata = {
            "id": video_id,
            "link": info.get("webpage_url", url),
            "caption": clean_caption,
            "likes": info.get("like_count", 0),
            "shares": info.get("repost_count", 0),
            "postdatetime": info.get("upload_date", "N/A"),
        }

    # Sanitize Check
    if original_path:
        sanitized_path = original_path.with_name(f"{original_path.stem}_fixed.mp4")
    else:
        # Fallback for when we skipped download but need path
        sanitized_path = Path(f"videos/{video_id}_fixed.mp4")

    if sanitized_path.exists():
        progress_message = f"Using existing sanitized video: {sanitized_path}\n"
    else:
        progress_message = f"Cleaning video file: {original_path}\n"
        if not original_path or not original_path.exists():
             # This shouldn't happen unless download failed silently or logic error
             raise FileNotFoundError("Video file not found and cannot be processed.")
             
        ffmpeg_video_command = [
            "ffmpeg", "-i", str(original_path), "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-c:a", "aac", "-y", str(sanitized_path)
        ]
        await run_subprocess_async(ffmpeg_video_command)

    # Audio Extraction Check
    audio_path = sanitized_path.with_suffix('.wav')
    if audio_path.exists():
         progress_message = "Using existing audio track.\n"
         audio_path_str = str(audio_path)
    else:
        progress_message = "Extracting audio...\n"
        try:
            ffmpeg_audio_command = [
                "ffmpeg", "-i", str(sanitized_path), "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", "-y", str(audio_path)
            ]
            await run_subprocess_async(ffmpeg_audio_command)
            audio_path_str = str(audio_path)
        except RuntimeError as e:
            logging.warning(f"Could not extract audio from {sanitized_path}: {e}")
            audio_path_str = None

    # Transcript Check
    if not transcript_path or not Path(transcript_path).exists():
        if LITE_MODE:
            logging.info("LITE mode: Local transcription disabled.")
        else:
            if audio_path_str and Path(audio_path_str).exists():
                # Check if a generated VTT exists matching audio name
                generated_vtt = audio_path.with_suffix('.vtt')
                if generated_vtt.exists():
                    progress_message = "Using existing local transcript.\n"
                    transcript_path = str(generated_vtt)
                else:
                    progress_message = "Generating transcript locally...\n"
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
    """
    global progress_message
    paths = None
    try:
        # Step 1: Prepare assets
        preparation_task = asyncio.create_task(prepare_video_assets_async(video_url))
        while not preparation_task.done():
            yield f"data: {progress_message}\n\n"
            await asyncio.sleep(0.2)
        yield f"data: {progress_message}\n\n"
        paths = await preparation_task
        video_path = paths.get("video")
        if not video_path:
            raise ValueError("Video file could not be prepared.")

        # Step 2: Route to model
        if model_selection == 'gemini':
            yield "data: Using Gemini Model for inference.\n\n"
            async for message in inference_logic.run_gemini_pipeline(video_path, question, checks, gemini_config):
                yield f"data: {message}\n\n"
        elif model_selection == 'vertex':
            yield "data: Using Vertex AI Model for inference.\n\n"
            async for message in inference_logic.run_vertex_pipeline(video_path, question, checks, vertex_config):
                yield f"data: {message}\n\n"
        else:
            if LITE_MODE:
                yield "data: ERROR: Local models are not available in this version.\n\n"
                return

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
    if not CROISSANT_AVAILABLE:
        return "N/A (croissant library not installed)"
    try:
        # Setup fields mapping to CSV structure
        # Note: The exact API for mlcroissant might differ slightly depending on version, 
        # adapting to standard record set creation.
        fields = [
            Field(name="id", description="Unique identifier.", data_types=DataType.TEXT),
            Field(name="link", description="URL.", data_types=DataType.URL),
            Field(name="visual_integrity_score", description="Score 1-10", data_types=DataType.INTEGER),
            Field(name="audio_integrity_score", description="Score 1-10", data_types=DataType.INTEGER),
            Field(name="source_credibility_score", description="Score 1-10", data_types=DataType.INTEGER),
            Field(name="logical_consistency_score", description="Score 1-10", data_types=DataType.INTEGER),
            Field(name="emotional_manipulation_score", description="Score 1-10", data_types=DataType.INTEGER),
            Field(name="final_veracity_score", description="Total score.", data_types=DataType.INTEGER),
            Field(name="grounding_check", description="RAG results.", data_types=DataType.TEXT),
        ]
        temp_csv = io.StringIO()
        fieldnames = [f.name for f in fields]
        writer = csv.DictWriter(temp_csv, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(row_data)

        record_set = RecordSet(name="video_metadata_record", fields=fields, data=temp_csv.getvalue())
        video_id = row_data.get('id', 'unknown_video')
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        link_url = row_data.get('link', f"http://vchat-dataset.org/data/{video_id}/{timestamp}")
        
        # distribution = [FileObject(name=f"source_video_url_{video_id}", encoding_format="text/html", content_url=link_url)]

        croissant_metadata = Metadata(
            name=f"vchat-label-{video_id}",
            url=link_url, 
            record_sets=[record_set],
        )
        metadata_path = Path("metadata") / f"{video_id}_{timestamp}.json"
        loop = asyncio.get_event_loop()
        json_content = croissant_metadata.to_json()
        await loop.run_in_executor(None, lambda: metadata_path.write_text(json.dumps(json_content, indent=2)))
        return str(metadata_path)
    except Exception as e:
        logging.error(f"Metadata error: {e}")
        return f"Error generating metadata: {e}"

async def get_labels_for_link(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
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
        transcript_text = "No transcript available."
        if transcript_path:
            transcript_text = parse_vtt(transcript_path)
            yield "  - Transcript found and parsed."
        
        caption = metadata.get("caption", "No caption available.")
        yield f"Step 3: Sending to {model_selection.capitalize()} for Veracity Vector Analysis..."
        
        final_labels = None
        raw_toon_text = ""
        
        if model_selection == 'gemini':
            async for message in inference_logic.run_gemini_labeling_pipeline(video_path, caption, transcript_text, gemini_config, include_comments):
                if isinstance(message, dict) and "parsed_data" in message:
                    final_labels = message["parsed_data"]
                    raw_toon_text = message.get("raw_toon", "")
                elif isinstance(message, str): 
                    yield message.replace(os.linesep, ' ')
        elif model_selection == 'vertex':
            async for message in inference_logic.run_vertex_labeling_pipeline(video_path, caption, transcript_text, vertex_config, include_comments):
                if isinstance(message, dict) and "parsed_data" in message: 
                    final_labels = message["parsed_data"]
                    raw_toon_text = message.get("raw_toon", "")
                elif isinstance(message, str): 
                    yield message.replace(os.linesep, ' ')

        if final_labels is None:
            raise RuntimeError("Failed to get parsed labels.")

        veracity_vectors = final_labels.get("veracity_vectors", {})
        factuality_factors = final_labels.get("factuality_factors", {})
        disinfo_analysis = final_labels.get("disinformation_analysis", {})
        final_assessment = final_labels.get("final_assessment", {})

        # ALGORITHMIC ID GENERATION
        # Ensure we have a valid ID. If metadata lacks it or is 'unknown', generate one hash-based.
        video_id = metadata.get("id", "")
        if not video_id or video_id == "unknown":
            video_id = hashlib.md5(video_url.encode("utf-8")).hexdigest()[:16]

        output_row_data = {
            "id": video_id,
            "link": metadata.get("link", video_url),
            "caption": metadata.get("caption", ""),
            "postdatetime": metadata.get("postdatetime", ""),
            "collecttime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "videotranscriptionpath": transcript_path or "",
            "video_context_summary": final_labels.get("video_context_summary", ""),
            "visual_integrity_score": veracity_vectors.get("visual_integrity_score", ""),
            "audio_integrity_score": veracity_vectors.get("audio_integrity_score", ""),
            "source_credibility_score": veracity_vectors.get("source_credibility_score", ""),
            "logical_consistency_score": veracity_vectors.get("logical_consistency_score", ""),
            "emotional_manipulation_score": veracity_vectors.get("emotional_manipulation_score", ""),
            "claim_accuracy": factuality_factors.get("claim_accuracy", ""),
            "grounding_check": factuality_factors.get("grounding_check", ""),
            "disinfo_classification": disinfo_analysis.get("classification", ""),
            "disinfo_intent": disinfo_analysis.get("intent", ""),
            "disinfo_threat_vector": disinfo_analysis.get("threat_vector", ""),
            "final_veracity_score": final_assessment.get("veracity_score_total", ""),
            "final_reasoning": final_assessment.get("reasoning", "")
        }
        yield {"csv_row": output_row_data, "full_json": final_labels, "raw_toon": raw_toon_text}

    except Exception as e:
        error_message = f"ERROR in get_labels_for_link for {video_url}: {str(e)}"
        logging.error(error_message, exc_info=True)
        yield {"error": error_message}

async def process_labeling_stream(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    """
    Automated labeling with existence check in dataset.csv.
    """
    # Check if already processed
    if check_if_processed(video_url):
         yield f"data: SKIPPING: This link ({video_url}) has already been processed in data/dataset.csv.\n\n"
         return

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
    full_labels_data = final_data["full_json"]
    raw_toon_text = final_data.get("raw_toon", "")
    
    labels_dir = Path("data/labels")
    video_id = final_row_data.get('id', 'unknown_video')
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Save JSON
    json_path = labels_dir / f"{video_id}_{timestamp}_labels.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_labels_data, f, indent=2, ensure_ascii=False)
        
    # Save raw TOON file
    toon_path = labels_dir / f"{video_id}_{timestamp}.toon"
    with open(toon_path, 'w', encoding='utf-8') as f:
        f.write(raw_toon_text)

    yield f"data: Step 4: Full analysis saved to {json_path} and {toon_path}\n\n"

    metadatapath_value = await generate_and_save_croissant_metadata(final_row_data)
    final_row_data["metadatapath"] = metadatapath_value
    
    dataset_path = Path("data/dataset.csv")
    all_headers = list(dict.fromkeys(list(final_row_data.keys()) + ["metadatapath"]))
    file_exists = dataset_path.is_file() and dataset_path.stat().st_size > 0
    
    with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        if not file_exists: writer.writeheader()
        writer.writerow(final_row_data)
    
    yield f"data: Successfully appended new record to {dataset_path}.\n"

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
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}
    
    if model_selection == 'gemini' and not gemini_api_key:
        return Response("Error: Gemini API Key required", status_code=400)

    async def stream_wrapper():
        async for message in process_labeling_stream(video_url, gemini_config, vertex_config, model_selection, include_comments):
            yield message
        yield "event: close\ndata: Task finished.\n\n"

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

@app.get("/download-dataset", response_class=FileResponse)
async def download_dataset():
    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists():
        return Response(content="Dataset file not found.", status_code=404)
    return FileResponse(path=dataset_path, filename="dataset.csv", media_type='text/csv')

async def process_batch_labeling_stream(csv_file: UploadFile, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    yield f"data: Batch labeling started. Using {model_selection}.\n\n"
    dataset_path = Path("data/dataset.csv")
    processed_links = set()
    
    if dataset_path.exists() and dataset_path.stat().st_size > 0:
        try:
            with open(dataset_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('link'): processed_links.add(row['link'])
            yield f"data: Found {len(processed_links)} processed links.\n\n"
        except Exception as e:
            yield f"data: Warning reading dataset: {e}\n\n"

    try:
        contents = await csv_file.read()
        decoded_content = contents.decode('utf-8', errors='ignore')
        reader = csv.DictReader(io.StringIO(decoded_content))
        input_rows = [row for row in reader if row.get('link')]
        
        for i, row in enumerate(input_rows):
            link = row.get('link')
            yield f"data: Processing {i+1}/{len(input_rows)}: {link}\n"
            
            if link in processed_links:
                yield f"data:   -> SKIPPING: Already in dataset.\n\n"
                continue

            label_data_packet = None
            async for result in get_labels_for_link(link, gemini_config, vertex_config, model_selection, include_comments):
                if isinstance(result, dict): label_data_packet = result
            
            if label_data_packet and "error" not in label_data_packet:
                csv_row = label_data_packet["csv_row"]
                full_row_data = row.copy()
                full_row_data.update(csv_row)
                raw_toon_text = label_data_packet.get("raw_toon", "")
                
                labels_dir = Path("data/labels")
                video_id = csv_row.get('id', f'batch_{i}')
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                
                # Save JSON
                with open(labels_dir / f"{video_id}_{timestamp}_labels.json", 'w', encoding='utf-8') as f:
                    json.dump(label_data_packet["full_json"], f, indent=2, ensure_ascii=False)
                
                # Save TOON
                with open(labels_dir / f"{video_id}_{timestamp}.toon", 'w', encoding='utf-8') as f:
                    f.write(raw_toon_text)

                # Metadata
                metadata_path = await generate_and_save_croissant_metadata(full_row_data)
                full_row_data['metadatapath'] = metadata_path
                
                # Append CSV
                file_exists = dataset_path.is_file() and dataset_path.stat().st_size > 0
                with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
                    output_headers = list(dict.fromkeys(list(full_row_data.keys()) + ["metadatapath"]))
                    writer = csv.DictWriter(f, fieldnames=output_headers, extrasaction='ignore')
                    if not file_exists: writer.writeheader()
                    writer.writerow(full_row_data)
                
                yield f"data:   -> Success. Appended to dataset.\n\n"
            else:
                err = label_data_packet.get('error', 'unknown') if label_data_packet else 'Failed'
                yield f"data:   -> FAILED: {err}\n\n"

    except Exception as e:
        yield f"data: FATAL BATCH ERROR: {e}\n\n"
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
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}

    if model_selection == 'gemini' and not gemini_api_key:
         return Response("Error: Gemini API Key required", status_code=400)

    return StreamingResponse(
        process_batch_labeling_stream(csv_file, gemini_config, vertex_config, model_selection, include_comments),
        media_type="text/event-stream"
    )