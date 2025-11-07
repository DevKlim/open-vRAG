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
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yt_dlp
import inference_logic
import factuality_logic
from factuality_logic import parse_vtt


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

@app.on_event("startup")
async def startup_event():
    """Load all models on application startup."""
    logging.info("Application starting up...")
    try:
        inference_logic.load_models()
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
    Downloads video (if URL) or finds local video, then prepares assets.
    Returns a dictionary of file paths and extracted metadata.
    """
    global progress_message
    loop = asyncio.get_event_loop()
    original_path = None
    transcript_path = None
    metadata = {}

    is_local_file = not (url.startswith("http://") or url.startswith("https://"))

    if is_local_file:
        progress_message = f"Processing local file: {url}\n"
        original_path = Path(url)
        if not original_path.exists():
            raise FileNotFoundError(f"Local video file not found at path: {url}")
        metadata = {
            "id": original_path.stem,
            "url": url,
            "caption": original_path.stem,
            "likes": 0,
            "shares": 0,
            "post_time": "N/A"
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
            transcript_path = next(video_dir.glob(f"{video_id}*.vtt"), None)
            
            # Extract metadata and clean caption
            caption_text = info.get("description", info.get("title", "N/A"))
            clean_caption = caption_text.encode('ascii', 'ignore').decode('ascii').strip()

            metadata = {
                "id": info.get("id", "N/A"),
                "url": info.get("webpage_url", url),
                "caption": clean_caption,
                "likes": info.get("like_count", 0),
                "shares": info.get("repost_count", 0),
                "post_time": info.get("upload_date", "N/A"), # YYYYMMDD format
            }

    progress_message = f"Cleaning video file: {original_path}\n"
    logging.info(f"Original video path: {original_path}")
    
    # Sanitize video and extract audio with ffmpeg (common step)
    sanitized_path = original_path.with_name(f"{original_path.stem}_fixed.mp4")
    ffmpeg_video_command = [
        "ffmpeg", "-i", str(original_path), "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-c:a", "aac", "-y", str(sanitized_path)
    ]
    await run_subprocess_async(ffmpeg_video_command)
    progress_message = "Video processed. Extracting audio...\n"

    audio_path = sanitized_path.with_suffix('.wav')
    ffmpeg_audio_command = [
        "ffmpeg", "-i", str(sanitized_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", str(audio_path)
    ]
    await run_subprocess_async(ffmpeg_audio_command)
    progress_message = "Audio extracted successfully.\n"

    logging.info(f"Assets prepared: Video='{sanitized_path}', Audio='{audio_path}', Transcript='{transcript_path}'")
    
    return {
        "video": str(sanitized_path),
        "audio": str(audio_path) if audio_path and audio_path.exists() else None,
        "transcript": str(transcript_path) if transcript_path and transcript_path.exists() else None,
        "metadata": metadata,
    }

async def process_request_stream(video_url: str, question: str, generation_config: dict, prompts: dict, model_selection: str, checks: dict, gemini_config: dict):
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
    
    async def stream_wrapper():
        async for message in process_request_stream(video_url, question, generation_config, prompts, model_selection, checks, gemini_config):
            yield message
        yield "event: close\ndata: Task finished.\n\n"

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


async def process_batch_stream(csv_file: UploadFile, question: str, generation_config: dict, prompts: dict, model_selection: str, checks: dict, gemini_config: dict):
    """
    Generator function that processes a batch of videos from a CSV file.
    """
    yield "data: Batch process started. Reading CSV file...\n\n"
    
    try:
        contents = await csv_file.read()
        decoded_content = contents.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded_content))
        
        video_urls = [row.get('url') for row in csv_reader if row.get('url')]
        
        if not video_urls:
            yield "data: ERROR: No 'url' column found or CSV is empty.\n\n"
            return

        yield f"data: Found {len(video_urls)} videos to process.\n\n"

        for i, url in enumerate(video_urls):
            yield f"data: \n\n================================\n"
            yield f"data: Starting Video {i+1}/{len(video_urls)}: {url}\n"
            yield f"data: ================================\n\n"
            
            try:
                # Reuse the single-video stream processor for each video
                async for message in process_request_stream(url, question, generation_config, prompts, model_selection, checks, gemini_config):
                    yield message
            except Exception as e:
                error_message = f"\n\nERROR processing {url}: {str(e)}\nThis video will be skipped."
                logging.error(f"Error in batch processing for URL '{url}'", exc_info=True)
                yield f"data: {error_message}\n\n"
            
            yield f"data: \n\n--- Finished analysis for: {url} ---\n\n"
    
    except Exception as e:
        error_message = f"\n\nFATAL BATCH ERROR: {str(e)}"
        logging.error("A fatal error occurred during batch processing.", exc_info=True)
        yield f"data: {error_message}\n\n"
    finally:
        yield "data: \n\nBatch processing complete.\n"
        yield "event: close\ndata: Task finished.\n\n"


@app.post("/batch_process")
async def batch_process_endpoint(
    # file upload
    csv_file: UploadFile = File(...),
    # core params
    question: str = Form(...),
    model_selection: str = Form("default"),

    # gemini params
    gemini_api_key: str = Form(""),
    gemini_model_name: str = Form(""),
    
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
    """Endpoint to handle batch video analysis from a CSV file."""
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

    return StreamingResponse(
        process_batch_stream(csv_file, question, generation_config, prompts, model_selection, checks, gemini_config),
        media_type="text/event-stream"
    )

async def process_labeling_stream(video_url: str, gemini_config: dict):
    """
    Generator function that yields progress updates for the automated labeling process.
    """
    global progress_message
    paths = None
    try:
        yield "data: Step 1: Preparing video assets (downloading, extracting metadata, etc.)...\n\n"
        preparation_task = asyncio.create_task(prepare_video_assets_async(video_url))
        while not preparation_task.done():
            yield f"data: {progress_message}\n\n"
            await asyncio.sleep(0.2)
        yield f"data: {progress_message}\n\n"
        paths = await preparation_task

        video_path = paths.get("video")
        transcript_path = paths.get("transcript")
        metadata = paths.get("metadata", {})

        if not video_path:
            raise ValueError("Video file could not be prepared.")
        
        yield f"data: Step 2: Reading audio transcript...\n"
        transcript_text = "No transcript available for this video."
        if transcript_path and os.path.exists(transcript_path):
            transcript_text = parse_vtt(transcript_path)
            yield f"data:   - Transcript found and parsed.\n\n"
        else:
            yield f"data:   - No transcript file found. Proceeding without it.\n\n"
        
        caption = metadata.get("caption", "No caption available.")

        yield "data: Step 3: Sending video, caption, and transcript to Gemini for labeling...\n\n"
        
        final_labels = None
        async for message in inference_logic.run_gemini_labeling_pipeline(video_path, caption, transcript_text, gemini_config):
            if isinstance(message, dict):
                final_labels = message # This is the final parsed JSON
            elif isinstance(message, str):
                 yield f"data: {message.replace(os.linesep, ' ')}\n\n"

        if final_labels is None:
            raise RuntimeError("Failed to get parsed labels from the Gemini pipeline.")

        yield "data: Step 4: Assembling final CSV data...\n\n"
        
        # Assemble the CSV row based on the spec
        output_row = {
            "id": metadata.get("id", ""),
            "twitterlink": metadata.get("url", video_url),
            "captions": caption,
            "likes": metadata.get("likes", 0),
            "shares": metadata.get("shares", 0),
            "videocontext": "", # Per spec, this is empty
            "videotranscriptionpath": transcript_path or "",
            "posttime": metadata.get("post_time", ""),
            "collecttime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Add the generated labels
            "politicalbias": final_labels.get("political_bias", ""),
            "misleading": final_labels.get("is_misleading", ""),
            "criticism": final_labels.get("criticism_level", ""),
            "videoaudiopairing": final_labels.get("video_audio_pairing", ""),
            "videocaptionpairing": final_labels.get("video_caption_pairing", ""),
            "audiocaptionparing": final_labels.get("audio_caption_pairing", "")
        }

        # Create a string buffer for the CSV output
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=output_row.keys())
        writer.writeheader()
        writer.writerow(output_row)
        csv_output_string = output.getvalue()
        
        yield "data: --- FINAL CSV OUTPUT ---\n\n"
        yield f"data: {csv_output_string.strip()}\n"
        yield "data: \nLabeling process complete.\n"

    except Exception as e:
        error_message = f"\n\n ERROR: \nAn error occurred: {str(e)}"
        logging.error(f"error in labeling stream for URL '{video_url}'", exc_info=True)
        yield f"data: {error_message}\n\n"
    finally:
        pass

@app.post("/label_video")
async def label_video_endpoint(
    video_url: str = Form(...),
    gemini_api_key: str = Form(""),
    gemini_model_name: str = Form(""),
):
    """Endpoint to handle the automated video labeling process."""
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    
    if not gemini_api_key:
        async def error_stream():
            yield "data: ERROR: Gemini API Key is required for the automated labeling feature. Please enter it in the 'Select Model' section.\n\n"
            yield "event: close\ndata: Task finished.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def stream_wrapper():
        async for message in process_labeling_stream(video_url, gemini_config):
            yield message
        yield "event: close\ndata: Task finished.\n\n"

    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")