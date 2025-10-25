import os
import sys
import asyncio
import subprocess
from pathlib import Path
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yt_dlp
import inference_logic
import factuality_logic

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
    Downloads video, subtitles, and extracts audio. Returns a dictionary of file paths.
    """
    global progress_message
    loop = asyncio.get_event_loop()
    progress_message = "starting video download...\r"
    
    ydl_opts = {
        'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': 'videos/%(id)s.%(ext)s',
        'progress_hooks': [progress_hook],
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
    }
    
    video_id = ""
    original_filepath_str = ""
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=True))
        original_filepath_str = ydl.prepare_filename(info)
        video_id = info.get("id")

    progress_message = f"cleaning video file: {original_filepath_str}\n"
    logging.info(f"og video downloaded to: {original_filepath_str}")
    original_path = Path(original_filepath_str)
    sanitized_path = original_path.with_name(f"{original_path.stem}_fixed.mp4")
    #reencoding as mov doesnt work!
    ffmpeg_video_command = [
        "ffmpeg", "-i", str(original_path), "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-c:a", "aac", "-y", str(sanitized_path)
    ]
    await run_subprocess_async(ffmpeg_video_command)
    progress_message = "video proc. extracting audio...\n"

    audio_path = sanitized_path.with_suffix('.wav')
    ffmpeg_audio_command = [
        "ffmpeg", "-i", str(sanitized_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", str(audio_path)
    ]
    await run_subprocess_async(ffmpeg_audio_command)
    progress_message = "Audio extracted successfully.\n"

    # for transcript
    video_dir = Path("videos")
    # yt-dlp might add language codes like .en.vtt
    transcript_path = next(video_dir.glob(f"{video_id}*.vtt"), None)

    logging.info(f"Assets prepared: Video='{sanitized_path}', Audio='{audio_path}', Transcript='{transcript_path}'")
    
    return {
        "video": str(sanitized_path),
        "audio": str(audio_path) if audio_path and audio_path.exists() else None,
        "transcript": str(transcript_path) if transcript_path and transcript_path.exists() else None,
    }

async def process_request_stream(video_url: str, question: str, generation_config: dict, prompts: dict, model_selection: str, checks: dict):
    """
    Generator function that yields progress updates.
    Routes to either the Q&A pipeline or the Factuality pipeline.
    """
    global progress_message
    paths = None
    try:
        inference_logic.switch_active_model(model_selection)
        yield f"data: Using {model_selection.capitalize()} Model for inference.\n\n"

        preparation_task = asyncio.create_task(prepare_video_assets_async(video_url))
        while not preparation_task.done():
            yield f"data: {progress_message}\n\n"
            await asyncio.sleep(0.2)
        yield f"data: {progress_message}\n\n"
        paths = await preparation_task

        # pipeline to run
        is_factuality_run = any(checks.values())
        
        if is_factuality_run:
            yield "data: Starting Factuality & Credibility pipeline...\n\n"
            async for message in factuality_logic.run_factuality_pipeline(paths, checks, generation_config):
                yield f"data: {message}\n\n"
        else:
            yield "data: Starting General Q&A pipeline...\n\n"
            video_path = paths.get("video")
            if not video_path:
                raise ValueError("Video file could not be prepared.")
            async for message in inference_logic.run_inference_pipeline(video_path, question, generation_config, prompts):
                yield f"data: {message}\n\n"

    except Exception as e:
        error_message = f"\n\n ERROR: \nAn error occurred: {str(e)}"
        logging.error(f"error in processing stream for URL '{video_url}'", exc_info=True)
        yield f"data: {error_message}\n\n"
    finally:
        yield "event: close\ndata: Task finished.\n\n"

@app.post("/process")
async def process_video_endpoint(
    # core params
    video_url: str = Form(...),
    question: str = Form(...),
    model_selection: str = Form("default"),

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
    """Endpoint to handle the main video analysis request."""
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

    return StreamingResponse(
        process_request_stream(video_url, question, generation_config, prompts, model_selection, checks),
        media_type="text/event-stream"
    )
