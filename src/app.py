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
import re
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

# --- Croissant Import Logic ---
try:
    import mlcroissant as mlc
    CROISSANT_AVAILABLE = True
except ImportError:
    # Fallback for older versions or different package names if needed
    try:
        import croissant as mlc
        CROISSANT_AVAILABLE = True
    except ImportError:
        mlc = None
        CROISSANT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("videos", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("data/labels", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

@app.on_event("startup")
async def startup_event():
    logging.info("Application starting up...")
    if not LITE_MODE:
        try:
            inference_logic.load_models()
            transcription.load_model()
        except Exception as e:
            logging.fatal(f"Could not load models. Error: {e}", exc_info=True)
    else:
        logging.info("Running in LITE mode.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
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
    if LITE_MODE: return "Running in LITE mode."
    if inference_logic.base_model: return str(inference_logic.base_model)
    return "Model not loaded."

@app.get("/download-dataset")
async def download_dataset():
    file_path = Path("data/dataset.csv")
    if file_path.exists():
        return FileResponse(path=file_path, filename="dataset.csv", media_type='text/csv')
    return Response("Dataset not found.", status_code=404)

progress_message = ""
def progress_hook(d):
    global progress_message
    if d['status'] == 'downloading':
        progress_message = f"Downloading: {d.get('_percent_str', 'N/A')} at {d.get('_speed_str', 'N/A')}\r"
    elif d['status'] == 'finished':
        progress_message = f"\nDownload finished. Preparing video assets...\n"

async def run_subprocess_async(command: list[str]):
    process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Process failed:\n{stderr.decode()}")
    return stdout.decode()

def extract_tweet_id(url: str) -> str | None:
    match = re.search(r"(?:twitter|x)\.com/[^/]+/status/(\d+)", url)
    return match.group(1) if match else None

def check_if_processed(link: str) -> bool:
    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists(): return False
    target_id = extract_tweet_id(link)
    try:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('link') == link: return True
                if target_id and row.get('id') == target_id: return True
    except Exception:
        return False
    return False

async def prepare_video_assets_async(url: str) -> dict:
    global progress_message
    loop = asyncio.get_event_loop()
    is_local = not (url.startswith("http://") or url.startswith("https://"))
    video_id = "unknown"
    transcript_path = None
    
    if is_local:
        original_path = Path(url)
        if not original_path.exists(): raise FileNotFoundError(f"File not found: {url}")
        video_id = hashlib.md5(str(url).encode('utf-8')).hexdigest()[:16]
        metadata = {"id": video_id, "link": url, "caption": original_path.stem}
    else:
        tweet_id = extract_tweet_id(url)
        video_id = tweet_id if tweet_id else hashlib.md5(url.encode('utf-8')).hexdigest()[:16]
        sanitized_check = Path(f"videos/{video_id}_fixed.mp4")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best', 'outtmpl': 'videos/%(id)s.%(ext)s',
            'progress_hooks': [progress_hook], 'quiet': True, 'noplaylist': True, 'no_overwrites': True,
            'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en']
        }
        
        if sanitized_check.exists():
            original_path = Path(f"videos/{video_id}.mp4")
            metadata = {"id": video_id, "link": url, "caption": "Cached Video"}
        else:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=True))
                original_path = Path(ydl.prepare_filename(info))
                metadata = {
                    "id": info.get("id", video_id), "link": info.get("webpage_url", url),
                    "caption": info.get("description", info.get("title", "N/A")).encode('ascii', 'ignore').decode('ascii').strip()[:500],
                    "postdatetime": info.get("upload_date", "N/A")
                }
                video_id = info.get("id", video_id)

        transcript_path = next(Path("videos").glob(f"{video_id}*.en.vtt"), None)
        if not transcript_path: transcript_path = next(Path("videos").glob(f"{video_id}*.vtt"), None)

    sanitized_path = Path(f"videos/{video_id}_fixed.mp4")
    if not sanitized_path.exists():
        await run_subprocess_async(["ffmpeg", "-i", str(original_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-y", str(sanitized_path)])

    audio_path = sanitized_path.with_suffix('.wav')
    if not audio_path.exists():
        try:
            await run_subprocess_async(["ffmpeg", "-i", str(sanitized_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(audio_path)])
        except: pass
    
    if not transcript_path and audio_path.exists() and not LITE_MODE:
        transcript_path = await loop.run_in_executor(None, transcription.generate_transcript, str(audio_path))

    return {"video": str(sanitized_path), "transcript": str(transcript_path) if transcript_path else None, "metadata": metadata}

async def generate_and_save_croissant_metadata(row_data: dict) -> str:
    """
    Generates ML Croissant metadata for a labeled video record.
    """
    if not CROISSANT_AVAILABLE: return "N/A"
    try:
        # FIX: When providing `data` directly to RecordSet, do NOT use Source/Extract 
        # that points to a file column. The dictionary keys in `data` imply the schema.
        
        fields = [
            mlc.Field(name="id", description="Unique ID", data_types=mlc.DataType.TEXT),
            mlc.Field(name="link", description="URL", data_types=mlc.DataType.URL),
            mlc.Field(name="visual_integrity_score", description="Score (1-10)", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="audio_integrity_score", description="Score (1-10)", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="source_credibility_score", description="Score (1-10)", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="logical_consistency_score", description="Score (1-10)", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="emotional_manipulation_score", description="Score (1-10)", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="video_audio_score", description="Modality Match: Video-Audio", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="video_caption_score", description="Modality Match: Video-Caption", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="audio_caption_score", description="Modality Match: Audio-Caption", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="final_veracity_score", description="Final Factuality Score (1-100)", data_types=mlc.DataType.INTEGER),
            mlc.Field(name="grounding_check", description="Evidence from search/RAG", data_types=mlc.DataType.TEXT),
        ]
        
        # Prepare the single record data ensuring types are correct for validation
        sanitized_data = {
            "id": str(row_data.get("id", "")),
            "link": str(row_data.get("link", "")),
            "visual_integrity_score": int(row_data.get("visual_integrity_score") or 0),
            "audio_integrity_score": int(row_data.get("audio_integrity_score") or 0),
            "source_credibility_score": int(row_data.get("source_credibility_score") or 0),
            "logical_consistency_score": int(row_data.get("logical_consistency_score") or 0),
            "emotional_manipulation_score": int(row_data.get("emotional_manipulation_score") or 0),
            "video_audio_score": int(row_data.get("video_audio_score") or 0),
            "video_caption_score": int(row_data.get("video_caption_score") or 0),
            "audio_caption_score": int(row_data.get("audio_caption_score") or 0),
            "final_veracity_score": int(row_data.get("final_veracity_score") or 0),
            "grounding_check": str(row_data.get("grounding_check", "")),
        }

        # Define the RecordSet with inline data
        record_set = mlc.RecordSet(
            name="video_metadata_record", 
            fields=fields, 
            data=[sanitized_data]
        )
        
        metadata = mlc.Metadata(
            name=f"vchat-label-{sanitized_data['id']}", 
            url=sanitized_data['link'], 
            record_sets=[record_set]
        )
        
        path = Path("metadata") / f"{sanitized_data['id']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        path.write_text(json.dumps(metadata.to_json(), indent=2))
        return str(path)
    except Exception as e:
        logging.error(f"Croissant Generation Error: {e}", exc_info=True)
        return "N/A (Error)"

async def get_labels_for_link(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    global progress_message
    try:
        yield "Step 1: Preparing assets..."
        paths = await prepare_video_assets_async(video_url)
        video_path = paths["video"]
        transcript_text = parse_vtt(paths["transcript"]) if paths["transcript"] else "No transcript."
        caption = paths["metadata"].get("caption", "")
        
        yield f"Step 2: Generating labels via {model_selection}..."
        final_labels = None
        raw_toon = ""
        
        pipeline = inference_logic.run_gemini_labeling_pipeline if model_selection == 'gemini' else inference_logic.run_vertex_labeling_pipeline
        config = gemini_config if model_selection == 'gemini' else vertex_config
        
        async for msg in pipeline(video_path, caption, transcript_text, config, include_comments):
            if isinstance(msg, dict) and "parsed_data" in msg:
                final_labels = msg["parsed_data"]
                raw_toon = msg.get("raw_toon", "")
            elif isinstance(msg, str): yield msg

        if not final_labels: raise RuntimeError("No labels generated.")
        
        # Extract Data with defaults to avoid KeyErrors
        vec = final_labels.get("veracity_vectors", {})
        mod = final_labels.get("modalities", {})
        fac = final_labels.get("factuality_factors", {})
        dis = final_labels.get("disinformation_analysis", {})
        fin = final_labels.get("final_assessment", {})
        
        row = {
            "id": paths["metadata"]["id"],
            "link": paths["metadata"]["link"],
            "caption": caption,
            "postdatetime": paths["metadata"].get("postdatetime", ""),
            "collecttime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "videotranscriptionpath": paths["transcript"] or "",
            "video_context_summary": final_labels.get("video_context_summary", ""),
            
            # Vectors
            "visual_integrity_score": vec.get("visual_integrity_score", "0"),
            "audio_integrity_score": vec.get("audio_integrity_score", "0"),
            "source_credibility_score": vec.get("source_credibility_score", "0"),
            "logical_consistency_score": vec.get("logical_consistency_score", "0"),
            "emotional_manipulation_score": vec.get("emotional_manipulation_score", "0"),
            
            # New Modalities
            "video_audio_score": mod.get("video_audio_score", "0"),
            "video_caption_score": mod.get("video_caption_score", "0"),
            "audio_caption_score": mod.get("audio_caption_score", "0"),
            
            "claim_accuracy": fac.get("claim_accuracy", ""),
            "grounding_check": fac.get("grounding_check", ""),
            "disinfo_classification": dis.get("classification", ""),
            "final_veracity_score": fin.get("veracity_score_total", "0"),
            "final_reasoning": fin.get("reasoning", "")
        }
        yield {"csv_row": row, "full_json": final_labels, "raw_toon": raw_toon}

    except Exception as e:
        yield {"error": str(e)}

async def process_labeling_stream(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    if check_if_processed(video_url):
        yield f"data: SKIPPING: {video_url} already in dataset.\n\n"
        return

    final_data = None
    async for res in get_labels_for_link(video_url, gemini_config, vertex_config, model_selection, include_comments):
        if isinstance(res, str): yield f"data: {res}\n\n"
        elif isinstance(res, dict):
            if "error" in res:
                yield f"data: ERROR: {res['error']}\n\n"
                return
            final_data = res
    
    if final_data:
        row = final_data["csv_row"]
        vid_id = row["id"]
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Save Files
        with open(f"data/labels/{vid_id}_{ts}_labels.json", 'w') as f: json.dump(final_data["full_json"], f, indent=2)
        with open(f"data/labels/{vid_id}_{ts}.toon", 'w') as f: f.write(final_data["raw_toon"])
        
        # Metadata
        row["metadatapath"] = await generate_and_save_croissant_metadata(row)
        
        # CSV Append logic
        dpath = Path("data/dataset.csv")
        exists = dpath.exists()
        fieldnames = list(row.keys())
        
        with open(dpath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            
        yield "data: Success! Data saved to dataset.csv and labels folder.\n\n"

@app.post("/process")
async def process_video_endpoint(video_url: str = Form(...), question: str = Form(...), model_selection: str = Form("default")):
    return Response("Q&A endpoint placeholder") 

@app.post("/label_video")
async def label_video_endpoint(
    video_url: str = Form(...), model_selection: str = Form(...),
    gemini_api_key: str = Form(""), gemini_model_name: str = Form(""),
    vertex_project_id: str = Form(""), vertex_location: str = Form(""), vertex_model_name: str = Form(""), vertex_api_key: str = Form(""),
    include_comments: bool = Form(False)
):
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}
    
    async def stream():
        async for msg in process_labeling_stream(video_url, gemini_config, vertex_config, model_selection, include_comments):
            yield msg
        yield "event: close\ndata: Done.\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/batch_label")
async def batch_label_endpoint(csv_file: UploadFile = File(...), model_selection: str = Form(...),
                               gemini_api_key: str = Form(""), gemini_model_name: str = Form(""),
                               vertex_project_id: str = Form(""), vertex_location: str = Form(""), vertex_model_name: str = Form(""), vertex_api_key: str = Form(""),
                               include_comments: bool = Form(False)):
    
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}

    async def batch_stream():
        yield "data: Starting Batch Process...\n\n"
        
        content = await csv_file.read()
        decoded = content.decode('utf-8').splitlines()
        reader = csv.DictReader(decoded)
        
        processed_count = 0
        
        for row in reader:
            link = row.get("link") or row.get("url")
            if not link: continue
            
            yield f"data: Processing: {link}\n\n"
            
            final_data = None
            async for res in get_labels_for_link(link, gemini_config, vertex_config, model_selection, include_comments):
                if isinstance(res, dict) and "csv_row" in res:
                    final_data = res
            
            if final_data:
                csv_row = final_data["csv_row"]
                vid_id = csv_row["id"]
                ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                
                with open(f"data/labels/{vid_id}_{ts}_labels.json", 'w') as f: json.dump(final_data["full_json"], f, indent=2)
                with open(f"data/labels/{vid_id}_{ts}.toon", 'w') as f: f.write(final_data["raw_toon"])
                
                csv_row["metadatapath"] = await generate_and_save_croissant_metadata(csv_row)
                
                dpath = Path("data/dataset.csv")
                exists = dpath.exists()
                fieldnames = list(csv_row.keys())
                
                with open(dpath, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    if not exists: writer.writeheader()
                    writer.writerow(csv_row)
                    
                processed_count += 1
                yield f"data: Finished {link}\n\n"
            else:
                yield f"data: Failed to label {link}\n\n"
                
        yield f"data: Batch Complete. Processed {processed_count} videos.\n\n"
        yield "event: close\ndata: Done.\n\n"

    return StreamingResponse(batch_stream(), media_type="text/event-stream")