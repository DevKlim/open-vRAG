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
    try:
        import croissant as mlc
        CROISSANT_AVAILABLE = True
    except ImportError:
        mlc = None
        CROISSANT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

app = FastAPI()

STATIC_DIR = "static"
if os.path.isdir("/usr/share/vchat/static"):
    STATIC_DIR = "/usr/share/vchat/static"
elif os.path.isdir("frontend/dist"):
    STATIC_DIR = "frontend/dist"
elif not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
    dummy_index = Path(STATIC_DIR) / "index.html"
    if not dummy_index.exists():
        dummy_index.write_text("<html><body>vChat Backend Running. Access via Port 8005 (Go Server).</body></html>")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=STATIC_DIR)

os.makedirs("data/videos", exist_ok=True)
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
    if not (Path(STATIC_DIR) / "index.html").exists():
        return HTMLResponse(content="Frontend not found. Please build frontend or access via Go server.", status_code=404)
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
        sanitized_check = Path(f"data/videos/{video_id}_fixed.mp4")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best', 
            'outtmpl': 'data/videos/%(id)s.%(ext)s',
            'progress_hooks': [progress_hook], 'quiet': True, 'noplaylist': True, 'no_overwrites': True,
            'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en']
        }
        
        if sanitized_check.exists():
            original_path = Path(f"data/videos/{video_id}.mp4")
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

        transcript_path = next(Path("data/videos").glob(f"{video_id}*.en.vtt"), None)
        if not transcript_path: transcript_path = next(Path("data/videos").glob(f"{video_id}*.vtt"), None)

    sanitized_path = Path(f"data/videos/{video_id}_fixed.mp4")
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

def safe_int(value):
    """Helper to safely convert string scores that might contain non-numeric chars."""
    try:
        # Strip non-digits like parens or whitespace
        clean = re.sub(r'[^\d]', '', str(value))
        return int(clean) if clean else 0
    except Exception:
        return 0

async def generate_and_save_croissant_metadata(row_data: dict) -> str:
    """
    Generates ML Croissant metadata for a labeled video record.
    """
    try:
        sanitized_data = {
            "id": str(row_data.get("id", "")),
            "link": str(row_data.get("link", "")),
            "visual_integrity_score": safe_int(row_data.get("visual_integrity_score")),
            "audio_integrity_score": safe_int(row_data.get("audio_integrity_score")),
            "source_credibility_score": safe_int(row_data.get("source_credibility_score")),
            "logical_consistency_score": safe_int(row_data.get("logical_consistency_score")),
            "emotional_manipulation_score": safe_int(row_data.get("emotional_manipulation_score")),
            "video_audio_score": safe_int(row_data.get("video_audio_score")),
            "video_caption_score": safe_int(row_data.get("video_caption_score")),
            "audio_caption_score": safe_int(row_data.get("audio_caption_score")),
            "final_veracity_score": safe_int(row_data.get("final_veracity_score")),
            "grounding_check": str(row_data.get("grounding_check", "")),
        }
    except Exception as e:
        logging.error(f"Data Sanitization Error before Croissant generation: {e}")
        return "N/A (Data Error)"

    try:
        video_id = sanitized_data["id"]
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        croissant_json = {
          "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "column": "http://mlcommons.org/croissant/1.0/column",
            "conformsTo": "http://dcterms.purl.org/conformsTo",
            "cr": "http://mlcommons.org/croissant/1.0/",
            "data": { "@id": "http://mlcommons.org/croissant/1.0/data", "@type": "@json" },
            "dataType": { "@id": "http://mlcommons.org/croissant/1.0/dataType", "@type": "@vocab" },
            "extract": "http://mlcommons.org/croissant/1.0/extract",
            "field": "http://mlcommons.org/croissant/1.0/field",
            "fileProperty": "http://mlcommons.org/croissant/1.0/fileProperty",
            "fileObject": "http://mlcommons.org/croissant/1.0/fileObject",
            "fileSet": "http://mlcommons.org/croissant/1.0/fileSet",
            "format": "http://mlcommons.org/croissant/1.0/format",
            "includes": "http://mlcommons.org/croissant/1.0/includes",
            "isEnumeration": "http://mlcommons.org/croissant/1.0/isEnumeration",
            "jsonPath": "http://mlcommons.org/croissant/1.0/jsonPath",
            "key": "http://mlcommons.org/croissant/1.0/key",
            "md5": "http://mlcommons.org/croissant/1.0/md5",
            "parentField": "http://mlcommons.org/croissant/1.0/parentField",
            "path": "http://mlcommons.org/croissant/1.0/path",
            "recordSet": "http://mlcommons.org/croissant/1.0/recordSet",
            "references": "http://mlcommons.org/croissant/1.0/references",
            "regex": "http://mlcommons.org/croissant/1.0/regex",
            "repeated": "http://mlcommons.org/croissant/1.0/repeated",
            "replace": "http://mlcommons.org/croissant/1.0/replace",
            "sc": "https://schema.org/",
            "separator": "http://mlcommons.org/croissant/1.0/separator",
            "source": "http://mlcommons.org/croissant/1.0/source",
            "subField": "http://mlcommons.org/croissant/1.0/subField",
            "transform": "http://mlcommons.org/croissant/1.0/transform"
          },
          "@type": "sc:Dataset",
          "name": f"vchat-label-{video_id}",
          "conformsTo": "http://mlcommons.org/croissant/1.0",
          "description": f"Veracity analysis labels for video {video_id}",
          "url": sanitized_data["link"],
          "recordSet": [
            {
              "@type": "cr:RecordSet",
              "name": "video_metadata_record",
              "description": "Factuality and veracity scores generated by VideoChat model.",
              "data": [ sanitized_data ],
              "field": [
                {"@type": "cr:Field", "name": "id", "dataType": "sc:Text", "source": {"extract": {"column": "id"}}},
                {"@type": "cr:Field", "name": "link", "dataType": "sc:URL", "source": {"extract": {"column": "link"}}},
                {"@type": "cr:Field", "name": "visual_integrity_score", "dataType": "sc:Integer", "source": {"extract": {"column": "visual_integrity_score"}}},
                {"@type": "cr:Field", "name": "audio_integrity_score", "dataType": "sc:Integer", "source": {"extract": {"column": "audio_integrity_score"}}},
                {"@type": "cr:Field", "name": "source_credibility_score", "dataType": "sc:Integer", "source": {"extract": {"column": "source_credibility_score"}}},
                {"@type": "cr:Field", "name": "logical_consistency_score", "dataType": "sc:Integer", "source": {"extract": {"column": "logical_consistency_score"}}},
                {"@type": "cr:Field", "name": "emotional_manipulation_score", "dataType": "sc:Integer", "source": {"extract": {"column": "emotional_manipulation_score"}}},
                {"@type": "cr:Field", "name": "video_audio_score", "dataType": "sc:Integer", "source": {"extract": {"column": "video_audio_score"}}},
                {"@type": "cr:Field", "name": "video_caption_score", "dataType": "sc:Integer", "source": {"extract": {"column": "video_caption_score"}}},
                {"@type": "cr:Field", "name": "audio_caption_score", "dataType": "sc:Integer", "source": {"extract": {"column": "audio_caption_score"}}},
                {"@type": "cr:Field", "name": "final_veracity_score", "dataType": "sc:Integer", "source": {"extract": {"column": "final_veracity_score"}}},
                {"@type": "cr:Field", "name": "grounding_check", "dataType": "sc:Text", "source": {"extract": {"column": "grounding_check"}}}
              ]
            }
          ]
        }

        path = Path("metadata") / f"{video_id}_{timestamp}.json"
        path.write_text(json.dumps(croissant_json, indent=2))
        return str(path)
        
    except Exception as e:
        logging.error("================ CROISSANT GENERATION ERROR ================")
        logging.error(f"Failed Data Payload: {json.dumps(sanitized_data, indent=2)}")
        logging.error(f"Exception: {e}", exc_info=True)
        logging.error("============================================================")
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
                logging.info(f"--- Raw AI Response ({model_selection}) ---\n{raw_toon}\n---------------------------------------")
            elif isinstance(msg, str): yield msg

        if not final_labels: raise RuntimeError("No labels generated.")
        
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
            
            "visual_integrity_score": vec.get("visual_integrity_score", "0"),
            "audio_integrity_score": vec.get("audio_integrity_score", "0"),
            "source_credibility_score": vec.get("source_credibility_score", "0"),
            "logical_consistency_score": vec.get("logical_consistency_score", "0"),
            "emotional_manipulation_score": vec.get("emotional_manipulation_score", "0"),
            
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
        
        with open(f"data/labels/{vid_id}_{ts}_labels.json", 'w') as f: json.dump(final_data["full_json"], f, indent=2)
        with open(f"data/labels/{vid_id}_{ts}.toon", 'w') as f: f.write(final_data["raw_toon"])
        
        row["metadatapath"] = await generate_and_save_croissant_metadata(row)
        
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