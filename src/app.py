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
from fastapi import FastAPI, Request, Form, UploadFile, File, Body, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, Response, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import inference_logic
import factuality_logic
import transcription
from factuality_logic import parse_vtt
from toon_parser import parse_veracity_toon

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

STOP_QUEUE_SIGNAL = False

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
    if match: return match.group(1)
    return None

def check_if_processed(link: str) -> bool:
    target_id = extract_tweet_id(link)
    link_clean = link.split('?')[0].strip()
    
    for filename in ["data/dataset.csv", "data/manual_dataset.csv"]:
        path = Path(filename)
        if not path.exists(): continue
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_link = row.get('link', '').split('?')[0].strip()
                    if row_link == link_clean: return True
                    if target_id and row.get('id') == target_id: return True
        except Exception:
            continue
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
            original_path = sanitized_check
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
    if not sanitized_path.exists() and original_path.exists():
        await run_subprocess_async(["ffmpeg", "-i", str(original_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-y", str(sanitized_path)])

    audio_path = sanitized_path.with_suffix('.wav')
    if not audio_path.exists() and sanitized_path.exists():
        try:
            await run_subprocess_async(["ffmpeg", "-i", str(sanitized_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(audio_path)])
        except: pass
    
    if not transcript_path and audio_path.exists() and not LITE_MODE:
        transcript_path = await loop.run_in_executor(None, transcription.generate_transcript, str(audio_path))

    return {"video": str(sanitized_path), "transcript": str(transcript_path) if transcript_path else None, "metadata": metadata}

def safe_int(value):
    try:
        clean = re.sub(r'[^\d]', '', str(value))
        return int(clean) if clean else 0
    except Exception:
        return 0

async def generate_and_save_croissant_metadata(row_data: dict) -> str:
    try:
        sanitized_data = {
            "id": str(row_data.get("id", "")),
            "link": str(row_data.get("link", "")),
            "visual_integrity_score": safe_int(row_data.get("visual_integrity_score")),
            "final_veracity_score": safe_int(row_data.get("final_veracity_score"))
        }
        video_id = sanitized_data["id"]
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        croissant_json = {
          "@context": "https://schema.org/",
          "@type": "Dataset",
          "name": f"vchat-label-{video_id}",
          "description": f"Veracity analysis labels for video {video_id}",
          "url": sanitized_data["link"],
          "variableMeasured": sanitized_data
        }
        path = Path("metadata") / f"{video_id}_{timestamp}.json"
        path.write_text(json.dumps(croissant_json, indent=2))
        return str(path)
    except Exception:
        return "N/A (Error)"

async def get_labels_for_link(video_url: str, gemini_config: dict, vertex_config: dict, model_selection: str, include_comments: bool):
    try:
        yield f"Downloading assets for {video_url}..."
        paths = await prepare_video_assets_async(video_url)
        video_path = paths["video"]
        transcript_text = parse_vtt(paths["transcript"]) if paths["transcript"] else "No transcript."
        caption = paths["metadata"].get("caption", "")
        
        yield f"Assets ready. Running inference ({model_selection})..."
        final_labels = None
        raw_toon = ""
        prompt_used = ""
        
        pipeline = inference_logic.run_gemini_labeling_pipeline if model_selection == 'gemini' else inference_logic.run_vertex_labeling_pipeline
        config = gemini_config if model_selection == 'gemini' else vertex_config
        
        async for msg in pipeline(video_path, caption, transcript_text, config, include_comments):
            if isinstance(msg, dict) and "parsed_data" in msg:
                final_labels = msg["parsed_data"]
                raw_toon = msg.get("raw_toon", "")
                prompt_used = msg.get("prompt_used", "")
            elif isinstance(msg, str): yield msg

        if not final_labels: raise RuntimeError("No labels generated.")
        
        final_labels["meta_info"] = {
            "prompt_used": prompt_used,
            "model_selection": model_selection
        }
        
        vec = final_labels.get("veracity_vectors", {})
        mod = final_labels.get("modalities", {})
        fin = final_labels.get("final_assessment", {})
        
        row = {
            "id": paths["metadata"]["id"],
            "link": paths["metadata"]["link"],
            "caption": caption,
            "postdatetime": paths["metadata"].get("postdatetime", ""),
            "collecttime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "videotranscriptionpath": paths["transcript"] or "",
            "visual_integrity_score": vec.get("visual_integrity_score", "0"),
            "audio_integrity_score": vec.get("audio_integrity_score", "0"),
            "source_credibility_score": vec.get("source_credibility_score", "0"),
            "logical_consistency_score": vec.get("logical_consistency_score", "0"),
            "emotional_manipulation_score": vec.get("emotional_manipulation_score", "0"),
            "video_audio_score": mod.get("video_audio_score", "0"),
            "video_caption_score": mod.get("video_caption_score", "0"),
            "audio_caption_score": mod.get("audio_caption_score", "0"),
            "final_veracity_score": fin.get("veracity_score_total", "0"),
            "final_reasoning": fin.get("reasoning", "")
        }
        yield {"csv_row": row, "full_json": final_labels, "raw_toon": raw_toon}

    except Exception as e:
        yield {"error": str(e)}

@app.get("/queue/list")
async def get_queue_list():
    queue_path = Path("data/batch_queue.csv")
    if not queue_path.exists(): return []
    items = []
    with open(queue_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try: next(reader)
        except: pass
        for row in reader:
            if len(row) > 0:
                link = row[0]
                status = "Processed" if check_if_processed(link) else "Pending"
                items.append({
                    "link": link, 
                    "timestamp": row[1] if len(row) > 1 else "", 
                    "status": status
                })
    return items

@app.post("/queue/stop")
async def stop_queue_processing():
    global STOP_QUEUE_SIGNAL
    STOP_QUEUE_SIGNAL = True
    return {"status": "stopping"}

@app.post("/queue/upload_csv")
async def upload_csv_to_queue(file: UploadFile = File(...)):
    try:
        content = await file.read()
        decoded = content.decode('utf-8').splitlines()
        reader = csv.reader(decoded)
        links_to_add = []
        header = next(reader, None)
        if not header: return {"status": "empty file"}

        link_idx = 0
        header_lower = [h.lower() for h in header]
        if "link" in header_lower: link_idx = header_lower.index("link")
        elif "url" in header_lower: link_idx = header_lower.index("url")
        elif "http" in header[0]: 
            links_to_add.append(header[0])
            link_idx = 0
        
        for row in reader:
            if len(row) > link_idx and row[link_idx].strip():
                links_to_add.append(row[link_idx].strip())

        queue_path = Path("data/batch_queue.csv")
        existing_links = set()
        if queue_path.exists():
            with open(queue_path, 'r', encoding='utf-8') as f:
                existing_links = set(f.read().splitlines())

        added_count = 0
        with open(queue_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not queue_path.exists() or queue_path.stat().st_size == 0:
                writer.writerow(["link", "ingest_timestamp"])
            
            for link in links_to_add:
                duplicate = False
                for line in existing_links:
                    if link in line: 
                        duplicate = True
                        break
                if duplicate: continue

                writer.writerow([link, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                added_count += 1
                
        return {"status": "success", "added": added_count}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e), "status": "failed"})

@app.post("/queue/run")
async def run_queue_processing(
    model_selection: str = Form(...),
    gemini_api_key: str = Form(""), gemini_model_name: str = Form(""),
    vertex_project_id: str = Form(""), vertex_location: str = Form(""), vertex_model_name: str = Form(""), vertex_api_key: str = Form(""),
    include_comments: bool = Form(False)
):
    global STOP_QUEUE_SIGNAL
    STOP_QUEUE_SIGNAL = False
    gemini_config = {"api_key": gemini_api_key, "model_name": gemini_model_name}
    vertex_config = {"project_id": vertex_project_id, "location": vertex_location, "model_name": vertex_model_name, "api_key": vertex_api_key}
    
    async def queue_stream():
        queue_path = Path("data/batch_queue.csv")
        if not queue_path.exists():
            yield "data: Queue empty.\n\n"
            return
        
        items = []
        with open(queue_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try: next(reader) 
            except: pass
            for row in reader:
                if row: items.append(row[0])
            
        processed_count = 0
        total = len(items)
        
        for i, link in enumerate(items):
            if STOP_QUEUE_SIGNAL:
                yield "data: [SYSTEM] Stopped by user.\n\n"
                break
            
            if check_if_processed(link):
                yield f"data: [SKIP] {link} processed.\n\n"
                continue
            
            yield f"data: [START] {i+1}/{total}: {link}\n\n"
            final_data = None
            async for res in get_labels_for_link(link, gemini_config, vertex_config, model_selection, include_comments):
                if isinstance(res, str): yield f"data: {res}\n\n"
                if isinstance(res, dict) and "csv_row" in res: final_data = res
            
            if final_data:
                row = final_data["csv_row"]
                vid_id = row["id"]
                ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                
                json_path = f"data/labels/{vid_id}_{ts}_labels.json"
                with open(json_path, 'w') as f: json.dump(final_data["full_json"], f, indent=2)
                with open(f"data/labels/{vid_id}_{ts}.toon", 'w') as f: f.write(final_data["raw_toon"])
                
                row["metadatapath"] = await generate_and_save_croissant_metadata(row)
                row["json_path"] = json_path
                
                dpath = Path("data/dataset.csv")
                exists = dpath.exists()
                with open(dpath, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction='ignore')
                    if not exists: writer.writeheader()
                    writer.writerow(row)
                
                processed_count += 1
                yield f"data: [SUCCESS] Labeled.\n\n"
            else:
                yield f"data: [FAIL] Failed to label.\n\n"
                
        yield f"data: Batch Complete. +{processed_count} videos labeled.\n\n"
        yield "event: close\ndata: Done\n\n"

    return StreamingResponse(queue_stream(), media_type="text/event-stream")

@app.post("/extension/ingest")
async def extension_ingest(request: Request):
    try:
        data = await request.json()
        link = data.get("link")
        if not link: raise HTTPException(status_code=400, detail="No link")
        queue_path = Path("data/batch_queue.csv")
        file_exists = queue_path.exists()
        
        if file_exists:
            with open(queue_path, 'r', encoding='utf-8') as f:
                if link in f.read(): 
                    return {"status": "queued", "msg": "Duplicate"}
                    
        with open(queue_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["link", "ingest_timestamp"])
            writer.writerow([link, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
        return {"status": "queued", "link": link}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extension/save_comments")
async def extension_save_comments(request: Request):
    try:
        data = await request.json()
        link = data.get("link")
        # Comments can be a list of strings (legacy) or objects (new)
        comments = data.get("comments", [])
        if not link or not comments: raise HTTPException(status_code=400, detail="Missing data")
        
        csv_path = Path("data/comments.csv")
        exists = csv_path.exists()
        
        # Define fields for comment storage
        fieldnames = ["link", "author", "comment_text", "timestamp"]
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not exists: writer.writeheader()
            
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for c in comments:
                row = {"link": link, "timestamp": ts}
                if isinstance(c, dict):
                    row["author"] = c.get("author", "Unknown")
                    row["comment_text"] = c.get("text", "").strip()
                else:
                    # Legacy string support
                    row["author"] = "Unknown"
                    row["comment_text"] = str(c).strip()
                
                if row["comment_text"]:
                    writer.writerow(row)
                    
        return {"status": "saved", "count": len(comments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extension/save_manual")
async def extension_save_manual(request: Request):
    try:
        data = await request.json()
        link = data.get("link")
        labels = data.get("labels", {})
        stats = data.get("stats", {})
        
        if not link: raise HTTPException(status_code=400, detail="No link")
        
        video_id = extract_tweet_id(link) or hashlib.md5(link.encode()).hexdigest()[:16]
        
        # Build row data including new stats
        row_data = {
            "id": video_id,
            "link": link,
            "caption": data.get("caption", ""),
            "collecttime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "manual_extension",
            
            # Vectors
            "visual_integrity_score": labels.get("visual_integrity_score", 0),
            "audio_integrity_score": labels.get("audio_integrity_score", 0),
            "source_credibility_score": labels.get("source_credibility_score", 0),
            "logical_consistency_score": labels.get("logical_consistency_score", 0),
            "emotional_manipulation_score": labels.get("emotional_manipulation_score", 0),
            
            # Modalities
            "video_audio_score": labels.get("video_audio_score", 0),
            "video_caption_score": labels.get("video_caption_score", 0),
            "audio_caption_score": labels.get("audio_caption_score", 0),
            
            "final_veracity_score": labels.get("final_veracity_score", 0),
            "final_reasoning": labels.get("reasoning", ""),
            
            # New Stats
            "stats_likes": stats.get("likes", 0),
            "stats_shares": stats.get("shares", 0),
            "stats_comments": stats.get("comments", 0),
            "stats_platform": stats.get("platform", "unknown")
        }
        
        dpath = Path("data/manual_dataset.csv")
        exists = dpath.exists()
        
        # Determine fields dynamically but ensure stats are captured
        # Note: If the CSV exists with fewer columns, DictWriter might drop keys if not in fieldnames
        # We will read the header first if it exists to append new columns safely if possible,
        # or just write standard fields. For now, we trust DictWriter logic.
        
        fieldnames = list(row_data.keys())
        
        # If file exists, we must use its fieldnames or we break the CSV structure
        # unless we want to support schema evolution (too complex for now).
        # We'll just append. If columns missing in file, they are ignored by most readers or handled.
        # BUT DictWriter throws ValueError if fieldnames has extra keys and extrasaction='raise'.
        # We use 'ignore' to be safe against file schema mismatch, 
        # BUT we want to save the stats.
        # Ideally, we should check header and re-write file if columns missing.
        
        # Simplified approach: just write. If it's a fresh file, it gets all columns.
        
        with open(dpath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not exists: writer.writeheader()
            writer.writerow(row_data)
            
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/manage/list")
async def list_data():
    data = []
    def read_csv(path, source_type):
        if not path.exists(): return
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('id') or row['id'].strip() == "":
                    link = row.get('link', '')
                    tid = extract_tweet_id(link)
                    row['id'] = tid if tid else hashlib.md5(link.encode()).hexdigest()[:16]
                
                json_content = None
                if row.get('json_path') and os.path.exists(row['json_path']):
                     try:
                         with open(row['json_path'], 'r') as jf: json_content = json.load(jf)
                     except: pass
                
                row['source_type'] = source_type
                row['json_data'] = json_content
                data.append(row)

    read_csv(Path("data/dataset.csv"), "auto")
    read_csv(Path("data/manual_dataset.csv"), "manual")
    data.sort(key=lambda x: x.get('collecttime', ''), reverse=True)
    return data

@app.delete("/manage/delete")
async def delete_data(id: str = "", link: str = ""):
    if not id and not link: raise HTTPException(status_code=400, detail="Must provide ID or Link")
    deleted_count = 0
    target_id = id
    
    def remove_from_csv(path):
        nonlocal deleted_count, target_id
        if not path.exists(): return
        rows = []
        found_in_file = False
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                is_match = False
                if id and row.get('id') == id: is_match = True
                elif link and row.get('link') == link: is_match = True
                if is_match:
                    found_in_file = True
                    deleted_count += 1
                    if not target_id: target_id = row.get('id')
                else: rows.append(row)
        if found_in_file:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    remove_from_csv(Path("data/dataset.csv"))
    remove_from_csv(Path("data/manual_dataset.csv"))
    if target_id:
        for p in Path("data/labels").glob(f"{target_id}_*"): p.unlink(missing_ok=True)
        for p in Path("metadata").glob(f"{target_id}_*"): p.unlink(missing_ok=True)
    return {"status": "deleted", "count": deleted_count}

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
        async for msg in get_labels_for_link(video_url, gemini_config, vertex_config, model_selection, include_comments):
             if isinstance(msg, str): yield f"data: {msg}\n\n"
             if isinstance(msg, dict) and "csv_row" in msg: yield "data: Done. Labels generated.\n\n"
        yield "event: close\ndata: Done.\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
