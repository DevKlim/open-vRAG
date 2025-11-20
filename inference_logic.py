import torch
import re
import ast
import sys
import os
import logging
import asyncio
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from my_vision_process import process_vision_info, client
from labeling_logic import (
    LABELING_PROMPT_TEMPLATE, SCORE_INSTRUCTIONS_SIMPLE, SCORE_INSTRUCTIONS_REASONING,
    EXAMPLE_JSON_SIMPLE, EXAMPLE_JSON_REASONING
)
from toon_parser import parse_veracity_toon

# Google GenAI Imports
try:
    import google.generativeai as genai_legacy
    from google.generativeai.types import generation_types
except ImportError:
    genai_legacy = None

try:
    # Modern Google GenAI SDK (v1)
    from google import genai
    from google.genai.types import (
        GenerateContentConfig,
        HttpOptions,
        Retrieval,
        Tool,
        VertexAISearch,
        GoogleSearch,
        Part
    )
    # Legacy Vertex AI for comparison or fallback if needed
    import vertexai
    from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
except ImportError:
    genai = None
    vertexai = None

# Check for LITE_MODE.
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

#  Globals for model management 
processor = None
base_model = None
peft_model = None
active_model = None
logger = logging.getLogger(__name__)

def load_models():
    """Loads the base model and, if LoRA adapters exist, the fine-tuned PEFT model."""
    if LITE_MODE:
        logger.info("LITE_MODE is enabled. Skipping local model loading.")
        return

    global processor, base_model, peft_model, active_model
    if base_model is not None: return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This application requires a GPU for local models.")
    
    device = torch.device("cuda")
    logger.info(f"CUDA is available. Initializing models on {device}...")

    local_model_path = "/app/local_model"
    if os.path.exists(local_model_path) and os.listdir(local_model_path):
        model_path = local_model_path
        logger.info(f"Found local model directory at '{model_path}'. Loading from local files.")
    else:
        error_msg = f"FATAL: Local model not found at '{local_model_path}'. Download 'OpenGVLab/VideoChat-R1_5' into ./model"
        logger.fatal(error_msg)
        raise RuntimeError(error_msg)
    
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = "sdpa"

    logger.info(f"Loading base model from {model_path}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation=attn_implementation
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Base model loaded.")

    lora_adapter_path = "./lora_adapters/final_checkpoint"
    if os.path.exists(lora_adapter_path):
        logger.info(f"Found LoRA adapters at '{lora_adapter_path}'. Loading and merging.")
        try:
            peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            peft_model = peft_model.merge_and_unload()
            peft_model.eval()
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters: {e}", exc_info=True)
            peft_model = None

    active_model = base_model
    logger.info(f"Default active model set to: Base Model")

def switch_active_model(model_name: str):
    global active_model, base_model, peft_model
    if model_name == "custom" and peft_model is not None:
        active_model = peft_model
    else:
        active_model = base_model

def inference_step(video_path, prompt, generation_kwargs, sampling_fps, pred_glue=None):
    global processor, active_model
    if active_model is None: raise RuntimeError("Models not loaded.")

    messages = [
        {"role": "user", "content": [
                {"type": "video", "video": video_path, 'key_time': pred_glue, 'fps': sampling_fps,
                 "total_pixels": 128*12 * 28 * 28, "min_pixels": 128 * 28 * 28},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, client=client)
    fps_inputs = video_kwargs['fps'][0]
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(active_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = active_model.generate(**inputs, **generation_kwargs, use_cache=True)
    
    generated_ids = [output_ids[i][len(inputs['input_ids'][i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return output_text[0]

async def run_inference_pipeline(video_path, question, generation_config, prompts):
    num_perceptions = generation_config.pop("num_perceptions")
    sampling_fps = generation_config.pop("sampling_fps", 2.0)
    if generation_config.get("temperature", 0.0) > 0.0: generation_config["do_sample"] = True
    pred_glue = None
    final_answer = "No answer."

    for percption in range(num_perceptions):
        yield f" Perception {percption + 1}/{num_perceptions} \n"
        current_prompt = prompts["glue"].replace("[QUESTION]", question) if percption < num_perceptions - 1 else prompts["final"].replace("[QUESTION]", question)
        
        ans = inference_step(video_path, current_prompt, generation_kwargs=generation_config, pred_glue=pred_glue, sampling_fps=sampling_fps)
        yield f"Model Output: {ans}\n"
        final_answer = ans
        
        match_glue = re.search(r'<glue>(.*?)</glue>', ans, re.DOTALL)
        if match_glue:
            pred_glue = ast.literal_eval(match_glue.group(1).strip())
            yield f"Found glue: {pred_glue}\n\n"
        else:
            pred_glue = None

    yield f"\n Final Answer \n{final_answer}\n"

# --- Gemini Pro Pipeline (Legacy SDK) ---
async def run_gemini_pipeline(video_path: str, question: str, checks: dict, gemini_config: dict):
    if genai_legacy is None:
        yield "ERROR: 'google-generativeai' not installed.\n"
        return

    api_key = gemini_config.get("api_key")
    model_name = gemini_config.get("model_name", "models/gemini-1.5-pro-latest")
    if not api_key: return

    try:
        genai_legacy.configure(api_key=api_key)
        loop = asyncio.get_event_loop()
        
        yield f"Uploading {os.path.basename(video_path)} to Google AI...\n"
        uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path))
        while uploaded_file.state.name == "PROCESSING":
            await asyncio.sleep(5)
            uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.get_file(name=uploaded_file.name))
        
        if uploaded_file.state.name != "ACTIVE":
            yield "File processing failed.\n"
            return

        model = genai_legacy.GenerativeModel(model_name)
        is_factuality_run = any(checks.values())

        if is_factuality_run:
            yield "Starting Factuality Pipeline (Gemini)...\n"
            prompts_to_run = []
            if checks.get("visuals"): prompts_to_run.append(("Visual Integrity", "Analyze visual components. Identify artifacts. Output TOON: visual: result[2]{score,report}:\nScore,\"Report text\""))
            if checks.get("content"): prompts_to_run.append(("Content Veracity", "Analyze spoken content. Identify claims. Output TOON: content: result[2]{score,report}:\nScore,\"Report text\""))
            if checks.get("audio"): prompts_to_run.append(("Audio Forensics", "Analyze audio. Detect robotic artifacts. Output TOON: audio: result[2]{score,report}:\nScore,\"Report text\""))

            for title, prompt_text in prompts_to_run:
                yield f"--- {title} ---\n"
                response = await loop.run_in_executor(None, lambda: model.generate_content([prompt_text, uploaded_file]))
                yield response.text + "\n\n"
        else:
            yield "Sending question...\n"
            response = await loop.run_in_executor(None, lambda: model.generate_content([question, uploaded_file]))
            yield response.text

        await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))

    except Exception as e:
        yield f"ERROR: {e}\n"

# --- Vertex AI Pipeline (New SDK) ---
async def run_vertex_pipeline(video_path: str, question: str, checks: dict, vertex_config: dict):
    """
    Uses the modern `google-genai` SDK to utilize updated Grounding/Search tools.
    """
    if genai is None:
        yield "ERROR: 'google-genai' package not installed.\n"
        return

    project_id = vertex_config.get("project_id")
    location = vertex_config.get("location", "us-central1")
    model_name = vertex_config.get("model_name", "gemini-2.5-flash") # Modern default
    
    if not project_id:
        yield "ERROR: Vertex AI Project ID is required.\n"
        return

    try:
        # Initialize Modern Client for Vertex
        yield f"Initializing Modern Client for project '{project_id}'...\n"
        client = genai.Client(vertexai=True, project=project_id, location=location)
        
        yield f"Reading video file {os.path.basename(video_path)}...\n"
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        # Create Content Part using the new SDK Part class
        video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        
        loop = asyncio.get_event_loop()
        is_factuality_run = any(checks.values())

        # Configuration with Google Search Tool (Modern Syntax)
        grounding_config = GenerateContentConfig(
            tools=[
                Tool(google_search=GoogleSearch())
            ],
            temperature=0.2,
            max_output_tokens=8192 
        )

        if is_factuality_run:
            yield "Starting Vertex AI Factuality (with Grounding)...\n\n"
            prompts_to_run = []
            if checks.get("visuals"):
                prompts_to_run.append(("Visual Artifacts", "Scan for visual manipulation. Search online for original footage. Output TOON: visual: result[2]{score,reason}:\nScore,\"Reasoning\""))
            if checks.get("content"):
                prompts_to_run.append(("Content & Credibility", "Listen to transcript. Search to verify claims. Output TOON: content: result[2]{score,citations}:\nScore,\"Citations\""))
            if checks.get("audio"):
                 prompts_to_run.append(("Audio Anomaly Detection", "Analyze audio for AI artifacts. Compare context via search. Output TOON: audio: result[2]{score,reason}:\nScore,\"Reasoning\""))

            for title, prompt_text in prompts_to_run:
                yield f"--- Running '{title}' Analysis ---\n"
                response = await loop.run_in_executor(
                    None, 
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=[video_part, prompt_text],
                        config=grounding_config
                    )
                )
                yield f"===== {title.upper()} =====\n"
                yield response.text if response.text else "No text returned."
                yield f"\n==========================\n\n"
        
        else:
            yield "Sending question to Vertex AI (with Grounding)...\n"
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[video_part, question],
                    config=grounding_config
                )
            )
            yield "\n--- Response ---\n"
            yield response.text if response.text else "No text returned."

    except Exception as e:
        yield f"ERROR: Vertex AI process failed: {e}\n"
        logger.error("Vertex AI pipeline error", exc_info=True)

async def run_gemini_labeling_pipeline(video_path: str, caption: str, transcript: str, gemini_config: dict, include_comments: bool):
    # Uses legacy SDK for AI Studio as the snippet focused on Vertex/Search tools
    if genai_legacy is None:
        yield "ERROR: Legacy SDK missing.\n"
        return
    
    api_key = gemini_config.get("api_key")
    if not api_key: return
    
    try:
        genai_legacy.configure(api_key=api_key)
        loop = asyncio.get_event_loop()
        uploaded_file = await loop.run_in_executor(None, lambda: genai_legacy.upload_file(path=video_path))
        while uploaded_file.state.name == "PROCESSING": await asyncio.sleep(2)
        
        model = genai_legacy.GenerativeModel("models/gemini-1.5-pro-latest")
        
        prompt_text = LABELING_PROMPT_TEMPLATE.format(
            caption=caption, transcript=transcript,
            score_instructions=SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE,
            example_json=EXAMPLE_JSON_REASONING if include_comments else EXAMPLE_JSON_SIMPLE
        )

        yield "Generating Labels..."
        response = await loop.run_in_executor(None, lambda: model.generate_content([prompt_text, uploaded_file], generation_config={"temperature": 0.1, "response_mime_type": "text/plain"}))
        
        parsed_data = parse_veracity_toon(response.text)
        
        # YIELD RAW TOON TEXT along with parsed data so app.py can save the file
        yield {"raw_toon": response.text, "parsed_data": parsed_data}
        
        await loop.run_in_executor(None, lambda: genai_legacy.delete_file(name=uploaded_file.name))

    except Exception as e:
        yield f"ERROR: {e}"

async def run_vertex_labeling_pipeline(video_path: str, caption: str, transcript: str, vertex_config: dict, include_comments: bool):
    """
    Uses the modern `google-genai` SDK for labeling.
    """
    if genai is None:
        yield "ERROR: 'google-genai' not installed.\n"
        return

    project_id = vertex_config.get("project_id")
    location = vertex_config.get("location", "us-central1")
    model_name = vertex_config.get("model_name", "gemini-1.5-pro")

    if not project_id: return

    try:
        client = genai.Client(vertexai=True, project=project_id, location=location)
        
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            
        video_part = Part.from_bytes(data=video_bytes, mime_type="video/mp4")

        prompt_text = LABELING_PROMPT_TEMPLATE.format(
            caption=caption, transcript=transcript,
            score_instructions=SCORE_INSTRUCTIONS_REASONING if include_comments else SCORE_INSTRUCTIONS_SIMPLE,
            example_json=EXAMPLE_JSON_REASONING if include_comments else EXAMPLE_JSON_SIMPLE
        )
        
        yield "Generating Labels (Vertex AI Modern with Grounding)..."
        
        # Using Google Search Tool for Fact-Checking
        config = GenerateContentConfig(
            temperature=0.1,
            response_mime_type="text/plain",
            max_output_tokens=8192,
            tools=[Tool(google_search=GoogleSearch())]
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: client.models.generate_content(
                model=model_name,
                contents=[video_part, prompt_text],
                config=config
            )
        )
        
        parsed_data = parse_veracity_toon(response.text)
        
        # YIELD RAW TOON TEXT along with parsed data
        yield {"raw_toon": response.text, "parsed_data": parsed_data}
            
    except Exception as e:
        yield f"ERROR: {e}"
        logger.error("Vertex Labeling Error", exc_info=True)