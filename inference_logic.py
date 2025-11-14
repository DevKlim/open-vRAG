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

try:
    import google.generativeai as genai
    from google.generativeai.types import generation_types
except ImportError:
    genai = None

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    import vertexai.preview.generative_models as generative_models
except ImportError:
    vertexai = None

# Check for LITE_MODE. This will be set by DockerfileLite.
LITE_MODE = os.getenv("LITE_MODE", "false").lower() == "true"

#  Globals for model management 
processor = None
base_model = None
peft_model = None
active_model = None
logger = logging.getLogger(__name__)

def load_models():
    """
    Loads the base model and, if LoRA adapters exist, the fine-tuned PEFT model.
    This function now REQUIRES the model to be pre-downloaded in the '/app/local_model' directory.
    """
    if LITE_MODE:
        logger.info("LITE_MODE is enabled. Skipping local model loading.")
        return

    global processor, base_model, peft_model, active_model
    if base_model is not None:
        logger.info("Models already loaded.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This application requires a GPU for local models.")
    
    device = torch.device("cuda")
    logger.info(f"CUDA is available. Initializing models on {device}...")

 
    local_model_path = "/app/local_model"
    if os.path.exists(local_model_path) and os.listdir(local_model_path):
        model_path = local_model_path
        logger.info(f"Found local model directory at '{model_path}'. Loading from local files.")
    else:
        # The application will not download the model anymore. It must be provided.
        error_msg = f"FATAL: Local model not found at '{local_model_path}'. " \
                    "Please download the 'OpenGVLab/VideoChat-R1_5' model and mount it into the container. " \
                    "See the README.md for instructions."
        logger.fatal(error_msg)
        raise RuntimeError(error_msg)
    
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        logger.info("flash-attn is available, using 'flash_attention_2'.")
    except ImportError:
        logger.warning("flash-attn not installed. Falling back to 'sdpa' (PyTorch's native attention).")
        attn_implementation = "sdpa"

    logger.info(f"Loading base model from {model_path}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Base model and processor loaded successfully.")

    lora_adapter_path = "./lora_adapters/final_checkpoint"
    if os.path.exists(lora_adapter_path):
        logger.info(f"Found LoRA adapters at '{lora_adapter_path}'. Loading and merging.")
        try:
            peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            peft_model = peft_model.merge_and_unload() # Merge weights for faster inference
            peft_model.eval()
            logger.info("Successfully loaded and merged LoRA adapters.")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters: {e}", exc_info=True)
            peft_model = None
    else:
        logger.info("No LoRA adapters found. Only the default model will be available.")

    active_model = base_model
    logger.info(f"Default active model set to: Base Model")
    
    logger.info("\n" + "="*50 + "\nBASE MODEL ARCHITECTURE\n" + "="*50)
    logger.info(base_model)
    logger.info("="*50 + "\n")

def switch_active_model(model_name: str):
    """Switches the active model between 'default' and 'custom'."""
    global active_model, base_model, peft_model
    if model_name == "custom" and peft_model is not None:
        if active_model != peft_model:
            logger.info("Switching to Custom Fine-tuned Model...")
            active_model = peft_model
    else:
        if active_model != base_model:
            logger.info("Switching to Default Model...")
            active_model = base_model
    logger.info(f"Active model is now: {'Custom (Fine-tuned)' if active_model == peft_model else 'Default (Base)'}")

def inference_step(video_path, prompt, generation_kwargs, sampling_fps, pred_glue=None):
    """Runs a single inference pass on the currently active model."""
    global processor, active_model
    if active_model is None or processor is None:
        raise RuntimeError("Models are not loaded. Please call load_models first.")

    messages = [
        {"role": "user", "content": [
                {"type": "video", "video": video_path, 'key_time': pred_glue,
                 'fps': sampling_fps,
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

    logger.info("Generating response from model...")
    logger.debug(f"Generation kwargs: {generation_kwargs}")
    with torch.no_grad():
        output_ids = active_model.generate(**inputs, **generation_kwargs, use_cache=True)
    logger.info("Generation complete.")

    generated_ids = [output_ids[i][len(inputs['input_ids'][i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    logger.debug(f"Raw model output:\n---\n{output_text[0]}\n---")
    return output_text[0]

async def run_inference_pipeline(video_path, question, generation_config, prompts):
    """Runs the multi-perception inference loop."""
    pred_glue = None
    final_answer = "No answer was generated."
    
    num_perceptions = generation_config.pop("num_perceptions")
    sampling_fps = generation_config.pop("sampling_fps", 2.0) # Default to 2.0 if not provided
    if generation_config.get("temperature", 0.0) > 0.0:
        generation_config["do_sample"] = True

    for percption in range(num_perceptions):
        yield f" Perception Iteration {percption + 1}/{num_perceptions} \n"

        if percption < num_perceptions - 1:
            current_prompt = prompts["glue"].replace("[QUESTION]", question)
        else:
            current_prompt = prompts["final"].replace("[QUESTION]", question)
        
        yield f"Prompt for this iteration:\n\n{current_prompt}\n\n\n"

        ans = inference_step(
            video_path,
            current_prompt,
            generation_kwargs=generation_config,
            pred_glue=pred_glue,
            sampling_fps=sampling_fps
        )

        yield f"Model Raw Output: {ans}\n"
        final_answer = ans
        
        pred_glue = None
        try:
            pattern_glue = r'<glue>(.*?)</glue>'
            match_glue = re.search(pattern_glue, ans, re.DOTALL)
            if match_glue:
                glue_str = match_glue.group(1).strip()
                pred_glue = ast.literal_eval(glue_str)
                yield f"Found glue for next iteration: {pred_glue}\n\n"
            else:
                yield "No glue found for next iteration.\n\n"
        except Exception as e:
            yield f"Could not parse glue from output: {e}\n\n"
            pred_glue = None
    
    yield f"\n Final Answer \n{final_answer}\n"


# --- Gemini Pro Vision Pipeline ---

async def run_gemini_pipeline(video_path: str, question: str, checks: dict, gemini_config: dict):
    """
    Handles the entire process of running inference with Gemini, including
    API configuration, file upload, and response generation.
    """
    if genai is None:
        yield "ERROR: 'google-generativeai' package not installed. Please install it to use Google AI Studio models.\n"
        return

    api_key = gemini_config.get("api_key")
    model_name = gemini_config.get("model_name", "models/gemini-2.5-pro")
    

    if not api_key:
        yield "ERROR: Google AI Studio API Key is required. Please provide it in the UI.\n"
        return

    try:
        yield "Configuring Google AI client...\n"
        genai.configure(api_key=api_key)
    except Exception as e:
        yield f"ERROR: Failed to configure Google AI client. Check your API key. Details: {e}\n"
        return

    loop = asyncio.get_event_loop()
    uploaded_file = None
    try:
        # 1. Upload the video file
        yield f"Uploading video '{os.path.basename(video_path)}' to Google AI... (This may take a moment)\n"
        uploaded_file = await loop.run_in_executor(None, lambda: genai.upload_file(path=video_path))
        yield f"File uploaded. Waiting for Google to process it...\n"
        
        # 2. Poll for processing status
        while uploaded_file.state.name == "PROCESSING":
            yield "  - Processing...\r"
            await asyncio.sleep(5)
            uploaded_file = await loop.run_in_executor(None, lambda: genai.get_file(name=uploaded_file.name))

        if uploaded_file.state.name != "ACTIVE":
            yield f"ERROR: File processing failed. State: {uploaded_file.state.name}\n"
            return
        
        yield "Video is processed and ready for analysis.\n\n"
        video_input = uploaded_file
        
        # 3. Generate content
        model = genai.GenerativeModel(model_name)
        is_factuality_run = any(checks.values())

        if is_factuality_run:
            yield "Starting Google AI Studio Factuality & Credibility pipeline...\n\n"
            prompts_to_run = []
            if checks.get("visuals"):
                prompts_to_run.append(("Visual Artifacts", "You are a digital forensics expert. Analyze this video for any signs of visual manipulation, such as AI-generated artifacts (waxy skin, strange physics), unusual editing cuts, or doctored overlays. Provide a detailed report on the video's visual coherence and authenticity."))
            if checks.get("content"):
                prompts_to_run.append(("Content & Credibility", "You are a professional fact-checker. Analyze the spoken content and context of this video. Evaluate its factual accuracy, identify potential misinformation or propaganda, and assess any noticeable political or commercial bias. Report on the credibility of the information presented."))
            if checks.get("audio"):
                 prompts_to_run.append(("Audio Anomaly Detection", "You are a media forensics analyst. Analyze the audio track of this video in conjunction with the visuals. Listen for signs of audio manipulation, such as abrupt cuts, changes in background noise that don't match the scene, or clear mismatches in lip-sync. Report on the audio's consistency and authenticity."))

            for title, prompt_text in prompts_to_run:
                yield f"--- Running '{title}' Analysis with Google AI Studio ---\n"
                response = await loop.run_in_executor(None, lambda: model.generate_content([prompt_text, video_input], request_options={'timeout': 600}))
                yield f"===== ANALYSIS RESULT: {title.upper()} =====\n"
                yield response.text
                yield f"\n========================================\n\n"

        else: # General Q&A
            yield "Sending question to Google AI Studio...\n"
            prompt = [question, video_input]
            response = await loop.run_in_executor(None, lambda: model.generate_content(prompt, request_options={'timeout': 600}))
            yield "\n--- Google AI Studio's Response ---\n"
            yield response.text

    except Exception as e:
        yield f"ERROR: An error occurred during the Google AI Studio process: {e}\n"
        logger.error("Google AI Studio pipeline error", exc_info=True)
    finally:
        # 4. Clean up the uploaded file
        if uploaded_file:
            try:
                yield "\nCleaning up uploaded file on Google AI...\n"
                await loop.run_in_executor(None, lambda: genai.delete_file(name=uploaded_file.name))
                yield "Cleanup complete.\n"
            except Exception as e:
                yield f"Warning: Could not clean up uploaded file {uploaded_file.name}. You may need to delete it manually from the Google AI Studio file manager. Details: {e}\n"

# --- Vertex AI Pipeline ---
async def run_vertex_pipeline(video_path: str, question: str, checks: dict, vertex_config: dict):
    """
    Handles inference with a model on Google Cloud Vertex AI.
    """
    if vertexai is None:
        yield "ERROR: 'google-cloud-aiplatform' package not installed. Please install it to use Vertex AI models.\n"
        return

    project_id = vertex_config.get("project_id")
    location = vertex_config.get("location", "us-central1")
    model_name = vertex_config.get("model_name", "gemini-1.5-pro-preview-0409")
    api_key = vertex_config.get("api_key")
    
    if not project_id:
        yield "ERROR: Vertex AI Project ID is required. Please provide it in the UI.\n"
        return

    try:
        if api_key:
            yield "NOTE: The API Key field is not used for Vertex AI authentication with this library version.\n"
            yield "Authentication relies on Application Default Credentials (ADC).\n"

        yield f"Initializing Vertex AI for project '{project_id}' in '{location}'...\n"
        yield "If this fails, run 'docker exec -it videochat_webui gcloud auth application-default login' on your host machine.\n"
        
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)
        
        yield "Vertex AI initialized successfully.\n"

        loop = asyncio.get_event_loop()
        yield f"Reading video file '{os.path.basename(video_path)}'...\n"
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        video_part = Part.from_data(video_bytes, mime_type="video/mp4")
        yield "Video data prepared for Vertex AI.\n\n"

        is_factuality_run = any(checks.values())
        if is_factuality_run:
            yield "Starting Vertex AI Factuality & Credibility pipeline...\n\n"
            prompts_to_run = []
            if checks.get("visuals"):
                prompts_to_run.append(("Visual Artifacts", "You are a digital forensics expert. Analyze this video for any signs of visual manipulation..."))
            if checks.get("content"):
                prompts_to_run.append(("Content & Credibility", "You are a professional fact-checker. Analyze the spoken content and context of this video..."))
            if checks.get("audio"):
                 prompts_to_run.append(("Audio Anomaly Detection", "You are a media forensics analyst. Analyze the audio track of this video..."))

            for title, prompt_text in prompts_to_run:
                yield f"--- Running '{title}' Analysis with Vertex AI ---\n"
                contents = [video_part, prompt_text]
                response = await loop.run_in_executor(None, lambda: model.generate_content(contents))
                yield f"===== ANALYSIS RESULT: {title.upper()} =====\n"
                yield response.text
                yield f"\n========================================\n\n"
        
        else:
            yield "Sending question to Vertex AI...\n"
            contents = [video_part, question]
            response = await loop.run_in_executor(None, lambda: model.generate_content(contents))
            yield "\n--- Vertex AI's Response ---\n"
            yield response.text

    except Exception as e:
        yield f"ERROR: An error occurred during the Vertex AI process: {e}\n"
        yield "Authentication may have failed. Please ensure your gcloud credentials are set up correctly.\n"
        logger.error("Vertex AI pipeline error", exc_info=True)


async def run_gemini_labeling_pipeline(video_path: str, caption: str, transcript: str, gemini_config: dict, include_comments: bool):
    """
    Runs the automated labeling pipeline using Google AI Studio.
    Yields progress updates and the final parsed JSON dictionary of labels.
    """
    if genai is None:
        yield "ERROR: 'google-generativeai' package not installed. Please install it to use Google AI Studio models.\n"
        return

    api_key = gemini_config.get("api_key")
    model_name = gemini_config.get("model_name", "models/gemini-1.5-pro-latest")

    if not api_key:
        yield "ERROR: Google AI Studio API Key is required.\n"
        return

    try:
        yield "Configuring Google AI client..."
        genai.configure(api_key=api_key)
    except Exception as e:
        yield f"ERROR: Failed to configure Google AI client. Check your API key. Details: {e}\n"
        return

    loop = asyncio.get_event_loop()
    uploaded_file = None
    try:
        yield f"Uploading video '{os.path.basename(video_path)}' to Google AI... (This may take a moment)"
        uploaded_file = await loop.run_in_executor(None, lambda: genai.upload_file(path=video_path))
        yield f"File uploaded. Waiting for Google to process it..."
        
        while uploaded_file.state.name == "PROCESSING":
            yield "  - Processing...\r"
            await asyncio.sleep(5)
            uploaded_file = await loop.run_in_executor(None, lambda: genai.get_file(name=uploaded_file.name))

        if uploaded_file.state.name != "ACTIVE":
            raise RuntimeError(f"File processing failed. State: {uploaded_file.state.name}")
        
        yield "Video is processed and ready for analysis."
        video_input = uploaded_file
        
        model = genai.GenerativeModel(model_name)
        yield "Constructing prompt for Google AI Studio model..."
        
        if include_comments:
            prompt_text = LABELING_PROMPT_TEMPLATE.format(
                caption=caption,
                transcript=transcript,
                score_instructions=SCORE_INSTRUCTIONS_REASONING,
                example_json=EXAMPLE_JSON_REASONING
            )
        else:
            prompt_text = LABELING_PROMPT_TEMPLATE.format(
                caption=caption,
                transcript=transcript,
                score_instructions=SCORE_INSTRUCTIONS_SIMPLE,
                example_json=EXAMPLE_JSON_SIMPLE
            )

        yield "Sending request to Google AI Studio for analysis and labeling..."
        
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
        
        response = await loop.run_in_executor(
            None, 
            lambda: model.generate_content(
                [prompt_text, video_input],
                request_options={'timeout': 600},
                safety_settings=safety_settings
            )
        )
        
        if not response.parts:
            block_reason = (response.prompt_feedback and response.prompt_feedback.block_reason) or "Unknown"
            raise RuntimeError(f"Request blocked by Google's safety filters. Reason: {block_reason}")

        yield "Received response from Google AI Studio. Parsing JSON..."
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response.", response.text, 0)
        
        parsed_json = json.loads(json_match.group(0))
        yield "Successfully parsed JSON labels."
        yield f"--- PARSED LABELS ---\n{json.dumps(parsed_json, indent=2)}\n---------------------\n"
        yield parsed_json
            
    except Exception as e:
        yield f"ERROR: An error occurred during the Google AI Studio labeling process: {e}"
        logger.error("Google AI Studio labeling pipeline error", exc_info=True)
        yield None
    finally:
        if uploaded_file:
            try:
                yield "\nCleaning up uploaded file on Google AI..."
                await loop.run_in_executor(None, lambda: genai.delete_file(name=uploaded_file.name))
                yield "Cleanup complete."
            except Exception as e:
                yield f"Warning: Could not clean up file {uploaded_file.name}. You may need to delete it manually. Details: {e}"

async def run_vertex_labeling_pipeline(video_path: str, caption: str, transcript: str, vertex_config: dict, include_comments: bool):
    """
    Runs the automated labeling pipeline using a Vertex AI model.
    Yields progress updates and the final parsed JSON dictionary of labels.
    """
    if vertexai is None:
        yield "ERROR: 'google-cloud-aiplatform' package not installed. Please install it to use Vertex AI models.\n"
        return

    project_id = vertex_config.get("project_id")
    location = vertex_config.get("location", "us-central1")
    model_name = vertex_config.get("model_name", "gemini-1.5-pro-preview-0409")
    api_key = vertex_config.get("api_key")

    if not project_id:
        yield "ERROR: Vertex AI Project ID is required.\n"
        return

    try:
        if api_key:
            yield "NOTE: The API Key field is not used for Vertex AI authentication with this library version.\n"
            yield "Authentication relies on Application Default Credentials (ADC).\n"

        yield "Initializing Vertex AI using Application Default Credentials...\n"
        yield "If this fails, run 'docker exec -it videochat_webui gcloud auth application-default login' on your host machine.\n"
        
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)

        yield "Vertex AI initialized successfully.\n"

        loop = asyncio.get_event_loop()
        yield f"Reading video file '{os.path.basename(video_path)}'..."
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        video_part = Part.from_data(video_bytes, mime_type="video/mp4")
        yield "Video data prepared for Vertex AI."

        yield "Constructing prompt for Vertex AI model..."
        if include_comments:
            prompt_text = LABELING_PROMPT_TEMPLATE.format(
                caption=caption,
                transcript=transcript,
                score_instructions=SCORE_INSTRUCTIONS_REASONING,
                example_json=EXAMPLE_JSON_REASONING
            )
        else:
            prompt_text = LABELING_PROMPT_TEMPLATE.format(
                caption=caption,
                transcript=transcript,
                score_instructions=SCORE_INSTRUCTIONS_SIMPLE,
                example_json=EXAMPLE_JSON_SIMPLE
            )
        contents = [video_part, prompt_text]
        
        yield "Sending request to Vertex AI for analysis and labeling..."
        
        generation_config = {"response_mime_type": "application/json"}
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = await loop.run_in_executor(
            None, 
            lambda: model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        )
        
        yield "Received response from Vertex AI. Parsing JSON..."
        
        parsed_json = json.loads(response.text)
        yield "Successfully parsed JSON labels."
        yield f"--- PARSED LABELS ---\n{json.dumps(parsed_json, indent=2)}\n---------------------\n"
        yield parsed_json
            
    except Exception as e:
        yield f"ERROR: An error occurred during the Vertex AI labeling process: {e}\n"
        yield "Authentication may have failed. Please ensure your gcloud credentials are set up correctly.\n"
        logger.error("Vertex AI labeling pipeline error", exc_info=True)
        yield None
