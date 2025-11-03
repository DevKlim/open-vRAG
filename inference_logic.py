import torch
import re
import ast
import sys
import os
import logging
import asyncio
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from my_vision_process import process_vision_info, client
try:
    import google.generativeai as genai
except ImportError:
    genai = None

#  Globals for model management 
processor = None
base_model = None
peft_model = None
active_model = None
logger = logging.getLogger(__name__)

def load_models():
    """
    Loads the base model and, if LoRA adapters exist, the fine-tuned PEFT model.
    Checks for a local model copy first before downloading from Hugging Face Hub.
    """
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
        logger.info(f"Found non-empty local model directory at '{model_path}'. Attempting to load from local files.")
    else:
        model_path = "OpenGVLab/VideoChat-R1_5"
        logger.info(f"Local model directory '{local_model_path}' not found or is empty. Will download from Hugging Face Hub.")
    
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
        torch_dtype=torch.bfloat16,
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
        yield "ERROR: 'google-generativeai' package not installed. Please install it to use Gemini models.\n"
        return

    api_key = gemini_config.get("api_key")
    model_name = gemini_config.get("model_name", "models/gemini-1.5-pro-latest")

    if not api_key:
        yield "ERROR: Gemini API Key is required. Please provide it in the UI or via URL parameters.\n"
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
            yield "Starting Gemini Factuality & Credibility pipeline...\n\n"
            prompts_to_run = []
            if checks.get("visuals"):
                prompts_to_run.append(("Visual Artifacts", "You are a digital forensics expert. Analyze this video for any signs of visual manipulation, such as AI-generated artifacts (waxy skin, strange physics), unusual editing cuts, or doctored overlays. Provide a detailed report on the video's visual coherence and authenticity."))
            if checks.get("content"):
                prompts_to_run.append(("Content & Credibility", "You are a professional fact-checker. Analyze the spoken content and context of this video. Evaluate its factual accuracy, identify potential misinformation or propaganda, and assess any noticeable political or commercial bias. Report on the credibility of the information presented."))
            if checks.get("audio"):
                 prompts_to_run.append(("Audio Anomaly Detection", "You are a media forensics analyst. Analyze the audio track of this video in conjunction with the visuals. Listen for signs of audio manipulation, such as abrupt cuts, changes in background noise that don't match the scene, or clear mismatches in lip-sync. Report on the audio's consistency and authenticity."))

            for title, prompt_text in prompts_to_run:
                yield f"--- Running '{title}' Analysis with Gemini ---\n"
                response = await loop.run_in_executor(None, lambda: model.generate_content([prompt_text, video_input], request_options={'timeout': 600}))
                yield f"===== ANALYSIS RESULT: {title.upper()} =====\n"
                yield response.text
                yield f"\n========================================\n\n"

        else: # General Q&A
            yield "Sending question to Gemini...\n"
            prompt = [question, video_input]
            response = await loop.run_in_executor(None, lambda: model.generate_content(prompt, request_options={'timeout': 600}))
            yield "\n--- Gemini's Response ---\n"
            yield response.text

    except Exception as e:
        yield f"ERROR: An error occurred during the Gemini process: {e}\n"
        logger.error("Gemini pipeline error", exc_info=True)
    finally:
        # 4. Clean up the uploaded file
        if uploaded_file:
            try:
                yield "\nCleaning up uploaded file on Google AI...\n"
                await loop.run_in_executor(None, lambda: genai.delete_file(name=uploaded_file.name))
                yield "Cleanup complete.\n"
            except Exception as e:
                yield f"Warning: Could not clean up uploaded file {uploaded_file.name}. You may need to delete it manually from the Google AI Studio file manager. Details: {e}\n"
