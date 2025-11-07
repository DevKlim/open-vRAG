import torch
import re
import ast
import sys
import os
import logging
from transformers import QwenX_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from my_vision_process import process_vision_info, client

# --- Globals for model management ---
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
        raise RuntimeError("CUDA is not available. This application requires a GPU.")
    
    device = torch.device("cuda")
    logger.info(f"CUDA is available. Initializing models on {device}...")

    # --- ROBUSTNESS FIX: Check for a local model copy and ensure it's not empty ---
    local_model_path = "/app/local_model"
    # Check if the directory exists AND is not empty before trying to use it.
    # An incorrect docker-compose mount can create an empty directory, causing a crash.
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

    # Load the base model
    logger.info(f"Loading base model from {model_path}...")
    base_model = QwenX_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Base model and processor loaded successfully.")

    # Check for and load LoRA adapters
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

    # Set the default active model
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
    fps_inputs = video_kwargs['fps']

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(active_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = active_model.generate(**inputs, **generation_kwargs, use_cache=True)

    # --- FIX: Use dictionary key access `inputs['input_ids']` instead of attribute access ---
    generated_ids = [output_ids[i][len(inputs['input_ids'][i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
        yield f"--- Perception Iteration {percption + 1}/{num_perceptions} ---\n"

        if percption < num_perceptions - 1:
            current_prompt = prompts["glue"].replace("[QUESTION]", question)
        else:
            current_prompt = prompts["final"].replace("[QUESTION]", question)
        
        yield f"Prompt for this iteration:\n---\n{current_prompt}\n---\n\n"

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
    
    yield f"\n--- Final Answer ---\n{final_answer}\n"
