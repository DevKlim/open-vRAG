import torch
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    QWen3VLForConditionalGeneration,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from my_vision_process import process_vision_info

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "OpenGVLab/VideoChat-R1_5"
# IMPORTANT: Create this file! See README.md for the required format.
DATASET_PATH = "./data/my_dataset.jsonl"
OUTPUT_DIR = "./lora_adapters"

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Target modules can vary, but these are common for Qwen models
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class MultiModalDataCollator:
    """A custom data collator to handle multimodal inputs (video + text) for the SFTTrainer."""
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This collator is designed for batch_size=1 due to high memory usage of video.
        if len(features) != 1:
            logger.warning(f"Data collator expected batch size 1, but got {len(features)}. Processing first item only.")
        
        feature = features[0]
        video_path = feature.get("video_path")
        text_prompt = feature.get("text")
        
        if not video_path or not text_prompt:
            raise ValueError("Dataset example missing 'video_path' or 'text' field.")

        # Create the message format expected by the processor
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": ""}]}]
        
        # We manually construct the final text input for SFTTrainer
        # The text field in the dataset should contain the full conversation turn
        # e.g., "USER: <video>\nWhat is happening? ASSISTANT: A dog is playing fetch."
        # The SFTTrainer will append this to the chat template.
        text_with_placeholder = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        final_text = text_with_placeholder + text_prompt

        # Process video separately
        image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
        
        model_inputs = self.processor(
            text=[final_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # SFTTrainer expects a 'labels' field. We use input_ids as labels for language modeling.
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"Trainable params: {trainable_params:,} || All params: {all_param:,} || "
        f"Trainable %: {100 * trainable_params / all_param:.2f}"
    )

def main():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        logger.error("Please create a JSONL file with your training data. See README.md for the format.")
        return

    # 1. Load processor and model with quantization for memory savings
    logger.info(f"Loading base model and processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = QWen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # 2. Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # 3. Load Dataset
    logger.info(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 4. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, # Increased epochs for better fine-tuning
        per_device_train_batch_size=1, # MUST be 1 due to video memory and collator design
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="steps", # Save checkpoints periodically
        save_steps=50,         # Save every 50 steps
        save_total_limit=3,    # Keep only the last 3 checkpoints
        optim="paged_adamw_8bit",
        report_to="none",
        bf16=True, # Use bfloat16 for stability and performance on modern GPUs
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text", # The SFTTrainer will handle the text part
        # The collator will handle video processing, but SFTTrainer needs the text field
        max_seq_length=2048,
        # Note: A fully custom collator like the one above might be needed if SFTTrainer
        # has issues. For now, we rely on its internal handling of the text field.
        # data_collator=MultiModalDataCollator(processor=processor), # Uncomment if needed
    )

    # 6. Start Training, resuming from a checkpoint if one exists
    logger.info("Starting fine-tuning...")
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # Get full paths of potential checkpoint directories
        potential_dirs = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir)]
        # Filter for actual directories that match the checkpoint pattern
        checkpoints = [d for d in potential_dirs if os.path.isdir(d) and os.path.basename(d).startswith("checkpoint-")]
        if checkpoints:
            # Sort by step number to find the latest
            checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
            last_checkpoint = checkpoints[-1]
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info("Training complete.")


    # 7. Save the final LoRA adapter
    final_adapter_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_adapter_path)
    logger.info(f"Final LoRA adapters saved to {final_adapter_path}")

if __name__ == "__main__":
    main()
