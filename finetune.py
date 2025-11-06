# finetune.py
import torch
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from my_vision_process import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "OpenGVLab/VideoChat-R1_5"
DATASET_PATH = "./data/insertlocaldataset.jsonl"
OUTPUT_DIR = "./lora_adapters"

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class MultiModalDataCollator:
    """A custom data collator to handle multimodal inputs (video + text) for the SFTTrainer."""
    processor: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            logger.warning(f"Data collator expected batch size 1, but got {len(features)}. Processing first item only.")
        
        feature = features[0]
        video_path = feature.get("video_path")
        text_prompt = feature.get("text")
        
        if not video_path or not text_prompt:
            raise ValueError("Dataset example missing 'video_path' or 'text' field.")

        # create the msg format expected by the processor
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": ""}]}]
        
        text_with_placeholder = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        final_text = text_with_placeholder + text_prompt

        image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
        
        model_inputs = self.processor(
            text=[final_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        

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


    logger.info(f"Loading base model and processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    logger.info("configuring LoRA...")
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

    logger.info(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="steps", 
        save_steps=50,         
        save_total_limit=3,    
        optim="paged_adamw_8bit",
        report_to="none",
        bf16=True, 
    )

    # init trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text", 
        max_seq_length=2048,

        # data_collator=MultiModalDataCollator(processor=processor), # Uncomment if needed
    )

 
    logger.info("fine-tuning stage")
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        potential_dirs = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir)]
        checkpoints = [d for d in potential_dirs if os.path.isdir(d) and os.path.basename(d).startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
            last_checkpoint = checkpoints[-1]
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info("Training complete.")

    fin_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(fin_path)
    logger.info(f"final LoRA adapters saved to {fin_path}")

if __name__ == "__main__":
    main()