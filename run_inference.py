import torch
import re
import ast
import os
import sys
import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from my_vision_process import process_vision_info, client

# --- Prompt Definitions ---
QA_THINK_GLUE = """Answer the question: "[QUESTION]" according to the content of the video. 

Output your think process within the <think> </think> tags. Use a Chain of Thought to reason step-by-step. Analyze the video content, identify key evidence, and logically deduce the answer.

Then, provide your answer within the <answer> </answer> tags. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer in the format of [(s1, e1), (s2, e2), ...]. For example: <think>...</think><answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""

QA_THINK = """Answer the question: "[QUESTION]" according to the content of the video.

Output your think process within the <think> </think> tags. Use a Chain of Thought to reason step-by-step. Analyze the video content, identify key evidence, and logically deduce the answer.

Then, provide your answer within the <answer> </answer> tags. For example: <think>...</think><answer>A</answer>.
"""

def setup_model():
    """Loads and returns the model and processor onto the GPU."""
    model_path = "OpenGVLab/VideoChat-R1_5"
    print(f"Loading model from {model_path} onto GPU...")
    
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("flash-attn is available, using 'flash_attention_2'.")
    except ImportError:
        print("flash-attn not installed. Falling back to 'sdpa' (PyTorch's native attention).")
        attn_implementation = "sdpa"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda", # Explicitly load the model on the GPU
        attn_implementation=attn_implementation
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model and processor loaded successfully onto GPU.")
    return model, processor

def inference(video_path, prompt, model, processor, max_new_tokens=2048, client=None, pred_glue=None):
    """Runs a single inference pass on the model."""
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                "video": video_path,
                'key_time': pred_glue,
                "total_pixels": 128*12 * 28 * 28, 
                "min_pixels": 128 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, client=client)
    fps_inputs = video_kwargs['fps'][0]
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def main(args):
    """Main function to orchestrate the multi-perception inference process."""
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU to run.", file=sys.stderr)
        sys.exit(1)
    print("CUDA is available. Proceeding with GPU setup.")

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at '{args.video_path}'", file=sys.stderr)
        sys.exit(1)

    model, processor = setup_model()
    
    answers = []
    pred_glue = None

    print(f"\nStarting inference for video: '{args.video_path}'")
    print(f"Question: '{args.question}'")
    print(f"Number of perception iterations: {args.num_perceptions}\n")

    for perception in range(args.num_perceptions):
        print(f"--- Perception Iteration {perception + 1}/{args.num_perceptions} ---")

        if perception < args.num_perceptions - 1:
            current_prompt = QA_THINK_GLUE.replace("[QUESTION]", args.question)
        else:
            current_prompt = QA_THINK.replace("[QUESTION]", args.question)
        
        ans = inference(
            args.video_path, current_prompt, model, processor,
            client=client, pred_glue=pred_glue
        )

        print(f"Model Raw Output: {ans}")
        answers.append(ans)
        
        pred_glue = None
        try:
            pattern_glue = r'<glue>(.*?)</glue>'
            match_glue = re.search(pattern_glue, ans, re.DOTALL)
            if match_glue:
                glue_str = match_glue.group(1).strip()
                pred_glue = ast.literal_eval(glue_str)
                print(f"Found glue for next iteration: {pred_glue}\n")
            else:
                print("No glue found for next iteration.\n")
        except Exception as e:
            print(f"Could not parse glue from output: {e}\n")
            pred_glue = None

    print("\n--- Final Answer ---")
    final_answer = answers[-1] if answers else "No answer was generated."
    print(final_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video chat inference from the command line.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("question", type=str, help="Question to ask about the video.")
    parser.add_argument("--num_perceptions", type=int, default=3, help="Number of perception iterations to run.")
    args = parser.parse_args()
    main(args)