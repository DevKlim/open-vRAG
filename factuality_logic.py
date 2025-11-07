# factuality_logic.py
import os
import re
import json
import logging
import asyncio
from pathlib import Path
import inference_logic # Use direct import instead of relative

logger = logging.getLogger(__name__)

# --- Enhanced Prompts Requesting Structured JSON Output with Scores ---

PROMPT_VISUAL_ARTIFACTS = (
    "You are a digital forensics expert. Analyze the video for visual manipulation. Provide your findings in a structured JSON format. The JSON object must contain three keys: 'analysis', 'score', and 'justification'.\n"
    "- 'analysis' (string): A detailed report on AI artifacts (waxy skin, strange physics), editing cuts, overlays, and overall visual coherence.\n"
    "- 'score' (integer): A score from 1 (blatant, obvious manipulation) to 10 (appears completely authentic and unedited).\n"
    "- 'justification' (string): A brief sentence explaining why you gave that score.\n\n"
    "Example format: {\"analysis\": \"...detailed report...\", \"score\": 3, \"justification\": \"Score is low due to inconsistent shadows and unnatural hand gestures.\"}"
)

PROMPT_CONTENT_ANALYSIS = (
    "You are a professional fact-checker and media analyst. Analyze the video's content and transcript for accuracy, bias, and propaganda. Provide your findings in a structured JSON format with three keys: 'analysis', 'score', and 'justification'.\n"
    "- 'analysis' (string): A detailed report on factual accuracy, potential misinformation, political or commercial bias, use of propaganda techniques, and source credibility. Analyze if it creates an echo chamber.\n"
    "- 'score' (integer): A score from 1 (pure propaganda/disinformation) to 10 (objective, well-sourced, and balanced reporting).\n"
    "- 'justification' (string): A brief sentence explaining the score based on your findings (e.g., 'Score is moderate due to one-sided presentation of arguments without citing sources.').\n\n"
    "**Transcript for Context:**\n\n{transcript}\n\n"
    "Respond ONLY with the JSON object."
)

PROMPT_AUDIO_ANALYSIS = (
    "You are a media forensics analyst. Based on the video's visuals and the transcript, analyze for potential audio manipulation. Provide your findings in a structured JSON format with three keys: 'analysis', 'score', and 'justification'.\n"
    "- 'analysis' (string): A report on inferred audio issues. Check for mismatches in lip-sync, evidence of audio cuts suggested by jarring visual edits, and consistency between the transcript's tone and the visual scene.\n"
    "- 'score' (integer): A score from 1 (audio appears heavily edited or mismatched) to 10 (audio appears perfectly synchronized and consistent with visuals).\n"
    "- 'justification' (string): A brief sentence explaining the score (e.g., 'Score is high as lip movements in the transcript align perfectly with the speaker visuals.').\n\n"
    "**Transcript for Context:**\n\n{transcript}\n\n"
    "Respond ONLY with the JSON object."
)


def parse_vtt(file_path: str) -> str:
    """Parses a .vtt subtitle file and returns the clean text content."""
    try:
        if not os.path.exists(file_path):
            return "Transcript file not found."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        text_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('WEBVTT') and not '-->' in line and not line.isdigit():
                clean_line = re.sub(r'<[^>]+>', '', line)
                if clean_line and (not text_lines or clean_line != text_lines[-1]):
                     text_lines.append(clean_line)
        
        return "\n".join(text_lines) if text_lines else "No speech found in transcript."
    except Exception as e:
        logger.error(f"Error parsing VTT file {file_path}: {e}")
        return f"Error reading transcript: {e}"

async def run_factuality_pipeline(paths: dict, checks: dict, generation_config: dict):
    """
    Asynchronously runs a pipeline of factuality checks, parses JSON scores, and yields results.
    """
    video_path = paths.get("video")
    transcript_path = paths.get("transcript")

    if not video_path:
        yield "ERROR: Video path not found. Cannot start analysis.\n\n"
        return

    yield "Step 1: Processing Transcript...\n"
    await asyncio.sleep(0.1)
    transcript = "No transcript was downloaded for this video."
    if transcript_path and os.path.exists(transcript_path):
        transcript = parse_vtt(transcript_path)
        yield f"  - Transcript file found and processed.\n"
    else:
        yield f"  - No transcript file was found.\n"
    
    yield f"\n--- Extracted Transcript ---\n{transcript}\n--------------------------\n\n"
    await asyncio.sleep(0.1)

    analysis_steps = []
    if checks.get("visuals"):
        analysis_steps.append(("Visual Artifacts", PROMPT_VISUAL_ARTIFACTS))
    if checks.get("content"):
        analysis_steps.append(("Content & Credibility", PROMPT_CONTENT_ANALYSIS.format(transcript=transcript)))
    if checks.get("audio"):
        analysis_steps.append(("Audio Anomaly Detection", PROMPT_AUDIO_ANALYSIS.format(transcript=transcript)))

    for i, (title, prompt) in enumerate(analysis_steps):
        yield f"--- Step {i + 2}: Running '{title}' Analysis ---\n"
        yield "(Model is generating a structured analysis with scores. This may take a moment...)\n\n"
        await asyncio.sleep(0.1)

        try:
            current_gen_config = generation_config.copy()
            sampling_fps = current_gen_config.pop("sampling_fps", 2.0)
            current_gen_config.pop("num_perceptions", None)
            
            if current_gen_config.get("temperature", 0.0) == 0.0:
                 current_gen_config["temperature"] = 0.1 # Use a very low temp for structured output
            current_gen_config["do_sample"] = True
                
            ans = inference_logic.inference_step(
                video_path=video_path,
                prompt=prompt,
                generation_kwargs=current_gen_config,
                sampling_fps=sampling_fps,
                pred_glue=None
            )

            yield f"  - Analysis Complete for '{title}'. Parsing result...\n\n"
            
            # --- Attempt to parse JSON from the model's response ---
            parsed_result = None
            try:
                # Find JSON object within the potentially messy string
                json_match = re.search(r'\{.*\}', ans, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in the model's response.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Could not parse JSON for '{title}'. Error: {e}. Falling back to raw text.")
                yield f"===== RAW MODEL OUTPUT: {title.upper()} =====\n"
                yield f"Warning: Model did not return a valid JSON object. Displaying raw output.\n\n"
                yield f"{ans.strip()}\n"
                yield f"========================================\n\n"
                continue # Move to the next analysis step

            # --- Display the parsed, structured result ---
            score = parsed_result.get('score', 'N/A')
            justification = parsed_result.get('justification', 'No justification provided.')
            analysis_text = parsed_result.get('analysis', 'No detailed analysis provided.')
            
            yield f"===== ANALYSIS RESULT: {title.upper()} =====\n"
            yield f"SCORE: {score}/10\n"
            yield f"Justification: {justification}\n\n"
            yield f"--- Detailed Analysis ---\n{analysis_text.strip()}\n"
            yield f"========================================\n\n"

        except Exception as e:
            error_message = f"An error occurred during the '{title}' analysis step: {e}"
            logger.error(error_message, exc_info=True)
            yield f"ERROR: {error_message}\n\n"
            break
    
    yield "Factuality Analysis Pipeline Finished.\n"